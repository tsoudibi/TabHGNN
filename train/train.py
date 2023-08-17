import torch.nn as nn
from data.dataset import HGNN_dataset
import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from tqdm import tqdm, trange
from model.model import TransformerDecoderModel
from model.metric import *
from utils.utils import *
import wandb


tmp_log = []
tmp__log = []
def train(model : nn.Module, 
          datset : HGNN_dataset, 
          epochs : int = 20,
          batch_size : int = 8,
          batch_size_test : int = 2,
          lr : float = 0.0001,
          K : int = 10,
          unseen_rate : float = 0.1,
          aug_strategy: str = 'random',
          evaluate_stride: int = 1,
          verbose : int = 1,
            # verbose = 0: no printed log
            # verbose = 1: print loss and AUC per train
            # verbose = 2: print loss and AUC per epoch
          wandb_log : bool = False,
            # inited outside
          log_name : str = 'unnamed',
          ):
    DEVICE = get_DEVICE()
    LABEL_POOL = datset.LABEL_POOL
    TEST_LABEL_POOL = datset.TEST_LABEL_POOL
    weight = torch.from_numpy(np.array([0.2, 1])).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_AUC = 0
    epoch_AUC_test = 0
    
    if verbose == 1:
        stepper_epoch = trange(epochs)
    else:
        stepper_epoch = range(epochs) 
    for epoch in stepper_epoch:
        import time 
        '''------------------------training------------------------'''
        print_checkpoint_time('start')
        QUERY_POOL = list(range(len(datset.FEATURE_POOL)))
        random.shuffle(QUERY_POOL)
        # train
        model.train()
        print_checkpoint_time('.train')
        # logs
        epoch_loss = 0
        # AUC_metric = BinaryAUROC().to(DEVICE)
        # AUC_metric_test = BinaryAUROC().to(DEVICE)
        train_metric = select_metric('binary_AUC', device=DEVICE, num_class=2)
        test_metric = select_metric('binary_AUC', device=DEVICE, num_class=2)
        
        torch.cuda.empty_cache()
        iter = 0
        if verbose >= 2:
            stepper = trange(len(datset.FEATURE_POOL)//batch_size)
            stepper.set_description(f"Epoch{epoch+1}/{epochs}")
        else:
            stepper = range(len(datset.FEATURE_POOL)//batch_size)
        for index in stepper: # query through all sample nodes (not infering node)
            optimizer.zero_grad()
            # pick query nodes
            query_indices = QUERY_POOL[:batch_size]
            QUERY_POOL = QUERY_POOL[batch_size:]
            print_checkpoint_time('QUERY_POOL')
            outputs = model(datset,
                            mode = 'train',
                            query_indices = query_indices,
                            K = K,
                            unseen_rate = unseen_rate,
                            aug_strategy = aug_strategy,
                            )
            print_checkpoint_time('model')
            # output shape:[q,2], example: torch.Size( 2, 2]
            # tensor([[-0.6845, -0.6323],
            #          [-0.7770, -0.4703]], device='cuda:0', grad_fn=<IndexBackward0>)
                
            # for trainning, only the query node's output is used
            # caculate loss
            LABEL_POOL_ = LABEL_POOL[query_indices] # shape:[q,2] ,example [[1. 0.], [1. 0.]]
                        
            # caculate loss
            batch_loss = criterion(outputs, torch.tensor(LABEL_POOL_,device=DEVICE))
            epoch_loss += batch_loss.item()
            # backpropagation
            batch_loss.backward()
            optimizer.step()
            print_checkpoint_time('loss 1')
            

            TRUE = np.argmax(LABEL_POOL_,axis=1)
            
            preds = np.array(outputs.softmax(dim=1).detach().cpu()).tolist()
            # the probability of the query node is 1 (from model output)
            train_metric.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
            # AUC_metric.update(torch.tensor(pred_prob_of_is_1,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
            iter += 1
            # if iter >= 100:
            #     break
            print_checkpoint_time('loss 2')
            if index == len(datset.FEATURE_POOL)//batch_size -1 and verbose >= 2:
                stepper.set_postfix({train_metric.name :float(train_metric.compute())})
                stepper.update()
                
        torch.cuda.empty_cache()
        epoch_loss = epoch_loss / batch_size
        epoch_AUC = float(train_metric.compute()) 
        train_metric.reset()
        if verbose == 1:
            stepper_epoch.set_postfix({'loss_train':epoch_loss, 'AUC_train':epoch_AUC, 'AUC_test':epoch_AUC_test})
        
        '''------------------------evaluate------------------------'''
        if (epoch+1) % evaluate_stride == 0:
            # evaluate
            model.eval()
            iter = 0
            with torch.no_grad():
                if verbose >= 2:
                    stepper = trange(len(datset.TEST_POOL)//batch_size_test)
                else:
                    stepper = range(len(datset.TEST_POOL)//batch_size_test)
                    
                    
                QUERY_POOL = list(range(len(datset.TEST_POOL)))
                random.shuffle(QUERY_POOL)
                
                for index in stepper:
                    # get the last 'batch_size_test' nodes as query nodes
                    query_indices = QUERY_POOL[:batch_size_test]
                    QUERY_POOL = QUERY_POOL[batch_size_test:]
                    print_checkpoint_time('QUERY_POOL')

                    outputs = model(datset, mode = 'inferring', query_indices = query_indices, K = None)
                    print_checkpoint_time('infering')
                    LABEL_POOL_ = TEST_LABEL_POOL[query_indices]
                    TRUE = np.argmax(LABEL_POOL_,axis=1)
                    # outputs = outputs.softmax(dim=1)
                    # pred_prob_of_is_1 = [probs[1] for probs in outputs] 
                    preds = np.array(outputs.softmax(dim=1).detach().cpu()).tolist()
                    
                    # AUC_metric_test.update(torch.tensor(pred_prob_of_is_1,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
                    test_metric.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
                    print_checkpoint_time('loss')
                    iter += 1
                    # if iter >= 3:
                    #     break
            torch.cuda.empty_cache()
            epoch_AUC_test = float(test_metric.compute()) 
            # epoch_AUC_test = float(AUC_metric_test.compute()) 
            if verbose == 1:
                stepper_epoch.set_postfix({'loss_train':epoch_loss, 'AUC_train':epoch_AUC, 'AUC_test':epoch_AUC_test})

            # AUC_metric.reset()
            test_metric.reset()
            # break
            # del AUC_metric_test, AUC_metric
            tmp_log.append(float(epoch_loss))
            tmp__log.append(float(epoch_AUC))

            
            # print(f"Epoch{epoch+1}/{epochs} | Loss: {epoch_loss} | AUC: {epoch_AUC} |")
            if verbose >= 2:
                print(f"Epoch{epoch+1}/{epochs} | Loss: {epoch_loss} | AUC_train: {epoch_AUC} | AUC_test: {epoch_AUC_test}")
        
        
        with open('logs/' + log_name + '.txt', 'a') as f:
            # f.write(f"Epoch{epoch+1}/{epochs} | Loss: {epoch_loss} | AUC: {epoch_AUC}| ")
            f.write(f"Epoch{epoch+1}/{epochs} | Loss: {epoch_loss} | AUC_train: {epoch_AUC}| AUC_test: {epoch_AUC_test}\n ")
        if wandb_log:
            wandb.log({'loss': epoch_loss, 'AUC_train': epoch_AUC, 'AUC_test': epoch_AUC_test, 'epoch': epoch})
    if verbose >=1:
        print(f"{log_name} | Loss: {epoch_loss} | AUC_train: {epoch_AUC} | AUC_test: {epoch_AUC_test}")

