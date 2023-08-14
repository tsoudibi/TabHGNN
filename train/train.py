import torch.nn as nn
from data.dataset import HGNN_dataset
import torch
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from tqdm import tqdm, trange
from model.model import TransformerDecoderModel
from utils.utils import set_seed, get_DEVICE
import wandb

# training
from torch import autograd
from torcheval.metrics.aggregation.auc import AUC
from torcheval.metrics import BinaryAUROC
from sklearn.metrics import roc_auc_score
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
        
        '''------------------------training------------------------'''
        
        QUERY_POOL = list(range(len(datset.FEATURE_POOL)))
        random.shuffle(QUERY_POOL)
        # train
        model.train()
        # logs
        loss_log = 0
        AUC_metric = BinaryAUROC().to(DEVICE)
        AUC_metric_test = BinaryAUROC().to(DEVICE)
        
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

            outputs = model(datset,
                            mode = 'train',
                            query_indices = query_indices,
                            K = K,
                            unseen_rate = unseen_rate,
                            aug_strategy = aug_strategy,
                            )
            
            # output shape:[q,2], example: torch.Size( 2, 2]
            # tensor([[-0.6845, -0.6323],
            #          [-0.7770, -0.4703]], device='cuda:0', grad_fn=<IndexBackward0>)
                
            # for trainning, only the query node's output is used
            # caculate loss
            LABEL_POOL_ = LABEL_POOL[query_indices] # shape:[q,2] ,example [[1. 0.], [1. 0.]]
                        
            # caculate loss
            batch_loss = criterion(outputs, torch.tensor(LABEL_POOL_,device=DEVICE))
            loss_log += batch_loss.item()
            # backpropagation
            batch_loss.backward()
            optimizer.step()

            TRUE = np.argmax(LABEL_POOL_,axis=1)
            
            outputs = outputs.softmax(dim=1)
            pred_prob_of_is_1 = [probs[1] for probs in outputs] 
            # the probability of the query node is 1 (from model output)
            
            AUC_metric.update(torch.Tensor(pred_prob_of_is_1),torch.Tensor(TRUE))
            torch.cuda.empty_cache()
            iter += 1
            # if iter >= 100:
            #     break
            if index == len(datset.FEATURE_POOL)//batch_size -1 and verbose >= 2:
                stepper.set_postfix(AUC=float(AUC_metric.compute()))
                stepper.update()
                
        epoch_loss = loss_log / batch_size
        epoch_AUC = float(AUC_metric.compute()) 
        AUC_metric.reset()
        if verbose == 1:
            stepper_epoch.set_postfix({'loss_train':epoch_loss, 'AUC_train':epoch_AUC, 'AUC_test':epoch_AUC_test})
        
        '''------------------------evaluate------------------------'''
        if (epoch+1) % 50 == 0:
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

                    outputs = model(datset, mode = 'inferring', query_indices = query_indices, K = None)
                    LABEL_POOL_ = TEST_LABEL_POOL[query_indices]
                    TRUE = np.argmax(LABEL_POOL_,axis=1)
                    outputs = outputs.softmax(dim=1)
                    pred_prob_of_is_1 = [probs[1] for probs in outputs] 
                    AUC_metric_test.update(torch.Tensor(pred_prob_of_is_1),torch.Tensor(TRUE))
                    torch.cuda.empty_cache()
                    iter += 1
                    # if iter >= 3:
                    #     break
            epoch_AUC_test = float(AUC_metric_test.compute()) 
            if verbose == 1:
                stepper_epoch.set_postfix({'loss_train':epoch_loss, 'AUC_train':epoch_AUC, 'AUC_test':epoch_AUC_test})

            AUC_metric.reset()
            AUC_metric_test.reset()
            # break
            del loss_log, AUC_metric
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

