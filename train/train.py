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

debug_array = np.array([])
debug_array_true = np.array([])
def train(model : nn.Module, 
          datset : HGNN_dataset, 
          metric : str,
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
          tensorboard_log : bool = False,
          writer = None,
          logger = None,
          log_name : str = 'unnamed',
          ):
    DEVICE = get_DEVICE()
    if get_task() == 'classification':
        LABEL_POOL = datset.LABEL_POOL
        TEST_LABEL_POOL = datset.TEST_LABEL_POOL
    elif get_task() == 'regression':
        LABEL_POOL = datset.ORIGINAL_LABEL_POOL
        TEST_LABEL_POOL = datset.ORIGINAL_TEST_LABEL_POOL
    if get_task() == 'classification':
        # weight = torch.from_numpy(np.array([0.2, 1])).float().to(DEVICE)
        # criterion = nn.CrossEntropyLoss(weight)
        criterion = nn.CrossEntropyLoss()
    elif get_task() == 'regression':
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
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
        # available_metrics = ['binary_AUC', 'ACC', 'R2']
        train_metric_ACC = select_metric('ACC', device=DEVICE)
        train_metric_AUC = select_metric('binary_AUC', device=DEVICE)
        test_metric_ACC = select_metric('ACC', device=DEVICE)
        test_metric_AUC = select_metric('binary_AUC', device=DEVICE)
        
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
            LABEL_POOL_ = LABEL_POOL[query_indices] # shape:[B,L] ,example [[1. 0.], [1. 0.]]
            # caculate loss
            batch_loss = criterion(outputs, torch.tensor(LABEL_POOL_,device=DEVICE,dtype=torch.float32))
            epoch_loss += batch_loss.item()
            # backpropagation
            batch_loss.backward()
            optimizer.step()
            print_checkpoint_time('loss 1')
            
            if get_task() == 'classification':
                TRUE = np.argmax(LABEL_POOL_,axis=1)
                preds = np.array(outputs.softmax(dim=1).detach().cpu()).tolist()
            elif get_task() == 'regression':
                TRUE = LABEL_POOL_
                preds = np.array(outputs.detach().cpu()).tolist()
            
            # the probability of the query node is 1 (from model output)
            # train_metric.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
            train_metric_ACC.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
            train_metric_AUC.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
            print_checkpoint_time('loss 2')
            if index == len(datset.FEATURE_POOL)//batch_size -1 and verbose >= 2:
                # stepper.set_postfix({train_metric.name :float(train_metric.compute())})
                stepper.set_postfix({train_metric_ACC.name :float(train_metric_ACC.compute()),
                                     train_metric_AUC.name :float(train_metric_AUC.compute())
                                     })
                stepper.update()
                
        torch.cuda.empty_cache()
        epoch_loss = epoch_loss / batch_size
        # epoch_metric = float(train_metric.compute()) 
        epoch_metric_ACC = float(train_metric_ACC.compute()) 
        epoch_metric_AUC = float(train_metric_AUC.compute()) 
        train_metric_ACC.reset()
        train_metric_AUC.reset()
        if verbose == 1:
            stepper_epoch.set_postfix({'loss_train':epoch_loss,
                                       train_metric_ACC.name+'_train':epoch_metric_ACC,
                                       train_metric_ACC.name+'_test':epoch_metric_test_ACC,
                                       train_metric_AUC.name+'_train':epoch_metric_AUC,
                                       train_metric_AUC.name+'_test':epoch_metric_test_AUC,
                                       })
        
        '''------------------------evaluate------------------------'''
        if (epoch+1) % evaluate_stride == 0: # evaluate_stride
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

                    outputs = model(datset, 
                                    mode = 'inferring', 
                                    query_indices = query_indices, 
                                    K = None)
                    print_checkpoint_time('infering')
                    LABEL_POOL_ = TEST_LABEL_POOL[query_indices]
                    
                    if get_task() == 'classification':
                        TRUE = np.argmax(LABEL_POOL_,axis=1)
                        preds = np.array(outputs.softmax(dim=1).detach().cpu()).tolist()
                        global debug_array, debug_array_true
                        debug_array = np.append(debug_array,preds)
                        debug_array_true = np.append(debug_array_true,TRUE)
                    elif get_task() == 'regression':
                        TRUE = LABEL_POOL_
                        preds = np.array(outputs.detach().cpu()).tolist()
                    
                    test_metric_ACC.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
                    test_metric_AUC.update(torch.tensor(preds,device=DEVICE),torch.tensor(TRUE,device=DEVICE))
                    print_checkpoint_time('loss')

            torch.cuda.empty_cache()
            epoch_metric_test_ACC = float(test_metric_ACC.compute()) 
            epoch_metric_test_AUC = float(test_metric_AUC.compute()) 
            if verbose == 1:
                stepper_epoch.set_postfix({'loss_train':epoch_loss,
                                           train_metric_ACC.name+'_train':epoch_metric_ACC,
                                           train_metric_ACC.name+'_test':epoch_metric_test_ACC,
                                           train_metric_AUC.name+'_train':epoch_metric_AUC,
                                           train_metric_AUC.name+'_test':epoch_metric_test_AUC,
                                           })


            test_metric_ACC.reset()
            test_metric_AUC.reset()
            # break
            # del AUC_metric_test, AUC_metric
            np.save('debug/credit'+str(epoch)+'.npy', [debug_array])
            np.save('debug/credit'+str(epoch)+'_true.npy', [debug_array_true])
            debug_array = np.array([])
            debug_array_true = np.array([])
            
            if verbose >= 2:
                print(f"Epoch{epoch+1}/{epochs} | Loss: {round(epoch_loss,3)} | {train_metric_ACC.name}_train: {round(epoch_metric_ACC,3)} | {train_metric_ACC.name}_test: {round(epoch_metric_test_ACC,3)} | {train_metric_AUC.name}_train: {round(epoch_metric_AUC,3)} | {train_metric_AUC.name}_test: {round(epoch_metric_test_AUC,3)}")
        
        
        if wandb_log:
            wandb.log({'loss': epoch_loss,
                       train_metric_ACC.name+'_train': epoch_metric_ACC,
                       train_metric_ACC.name+'_test': epoch_metric_test_ACC,
                       train_metric_AUC.name+'_train': epoch_metric_AUC,
                       train_metric_AUC.name+'_test': epoch_metric_test_AUC,
                       'epoch': epoch})
        if tensorboard_log:
            writer.add_scalar('loss', epoch_loss, epoch)
            writer.add_scalar(train_metric_ACC.name+'_train', epoch_metric_ACC, epoch)
            writer.add_scalar(train_metric_ACC.name+'_train', epoch_metric_ACC, epoch)
            writer.add_scalar(train_metric_AUC.name+'_test', epoch_metric_test_AUC, epoch)
            writer.add_scalar(train_metric_AUC.name+'_test', epoch_metric_test_AUC, epoch)
            writer.flush()
        if logger is not None:
            logger.update(epoch, epoch_loss, epoch_metric_ACC, epoch_metric_test_ACC,
                          epoch_metric_AUC, epoch_metric_test_AUC)
            logger.save()
            
    if logger is not None:
        name, best_loss, best_train_metric, best_test_metric, best_epoch = logger.get_best(0)
        name_, best_loss_, best_train_metric_, best_test_metric_, best_epoch_ = logger.get_best(1)
        print(f"best_loss: {best_loss} | best_{name}_train: {best_train_metric} | best_{name}_test: {best_test_metric} | best_epoch: {best_epoch}")
        print(f"best_loss: {best_loss_} | best_{name_}_train: {best_train_metric_} | best_{name_}_test: {best_test_metric_} | best_epoch: {best_epoch_}")
    if verbose >=1:
        pass
