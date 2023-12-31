import pandas as pd
import numpy as np
import math
import random
import torch
from torch import Tensor
from typing import Optional, Tuple
from data.preprocess import POOL_preprocess, POOL_preprocess_inference
from utils.utils import *



class HGNN_dataset():
    def __init__(self,
                 data_df : pd.DataFrame,
                 label_column : str,
                 test_df : pd.DataFrame = None,
                 split_ratio : float = None,
                 embedding_dim : int = 128,
                 N_BINS : int = 100
                 ):
        DEVICE = get_DEVICE()
        if test_df is None:
            # shuffle and cut data
            data_df = data_df.sample(frac=1,random_state=42).reset_index(drop=True)
            test_size = math.ceil(data_df.shape[0] * (1-split_ratio))
            train_pool = data_df[test_size:]
            test_pool = data_df[:test_size]
            print('total data num:' , data_df.shape[0])
            print('trian data num:' , train_pool.shape[0])
            print('test data num:' , test_pool.shape[0])
        else: 
            # given train and test data, seperated (K-fold)
            train_pool = data_df
            test_pool = test_df
            print('trian data num:' , train_pool.shape[0])
            print('test data num:' , test_pool.shape[0])

        
        # to-dos:
        # train
        #   
        TRAIN_POOL, self.inference_package, self.NUM_vs_CAT, C_POOL = POOL_preprocess(train_pool, N_BINS = N_BINS)
        TEST_POOL, self.unseen_node_indexs_C, self.TEST_POOL_VALUES = POOL_preprocess_inference(test_pool, self.inference_package)
        LABEL_COLUMN = label_column

        # cut feature and lable
        FEATURE_POOL = TRAIN_POOL.drop(LABEL_COLUMN, axis=1)
        LABEL_POOL = TRAIN_POOL[LABEL_COLUMN]
        TEST_LABEL_POOL = TEST_POOL[LABEL_COLUMN]
        
        
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        if get_task() == 'regression':
            self.ORIGINAL_LABEL_POOL = LABEL_POOL.to_numpy().reshape(-1,1)
            self.ORIGINAL_TEST_LABEL_POOL = TEST_LABEL_POOL.to_numpy().reshape(-1,1)
        LABEL_POOL = enc.fit_transform(LABEL_POOL.values.reshape(-1,1)).toarray()
        TEST_LABEL_POOL = enc.fit_transform(TEST_LABEL_POOL.values.reshape(-1,1)).toarray()
        # L: number of lable nodes, the last node of Lable nodes is served as unknown lable node
        L = LABEL_POOL.shape[1] + 1

        # S: number of sample nodes, the last node of sample nodes is served as infering node
        S = FEATURE_POOL.shape[0] + 1
        
        # F: number of field (column) nodes
        F = FEATURE_POOL.shape[1]

        # C: number of catagory nodes, each field(column) has its own "unseen" catagory nodes
        self.nodes_of_fields = []
        for column in FEATURE_POOL.columns:
            self.nodes_of_fields.append(FEATURE_POOL[column].nunique()+1)
        C = sum(self.nodes_of_fields) # the total number of nodes equals to the sum of nodes of each field

        nodes_num = {'L':L, 'S':S, 'C':C, 'F':F}
        print('node_nums', nodes_num)
        
        # get samples indexs for each label
        self.labe_to_index = {}
        tmp_pool = TRAIN_POOL.copy().reset_index(drop=True)
        if isinstance(LABEL_COLUMN, list):
            # LABEL_COLUMN = LABEL_COLUMN[0]
            raise ValueError('ERROR: multiple label columns are not supported yet')
        for label in tmp_pool[LABEL_COLUMN].unique():
            self.labe_to_index[label] = (tmp_pool[tmp_pool[LABEL_COLUMN] == label].index).tolist()
        
        self.TRAIN_POOL = TRAIN_POOL
        self.TEST_POOL = TEST_POOL
        self.TEST_LABEL_POOL = TEST_LABEL_POOL
        self.LABEL_COLUMN = LABEL_COLUMN
        self.FEATURE_POOL = FEATURE_POOL
        self.LABEL_POOL = LABEL_POOL
        self.C_POOL = C_POOL   
        self.nodes_num = nodes_num
        self.N_BINS = N_BINS
        self.embedding_dim = embedding_dim
        
        self.make_input_tensor()
        # self.get_sample(10)        
        self.make_mask_all()
        
        # self.make_mask()
        
        
    def make_mask_subgraph(self,
                  sample_indices: Optional[list] = None,
                  query_indices: Optional[list] = None,
                ):
        '''Makeing masks for subgraph. Mask values are 1 if two nodes are connected, otherwise 0.
        
        Args:
            sample_indices: list of list of sample node indices, in shape of `[batch_size, sample_size]`
            query_indices: list of query node indices for each batch, in shape of `[batch_size]`
        
        for example, with:
            {'L': 3, 'S': 39074, 'C': 470, 'F': 14, 'K': 10}
            
        the masks will be:
            masks['L2S'] = torch.Size([16, 8]), values in torch.Size([10, 3])\\
            masks['S2C'] = torch.Size([472, 16]), values in torch.Size([470, 10])\\
            masls['C2F'] = torch.Size([16, 472]), values in torch.Size([14, 470])\\
        Notice: xformer require the mask's tensor must align on memory, and should be slice of a tensor if shape cannot be divided by 8
        '''
        sample_size = len(sample_indices[0])
        # caculate masking
        masks = {}
        masks['L2S'] = self.MASKS_FULL['L2S'][sample_indices]
        masks['S2C'] = self.MASKS_FULL['S2C'][:,sample_indices].permute(1,0,2)
        masks['C2F'] = self.MASKS_FULL['C2F'].repeat(len(query_indices),1,1)
        self.MASKS = masks
        self.nodes_num['K'] = sample_size
        
    def make_mask_all(self):
        '''Makeing masks for the entire graph. Mask values are 1 if two nodes are connected, otherwise 0.

        for example, with:
            {'L': 3, 'S': 39074, 'C': 470, 'F': 14, 'K': 10}.
            
        the masks will be:
            masks['L2S']: torch.Size([39080, 8]), values in torch.Size([39074, 3]).\\
            masks['S2C']: torch.Size([472, 39080]), values in torch.Size([470, 39074]).\\
            masls['C2F']: torch.Size([16, 472]), values in torch.Size([14, 470]).\\
            
        Notice: xformer require the mask's tensor must align on memory, and should be slice of a tensor if shape cannot be divided by 8
        '''
        DEVICE = get_DEVICE()
        L, S, C, F = self.nodes_num['L'], self.nodes_num['S'], self.nodes_num['C'], self.nodes_num['F']
        # caculate masking
        masks = {}
        
        # label to sample 
        tmp = torch.zeros([math.ceil(S/8) * 8, math.ceil(L/8) * 8], dtype=torch.float, device=DEVICE)
        label_ids = sorted(self.TRAIN_POOL[self.LABEL_COLUMN].unique())
        for i, value_df in enumerate(self.TRAIN_POOL[self.LABEL_COLUMN]):
            for j, value_label in enumerate(label_ids):
                if value_label == value_df:
                    tmp[i][j] = 1
                    break
        masks['L2S'] = tmp

        # sample to catagory
        tmp = torch.zeros([math.ceil(C/8) * 8, math.ceil(S/8) * 8], dtype=torch.float, device=DEVICE).T
        tmp_df = self.FEATURE_POOL
        tmp[torch.arange(len(self.FEATURE_POOL), device=DEVICE).unsqueeze(-1), torch.tensor(tmp_df.values, device=DEVICE)] = 1
        tmp = tmp.T.contiguous()
        masks['S2C'] = tmp

        # catagory to field
        # to do : this is wrong , should connect all catagory nodes (even unseen nodes))
        tmp = torch.zeros([math.ceil(F/8) * 8, math.ceil(C/8) * 8], dtype=torch.float, device=DEVICE)
        unique_items = [sorted(self.FEATURE_POOL[column].unique()) for column in (self.FEATURE_POOL.columns)]
        for i in range(F):
            for j in (unique_items[i]):
                tmp[i][j] = 1
        masks['C2F'] = tmp
        self.MASKS = masks
        self.MASKS_FULL = masks
        
    def make_mask_test(self, 
                       indexs_in_test_pool : list 
                       ):
        '''Make mask tensor for the testing scenario. \n
        In testing scenario, L, S, C, F remain the same, while all INPUTs are the same (sience they are initialized fixed vlaues\n
        All we need to do is to update masks(L2S, S2C) for the new inference node
        Args:
            indexs_in_test_pool: list of indexs of nodes in test pool, which are the query nodes for inference
                                that is, query_indices.
        '''
        S = self.nodes_num['S']
        
        masks = {}

        # for i query nodes
        C_connection = self.TEST_POOL.drop(self.LABEL_COLUMN, axis=1).values

        masks['L2S'] = self.MASKS_FULL['L2S'].clone().detach()
        masks['L2S'][S-1, :] = 1 # connect to all Label nodes
        masks['L2S'] = masks['L2S'].repeat(len(indexs_in_test_pool),1,1) # repeat for batch
        
        def edit_S2C(index):
            masks['S2C'][index, C_connection[indexs_in_test_pool[index]],S-1] = 1  
        masks['S2C'] = self.MASKS_FULL['S2C'].clone().detach().repeat(len(indexs_in_test_pool),1,1)
        _ = list(map(edit_S2C, range(len(indexs_in_test_pool))))

        # C2F remains the same
        masks['C2F'] = self.MASKS_FULL['C2F'].repeat(len(indexs_in_test_pool),1,1)
        
        self.MASKS = masks

        
        
    def make_input_tensor(self):
        '''Makeing input tensor for the entire graph.
            
        for example, with:
            {'L': 3, 'S': 39074, 'C': 470, 'F': 14, 'K': 10}.
                
        the input tensor will be:
            L_input: torch.Size([3, 1]).
            S_input: torch.Size([39074, 128]).
            C_input: torch.Size([470, 1]).
            F_input: torch.Size([14, 1]).
        '''
        DEVICE = get_DEVICE()
        # make input tensor
        L, S, C, F = self.nodes_num['L'], self.nodes_num['S'], self.nodes_num['C'], self.nodes_num['F']
        # L
        L_input = torch.tensor([range(L)], device=DEVICE).reshape(-1,1)
        # S (initialize by random)
        S_input = torch.rand(self.embedding_dim, device=DEVICE).repeat(S,1)
        # C 
        C_input = torch.tensor(np.array([self.C_POOL]), device=DEVICE).reshape(-1,1)
        # F 
        F_input = torch.tensor([range(F)], device=DEVICE).reshape(-1,1)
        # 
        self.INPUTS = (L_input, S_input, C_input, F_input)
        self.INPUT_DIMS = (L_input.size(1), S_input.size(1), C_input.size(1), F_input.size(1))
        
    def get_sample(self, sample_size, query_indices = []):
        '''get sample nodes indices, and update mask and input tensor
        
        Args:
            sample_size: number of sample nodes required for each batch.
            query_indices (optional): list of nodes indices that must be included in nodes indices (one for each batch).
        Return:
            sample_indices: list of sample nodes indices, in shape of `[batch_size, sample_size]`
        
        For example, with `sample_size = 3`, `query_indices = [1,2,3]`\n
        means that there are `batch_size = 3` batches, each batch has `3` nodes.\n
        particularly, the three batches' `sample_indices` could be:\n
        `[1,324,656]`, `[2, 435, 9867]`, `[3, 789, 1343]`
        
        The included nodes shold not and will not be repeated, in case of the lable leakage.
        '''
            
        # torch.rand + argsort 
        query_indices_tensor = torch.tensor(query_indices,device=get_DEVICE()).unsqueeze(-1)
        random_indices_tensor = torch.rand(len(query_indices),len(self.TRAIN_POOL),device=get_DEVICE()).argsort(dim=-1)[:,:sample_size-1]
        while (random_indices_tensor == query_indices_tensor.view(-1, 1)).any():
            random_indices_tensor = torch.where(random_indices_tensor == query_indices_tensor.view(-1, 1), torch.randint(0,len(self.TRAIN_POOL),(1,),device=get_DEVICE()), random_indices_tensor)
        sample_indices = np.array(torch.cat((query_indices_tensor, random_indices_tensor), dim=1).tolist())
        
        # update mask
        # modify input tensor
        L_input, S_input, C_input, F_input = self.INPUTS
        S_input_masked = S_input[sample_indices]
        self.MASKED_INPUTS = (L_input, S_input_masked, C_input, F_input) 
          
        return sample_indices

    def random_connect_unseen(self, 
                              mask, 
                              mask_ratio = 0.1,
                              strategy: Optional[str] = 'unseen',
                              ):
        '''
        Randomly connect nodes to unseen nodes.\\
        Args:
            mask: mask tensor to be updated, shape of `[B, C, S]`
            mask_ratio: the ratio of nodes to be connected to unseen nodes
            
        By changing the mask into `SpareTensor` and updating the indices, we can randomly connect nodes to unseen nodes
        '''
        if mask_ratio <= 0:
            return mask
        DEVICE = get_DEVICE()
        unseen_node_indexs_list = list(self.unseen_node_indexs_C.values())
        mask_sparse = torch.clone(mask).to_sparse()
        indices = torch.clone(mask_sparse.indices()[1])
        num_batches = mask_sparse.indices()[0].max()+1
        edges_per_batch = indices.shape[0] // num_batches
        
        
        replacement_table = None
        if strategy == 'unseen':
            # create replacement table, which is like...    
            # [74, 74, 74, ........74,
            #  128,128,128,........128,
            #  .......................] as a tensor
            replacement_table = [[unseen_index]*(mask_sparse.indices()[2].max()+1) for unseen_index in unseen_node_indexs_list]
            replacement_table = torch.tensor(replacement_table, device=DEVICE).flatten()
        elif strategy == 'random':
            def get_random_int(lower_bound, unseen_node_index):
                return torch.randint(lower_bound, unseen_node_index+1)
            replacement_table = []
            lower_bound = 0
            for unseen_node_index in unseen_node_indexs_list:
                replacement_table.append(torch.randint(lower_bound, unseen_node_index+1, (mask_sparse.indices()[2].max()+1,),device=DEVICE))
                lower_bound = unseen_node_index+1
            replacement_table = torch.stack(replacement_table, dim = 0).flatten()
        else:
            raise ValueError('strategy should be either unseen or random')
        
        for batch in range(num_batches):
            this_batch_indices = indices[batch*edges_per_batch:(batch+1)*edges_per_batch]
            num_samples = (torch.rand(len(this_batch_indices), device=DEVICE) < mask_ratio).int() * torch.add(torch.arange(len(this_batch_indices), device=DEVICE),1)
            to_be_replace = torch.add(num_samples[num_samples.nonzero()].squeeze(-1), -1 )
            # to_be_replace is a mask of indices to be replaced

            indices[to_be_replace + batch*edges_per_batch] = replacement_table[to_be_replace]

        mask_sparse.indices()[1]  = indices
        return mask_sparse.to_dense()

