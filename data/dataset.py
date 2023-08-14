import pandas as pd
import numpy as np
import math
import random
import torch
from torch import Tensor
from typing import Optional, Tuple

from data.preprocess import POOL_preprocess, POOL_preprocess_inference
from utils.utils import get_DEVICE



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
        TEST_POOL, self.unseen_node_indexs_C = POOL_preprocess_inference(test_pool, self.inference_package)
        LABEL_COLUMN = label_column

        # cut feature and lable
        FEATURE_POOL = TRAIN_POOL.drop(LABEL_COLUMN, axis=1)
        LABEL_POOL = TRAIN_POOL[LABEL_COLUMN]
        TEST_LABEL_POOL = TEST_POOL[LABEL_COLUMN]
        
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
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
        # C_POOL = range(int(C))

        nodes_num = {'L':L, 'S':S, 'C':C, 'F':F}
        print('node_nums', nodes_num)
        # print('total', L+S+C+F, 'nodes')
        
        # get samples indexs for each label
        self.labe_to_index = {}
        tmp_pool = TRAIN_POOL.copy().reset_index(drop=True)
        for label in tmp_pool['income'].unique():
            self.labe_to_index[label] = (tmp_pool[tmp_pool['income'] == label].index).tolist()
        
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
        DEVICE = get_DEVICE()
        L, S, C, F = self.nodes_num['L'], self.nodes_num['S'], self.nodes_num['C'], self.nodes_num['F']

        sample_size = len(sample_indices[0])
        # caculate masking
        masks = {}
        
        tmp_L2S = []
        tmp_S2C = []
        for batch_indices in sample_indices:
            # masked_POOL = self.TRAIN_POOL.iloc[batch_indices] # sample dataframe into shape (10,14)
            
            # label to sample
            tmp = self.MASKS_FULL['L2S']
            tmp = torch.index_select(tmp, 0, torch.tensor(batch_indices, device=DEVICE)) #The returned tensor does not use the same storage as the original tensor
            # tmp = torch.zeros([math.ceil(sample_size/8) * 8, math.ceil(L/8) * 8], dtype=torch.float, device=DEVICE) 
            # label_value = masked_POOL[self.LABEL_COLUMN].values
            # tmp[torch.arange(sample_size, device=DEVICE), torch.tensor(label_value - min(label_value), device=DEVICE)] = 1
            shape = (math.ceil(sample_size/8) * 8, math.ceil(L/8) * 8)
            new_tensor = torch.zeros(*shape, device=DEVICE)
            new_tensor[:tmp.shape[0], :tmp.shape[1]] = tmp
            tmp = new_tensor.view(*shape)
            tmp_L2S.append(tmp)
            # masks['L2S'] = tmp.repeat(batch_size,1,1)
            

            # sample to catagory
            tmp = self.MASKS_FULL['S2C']
            tmp = torch.index_select(tmp, 1, torch.tensor(batch_indices, device=DEVICE)) # The returned tensor does not use the same storage as the original tensor
            # tmp = torch.zeros([math.ceil(C/8) * 8, math.ceil(sample_size/8) * 8], dtype=torch.float, device=DEVICE).T
            # tmp_df = masked_POOL.drop(self.LABEL_COLUMN, axis=1)
            # tmp[torch.arange(sample_size, device=DEVICE).unsqueeze(-1), torch.tensor(tmp_df.values, device=DEVICE)] = 1
            # tmp = tmp.T.contiguous()
            shape = (math.ceil(C/8) * 8, math.ceil(sample_size/8) * 8)
            new_tensor = torch.zeros(*shape,device=DEVICE)
            new_tensor[:tmp.shape[0], :tmp.shape[1]] = tmp
            tmp = new_tensor.view(*shape)
            tmp_S2C.append(tmp)
        
        masks['L2S'] = torch.stack(tmp_L2S, dim = 0)
        masks['S2C'] = torch.stack(tmp_S2C, dim = 0)
        
        # masks['S2C'] = Tensor.contiguous(tmp.repeat(batch_size,1,1))
        # masks['S2C'] = torch.stack(tmp_, dim = 0)

        # catagory to field
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
        label_ids = self.TRAIN_POOL[self.LABEL_COLUMN].unique()
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
        '''
        L, S, C, F = self.nodes_num['L'], self.nodes_num['S'], self.nodes_num['C'], self.nodes_num['F']
        
        masks = {}
        tmp_L2S = []
        tmp_S2C = []
        # for i query nodes
        for index_in_test_pool in indexs_in_test_pool:
            # L2S shape: torch.Size([39080, 8]), values in torch.Size([39074, 3]).
            # number of sample nodes : 39073 + 1 (inference node)
            # S = 39074, -1 to convert to index of last node
            tmp = self.MASKS_FULL['L2S'].clone().detach()
            tmp[S-1, L-1] = 1 # connect inference node to unseen lable nodes
            tmp_L2S.append(tmp)
            # masks['L2S'] = tmp.unsqueeze(0)
        
            # S2C shape: torch.Size([472, 39080]), values in torch.Size([470, 39074]).
            # self.MASKS_FULL['S2C'].T :[39080, 472], values in [39074, 470]
            # self.TEST_POOL.drop(self.LABEL_COLUMN, axis=1).values[index_in_test_pool]
            tmp = self.MASKS_FULL['S2C'].T.clone().detach()
            # connect the last sample node (inference node) with it's catagory nodes
            tmp[S-1, self.TEST_POOL.drop(self.LABEL_COLUMN, axis=1).values[index_in_test_pool]] = 1  
            tmp_S2C.append(tmp.T)
            
            # masks['S2C'] = tmp.T.contiguous().unsqueeze(0)
            
        masks['L2S'] = torch.stack(tmp_L2S, dim = 0)
        masks['S2C'] = torch.stack(tmp_S2C, dim = 0)
        # C2F remains the same
        masks['C2F'] = self.MASKS_FULL['C2F'].repeat(len(indexs_in_test_pool),1,1)
        
        self.MASKS = masks
        # print('masks[\'L2S\']',masks['L2S'].shape)
        # print('masks[\'S2C\']',masks['S2C'].shape)
        # print('masks[\'C2F\']',masks['C2F'].shape)
        
        
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
        # print('L_input', L_input.type(), L_input.shape)
        
        # S (normalized by standard scaler)
        # features = torch.tensor(self.FEATURE_POOL.values, device=DEVICE).float()
        # normalized_features = (features - torch.mean(features, dim = 0)) / torch.std(features, dim = 0)
        # S_input = torch.cat([normalized_features, torch.tensor([[0]*F], device=DEVICE)],dim = 0).float() # add infering node
        
        # S (initialize by random)
        S_input = torch.rand(self.embedding_dim, device=DEVICE).repeat(S,1)
        # print('S_input', S_input.type(), S_input.shape)
        # C 
        C_input = torch.tensor(np.array([self.C_POOL]), device=DEVICE).reshape(-1,1)
        # print('C_input', C_input.type(), C_input.shape)
        # F 
        F_input = torch.tensor([range(F)], device=DEVICE).reshape(-1,1)
        # print('F_input', F_input.type(), F_input.shape)
        # 
        self.INPUTS = (L_input, S_input, C_input, F_input)
        self.INPUT_DIMS = (L_input.size(1), S_input.size(1), C_input.size(1), F_input.size(1))

    def sample_with_distrubution(self, sample_size):
        '''
        Sample equally from each label with required sample size\\
        forced to make balenced sample
        '''
        # decide each label's number of samples (fourced to be balenced if possible) 
        label_list = []
        label_unique = list(self.labe_to_index.keys())
        count = sample_size // len(label_unique)
        remainder = sample_size % len(label_unique)
        label_list = [item for item in label_unique for _ in range(count)]
        label_list.extend(random.sample(label_unique, remainder))
        # sample from indexes
        indices = [random.choice(self.labe_to_index[label]) for label in label_list]
        return indices     
        
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
        DEVICE = get_DEVICE()
        # include specific nodes (e.g. query nodes), while remaining sample_size
        sample_indices = []
        if query_indices is not []:
            for query in query_indices:
                indices = self.sample_with_distrubution(sample_size - 1)
                while query in indices:
                    indices = self.sample_with_distrubution(sample_size - 1)
                # add query nodes into sample_indices
                indices.append(query)
                sample_indices.append(sorted(indices))
        else:
            indices = self.sample_with_distrubution(sample_size - len(query_indices))
            sample_indices.append(sorted(indices))
        # update mask
        # modify input tensor
        L_input, S_input, C_input, F_input = self.INPUTS
        S_input_masked = []
        for i in range(len(query_indices)):
            S_input_masked.append(torch.index_select(S_input, 0, torch.tensor(sample_indices[i], device=DEVICE)))
        S_input_masked = torch.stack(S_input_masked, dim = 0) # convert back to tensor
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
        DEVICE = get_DEVICE()
        if mask_ratio == 0:
            return mask
        unseen_node_indexs_list = list(self.unseen_node_indexs_C.values())
        # print('unseen_node_indexs_list', unseen_node_indexs_list)
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
            replacement_table = []
            lower_bound = 0
            for unseen_node_index in unseen_node_indexs_list:
                for _ in range(mask_sparse.indices()[2].max()+1):
                    replacement_table.append(random.randint(lower_bound, unseen_node_index+1))
                lower_bound = unseen_node_index+1
            replacement_table = torch.tensor(replacement_table, device=DEVICE).flatten()
        else:
            raise ValueError('strategy should be either unseen or random')
        
        for batch in range(num_batches):
            this_batch_indices = indices[batch*edges_per_batch:(batch+1)*edges_per_batch]
            num_samples = (torch.rand(len(this_batch_indices), device=DEVICE) < mask_ratio).int() * torch.add(torch.arange(len(this_batch_indices), device=DEVICE),1)
            to_be_replace = torch.add(num_samples[num_samples.nonzero()].squeeze(-1), -1 )
            # to_be_replace is a mask of indices to be replaced

            indices[to_be_replace + batch*edges_per_batch] = replacement_table[to_be_replace]

        mask_sparse.indices()[1]  = indices
        # print(indices)
        return mask_sparse.to_dense()

