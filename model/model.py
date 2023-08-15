from torch import Tensor
from typing import Optional, Any, Union, Callable
from utils.utils import *
from data.dataset import HGNN_dataset
import xformers.ops as xops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from tqdm import trange, tqdm


class TabTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_ori:Tensor, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None, tgt_key_padding_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None) -> Tensor:
        '''
        Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        '''
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_ori, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class TabHyperformer_Layer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation='relu'):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # remove defined modules
        delattr(self, 'self_attn')
        delattr(self, 'norm1')
        delattr(self, 'dropout1')
        self.FF = nn.Sequential(
                nn.Linear(d_model,d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model)
        )
        self.scoring_layer = nn.Sequential(
            nn.Linear(d_model, 1)
        )
        self.pre_transform_Q = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.pre_transform_K = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.pre_transform_V = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, tgt, memory, tgt_ori, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        x_ori = tgt_ori
        # x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + x_ori + self._mha_block(x, memory, memory_mask))
        # x =  x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
        #x = self.norm3(x + self.FF(x))
        
        ff_out = self.FF(x)
        ff_score = self.scoring_layer(ff_out)
        x_score = self.scoring_layer(x)
        score = torch.cat([ff_score, x_score], dim=-1).softmax(-1)
        x = ff_out * score[:, :, :1] + x * score[:, :, 1:]
        
        return x
    
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],) -> Tensor:
        # linear projection
        # Q = self.pre_transform_Q(x)
        # K = self.pre_transform_K(mem)
        # V = self.pre_transform_V(mem)
        x = xops.memory_efficient_attention(x, mem, mem, attn_mask)
        
        # return self.dropout(x)
        return (x)
    
# baic transformer decoder model
class TransformerDecoderModel(nn.Module):
    def __init__(self, 
                 dataset : HGNN_dataset, 
                 num_layers = 1, 
                 embedding_dim = 128, 
                 propagation_steps = 1,
                 ):
        super(TransformerDecoderModel, self).__init__()

        DEVICE = get_DEVICE()
        L_dim, S_dim, C_dim, F_dim = dataset.INPUT_DIMS
        L, S, C, F = dataset.nodes_num['L'], dataset.nodes_num['S'], dataset.nodes_num['C'], dataset.nodes_num['F']
        num_NUM , num_CAT = dataset.NUM_vs_CAT
        

        self.Lable_embedding = nn.Embedding(L, embedding_dim, dtype=torch.float)
    
        # self.Catagory_embedding_num = nn.Linear(C_dim, embedding_dim, dtype=torch.float)
        # for every numrical filed, construct it's own Linear embedding layer
        self.Catagory_embedding_nums = []
        for i in range(num_NUM):
            self.Catagory_embedding_nums.append(
                nn.Linear(C_dim, embedding_dim, dtype=torch.float, device=DEVICE)
            )
        catagories = dataset.nodes_of_fields[-num_CAT:] # number of all possible catagories nodes
        self.Catagory_embedding_cat = nn.Embedding(sum(catagories), embedding_dim, dtype=torch.float)
        
        self.Field_embedding = nn.Embedding(F, embedding_dim, dtype=torch.float)
        
        self.transformer_decoder = TabTransformerDecoder(
            TabHyperformer_Layer(embedding_dim,  nhead = 2 ),
            num_layers
        )
        
        # downstream task
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, 2),
        )
        
        # initialize MASK_FULL
        dataset.make_mask_all()
        # dataset.make_input_tensor()
        
        self.tmpmask_L2S = dataset.MASKS['L2S'].clone()

        self.propagation_steps = propagation_steps
        
        self.C_num_starts = []
        self.C_num_ends = []
        start = end = 0
        for nodes in dataset.nodes_of_fields[:num_NUM]:
            end = start + nodes
            self.C_num_starts.append(start)
            self.C_num_ends.append(end)
            start = end

    def maskout_lable(self,
                      dataset: HGNN_dataset,
                      query_indices: list, # must be sorted
                      sample_indices: Optional[list] = None, 
                      ):
        L = dataset.nodes_num['L']
        if sample_indices is not None:
            self.tmpmask_L2S = dataset.MASKS['L2S'].clone().detach()
            for index, sample_indice in enumerate(sample_indices): # sample_indice in length K
                # modify the mask to mask out the queries node's edge to it's label node
                query_index = sample_indice.index(query_indices[index]) # query_index: index of query node in sample_indice of the batch
                # L2S mask shape : B, S, L
                # self.tmpmask_L2S[index, query_index][:-1] = 1 # connect to all label nodes except the unseen label
                self.tmpmask_L2S[index, query_index] = 0
                self.tmpmask_L2S[index, query_index][L-1] = 1 # make it as unseen label
        else:
            self.tmpmask_L2S = dataset.MASKS['L2S'].clone().detach()
            for index, query in enumerate(query_indices):
                # self.tmpmask_L2S[index, query][:-1] = 1 # connect to all label nodes except the unseen label
                self.tmpmask_L2S[index, query] = 0
                self.tmpmask_L2S[index, query][L-1] = 1 # make it as unseen label
                
                
    def forward(self, 
                dataset: HGNN_dataset, 
                mode : str = 'train',
                query_indices: list = None,  # must be sorted
                K : Optional[int] = 10,
                unseen_rate : Optional[float] = 0.1,
                aug_strategy: Optional[str] = 'random',
                ):
        print_checkpoint_time('start forward')
        L, S, C, F = dataset.nodes_num['L'], dataset.nodes_num['S'], dataset.nodes_num['C'], dataset.nodes_num['F']
        num_NUM, num_CAT = dataset.NUM_vs_CAT
        batch_size = len(query_indices)
        # decide scenario
        if mode == 'train':
            # generate subgraph with K nodes, including query_indices
            # update mask and input tensor
            sample_indices = dataset.get_sample(K, query_indices = query_indices) # update mask
            print_checkpoint_time('get_sample')
            dataset.make_mask_subgraph(sample_indices, query_indices)
            print_checkpoint_time('make_mask_subgraph')
            masks = dataset.MASKS
            
            # data_augmentation
            masks['S2C'] = dataset.random_connect_unseen(masks['S2C'], mask_ratio = unseen_rate, strategy = aug_strategy)
            print_checkpoint_time('random_connect_unseen')
            # get updated masked input tensor and mask 
            L_input, S_input, C_input, F_input = dataset.MASKED_INPUTS
            L_input = L_input.clone().detach().repeat(batch_size,1,1)
            # S_input is already in shape [batch_size, sample_size, embedding_dim], see get_sample()
            S_input = S_input.clone().detach()
            C_input = C_input.clone().detach().repeat(batch_size,1,1)
            F_input = F_input.clone().detach().repeat(batch_size,1,1)
            print_checkpoint_time('get masked input')
            # mask out the queries node's edge to it's label node, prevent label leakage
            self.maskout_lable(dataset, query_indices, sample_indices)
            print_checkpoint_time('maskout_lable')
            # the query node's indexs in sample_indices
            query_indexs = [sample_indices[i].index(query) for i, query in enumerate(query_indices)]
            S_ = K # the S used in transformer decoder
            print_checkpoint_time('get query_indexs')
        elif mode == 'inferring':
            # use all nodes in the graph 
            # get input tensor (no need to update)
            L_input, S_input, C_input, F_input = dataset.INPUTS
            L_input = L_input.repeat(batch_size,1,1)
            S_input = S_input.repeat(batch_size,1,1)
            C_input = C_input.repeat(batch_size,1,1)
            F_input = F_input.repeat(batch_size,1,1)
            # updata mask for inference node
            dataset.make_mask_test(query_indices) # query node equal to inference node, only one query node is allowed
            masks = dataset.MASKS
            
            self.maskout_lable(dataset, query_indices, None)
            
            
            # the query node's indexs in sample_indices
            # query_indexs = [S-1]
            query_indexs = [S-1]*batch_size
            S_ = S # the S used in transformer decoder
        else:
            raise NotImplementedError

        # for S and C, we use two different embedding methods, for CAT and NUM, respectively
        # Squeeze for making batch dimantion
        L_embedded = self.Lable_embedding(L_input.long()).squeeze(2).float()
        
        S_embedded = S_input.float()
        print_checkpoint_time('get L+S_embedded')
        # for every numrical filed, use it's own Linear embedding layer
        C_embedded_nums = []
        field = dataset.nodes_of_fields
        start = 0
        for index, nodes in enumerate(field[:num_NUM]): # pick numrical fields
            end = start + nodes
            C_embedded_nums.append(self.Catagory_embedding_nums[index](C_input[:,start:end].float()))
            start = end
        
        # numerical_field = field[:num_NUM]
        # def C_embedded_num_cal(index):
        #     start = sum(numerical_field[:index])
        #     end = sum(numerical_field[:index+1])
        #     return (self.Catagory_embedding_nums[index](C_input[:,start:end].float()))
        # C_embedded_nums = list(map(C_embedded_num_cal, range(num_NUM)))
        C_embedded_num = torch.cat(C_embedded_nums, dim = 1)
        
        catagorical_filed_nodes = sum(field[-num_CAT:]) # pick catagory fields
        C_embedded_cat = self.Catagory_embedding_cat(C_input[:,-catagorical_filed_nodes:].squeeze(2).long()).float() 
        # print(C_embedded_num.shape, C_embedded_cat.shape)
        C_embedded = torch.cat([C_embedded_num, C_embedded_cat], dim = 1)
        print_checkpoint_time('get C_embedded')
        
        F_embedded = self.Field_embedding(F_input.long()).squeeze(2).float()
        print_checkpoint_time('get F_embedded')
        # print(query_indices, K)
        # print(L_embedded.shape, S_embedded.shape, C_embedded.shape, F_embedded.shape)
        
        
        # propagate steps: L→S→C→F
        #                  L←S←C←
        # more steps more menory usage
        PROPAGATE_STEPS = self.propagation_steps
        origin_S = S_embedded.clone()
        origin_C = C_embedded.clone()
        origin_F = F_embedded.clone()
        origin_L = L_embedded.clone()
        
        RANDOM_PROPAGATE = False
        if RANDOM_PROPAGATE:
            # random propagate
            for _ in range(PROPAGATE_STEPS):
                possible_steps = ['S2C', 'C2F', 'F2C', 'C2S', 'S2L', 'L2S']
                selected_steps = np.random.choice(possible_steps, len(possible_steps), replace = False)
                
                for step in selected_steps:
                    if step == 'L2S':
                        S_embedded = self.transformer_decoder(S_embedded,L_embedded, origin_S,
                                                        memory_mask = self.tmpmask_L2S.clone().detach()[:,:S_,:L])# + S_embedded
                    elif step == 'S2C':
                        C_embedded = self.transformer_decoder(C_embedded,S_embedded, origin_C,
                                                        memory_mask = masks['S2C'].clone().detach()[:,:C,:S_])# + C_embedded
                    elif step == 'C2F':
                        F_embedded = self.transformer_decoder(F_embedded,C_embedded, origin_F,
                                                        memory_mask = masks['C2F'].clone().detach()[:,:F,:C])# + F_embedded
                    elif step == 'F2C':
                        C_embedded = self.transformer_decoder(C_embedded,F_embedded, origin_C,
                                                        memory_mask = Tensor.contiguous(masks['C2F'].clone().detach().transpose(1, 2))[:,:C,:F])# + C_embedded
                    elif step == 'C2S':
                        S_embedded = self.transformer_decoder(S_embedded,C_embedded, origin_S,
                                                        memory_mask = Tensor.contiguous(masks['S2C'].clone().detach().transpose(1, 2))[:,:S_,:C])# + S_embedded
                    elif step == 'S2L':
                        L_embedded = self.transformer_decoder(L_embedded,S_embedded, origin_L,
                                                        memory_mask = Tensor.contiguous(self.tmpmask_L2S.clone().detach().transpose(1, 2))[:,:L,:S_])# + L_embedded
        else:
            for i in range(PROPAGATE_STEPS):
                S_embedded = self.transformer_decoder(S_embedded,L_embedded, origin_S,
                                                    memory_mask = self.tmpmask_L2S.clone().detach()[:,:S_,:L])# + S_embedded
                C_embedded = self.transformer_decoder(C_embedded,S_embedded, origin_C,
                                                    memory_mask = masks['S2C'].clone().detach()[:,:C,:S_])# + C_embedded   
                F_embedded = self.transformer_decoder(F_embedded,C_embedded, origin_F,
                                                    memory_mask = masks['C2F'].clone().detach()[:,:F,:C])# + F_embedded
                C_embedded = self.transformer_decoder(C_embedded,F_embedded, origin_C,
                                                    memory_mask = Tensor.contiguous(masks['C2F'].clone().detach().transpose(1, 2))[:,:C,:F])# + C_embedded
                S_embedded = self.transformer_decoder(S_embedded,C_embedded, origin_S,
                                                    memory_mask = Tensor.contiguous(masks['S2C'].clone().detach().transpose(1, 2))[:,:S_,:C])# + S_embedded
                L_embedded = self.transformer_decoder(L_embedded,S_embedded, origin_L,
                                                    memory_mask = Tensor.contiguous(self.tmpmask_L2S.clone().detach().transpose(1, 2))[:,:L,:S_])# + L_embedded
        print_checkpoint_time('propagate')
        # print('after',S_embedded[0][0])
        output = self.MLP(S_embedded)
        outputs = output[np.arange(len(query_indexs)), query_indexs]
        print_checkpoint_time('MLP')
        # outputs = []
        # for index, query in enumerate(query_indexs):
            # outputs.append(output[index, query])
        # outputs = torch.stack(outputs, dim = 0)
        # output_batch = [output[:,query_indexs][query_indexs[i]] for i in range(batch_size)]
        # print(output_batch)
        return outputs
  