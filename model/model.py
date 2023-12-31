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
    def forward(self, tgt: Tensor, memory: Tensor, tgt_ori:Tensor, tgt_mask: Tensor = None, memory_mask: Tensor = None, tgt_key_padding_mask: Tensor  = None, memory_key_padding_mask: Tensor = None) -> Tensor:
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
        
        if get_task() == 'classification':
            self.Lable_embedding = nn.Embedding(L, embedding_dim, dtype=torch.float)
        elif get_task() == 'regression':
            self.Lable_embedding = nn.Linear(L_dim, embedding_dim, dtype=torch.float)
    
        # for every numrical filed, construct it's own Linear embedding layer
        self.Catagory_embedding_nums = nn.ModuleList()
        for i in range(num_NUM):
            self.Catagory_embedding_nums.append(
                nn.Linear(C_dim, embedding_dim, dtype=torch.float, device=DEVICE)
            )
            
        catagories = dataset.nodes_of_fields[-num_CAT:] # number of all possible catagories nodes
        self.Catagory_embedding_cat = nn.Embedding(sum(catagories), embedding_dim, dtype=torch.float)
        
        self.Field_embedding = nn.Embedding(F, embedding_dim, dtype=torch.float)
        
        self.transformer_decoder = {}
        self.transformer_decoder['L2S'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        self.transformer_decoder['S2C'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        self.transformer_decoder['C2F'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        self.transformer_decoder['F2C'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        self.transformer_decoder['C2S'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        self.transformer_decoder['S2L'] = TabTransformerDecoder(TabHyperformer_Layer(embedding_dim,  nhead = 2 ), num_layers).to(DEVICE)
        
        
        # downstream task
        if get_task() == 'classification':
            self.MLP = nn.Sequential(
                nn.Linear(embedding_dim, get_num_classes()),
            )
        elif get_task() == 'regression':
            self.MLP = nn.Sequential(
                nn.Linear(embedding_dim, 1),
            )
        else:
            raise NotImplementedError('task', get_task(), 'not implemented')
        
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
            def maskout_label(batch_index, sample_indice):
                query_index = np.argwhere(sample_indice == query_indices[batch_index])[0][0]
                # L2S mask shape : B, S, L
                self.tmpmask_L2S[batch_index, query_index] = 1
                self.tmpmask_L2S[batch_index, query_index][L-1] = 0  # connect to all label nodes except the unseen label
            _ = list(map(maskout_label, range(len(sample_indices)), sample_indices))
        else:
            self.tmpmask_L2S = dataset.MASKS['L2S'].clone().detach()
            # return
            def maskout_label(index, query):
                self.tmpmask_L2S[index, query] = 0
                self.tmpmask_L2S[index, query][L-1] = 1 # make it as unseen label
            _ = list(map(maskout_label, range(len(query_indices)), query_indices))
                
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
            # query_indexs = [sample_indices[i].index(query) for i, query in enumerate(query_indices)]
            query_indexs = np.argmax(np.array(sample_indices) == np.array(query_indices)[:, np.newaxis], axis=1)
            S_ = K # the S used in transformer decoder
            print_checkpoint_time('get query_indexs')
        elif mode == 'inferring':
            # use all nodes in the graph 
            # get input tensor (no need to update)
            L_input, S_input, C_input, F_input = dataset.INPUTS
            L_input = L_input.repeat(batch_size,1,1)
            S_input = S_input.repeat(batch_size,1,1)
            C_input = C_input.repeat(batch_size,1,1) # (batch, C node, 1)
            F_input = F_input.repeat(batch_size,1,1)
            # updata mask for inference node
            dataset.make_mask_test(query_indices) # query node equal to inference node, only one query node is allowed
            print_checkpoint_time('make_mask_test')
            masks = dataset.MASKS
            # modify C_input, make unseen catagory nodes' value equal to sample's value
            NUM, _, _ = get_feilds_attributes()
            
            # cacyulate numerical_unseen_indexes
            # numerical_unseen_indexes = [dataset.unseen_node_indexs_C[filed] for filed in NUM]
            get_index_list = np.vectorize(lambda field: dataset.unseen_node_indexs_C[field])
            numerical_unseen_indexes = get_index_list(NUM)
            
            test_data_pool = dataset.TEST_POOL_VALUES.drop(columns=[get_target()])
            C_input[:,numerical_unseen_indexes] = torch.tensor((test_data_pool.iloc[query_indices][NUM]).to_numpy(),dtype=torch.double,device=get_DEVICE()).unsqueeze(-1)
            print_checkpoint_time('modify C_input')

            self.maskout_lable(dataset, query_indices, None)
            print_checkpoint_time('maskout_lable')
            
            
            # the query node's indexs in sample_indices
            # query_indexs = [S-1]
            query_indexs = [S-1]*batch_size
            S_ = S # the S used in transformer decoder
        else:
            raise NotImplementedError

        # for S and C, we use two different embedding methods, for CAT and NUM, respectively
        # Squeeze for making batch dimantion
        if get_task() == 'classification':
            L_embedded = self.Lable_embedding(L_input.long()).squeeze(2).float()
        elif get_task() == 'regression':
            L_embedded = self.Lable_embedding(L_input.float()).squeeze(2).float()
        
        S_embedded = S_input.float()
        print_checkpoint_time('get L+S_embedded')
        # for every numrical filed, use it's own Linear embedding layer
        field = dataset.nodes_of_fields

        weights = []
        biases = []
        C_inputs = []
        
        start = 0
        for index, nodes in enumerate(field[:num_NUM]):
            end = start + nodes
            weights.append(torch.repeat_interleave(self.Catagory_embedding_nums[index].weight.unsqueeze(0), nodes, dim = 0))
            biases.append(torch.repeat_interleave(self.Catagory_embedding_nums[index].bias.unsqueeze(0), nodes, dim=0))
            C_inputs.append(C_input[:,start:end])
            start = end
        
        weights = torch.cat(weights, dim=0) # (node, hid, 1)
        biases = torch.cat(biases, dim=0) # (node, hid)
        C_inputs = torch.cat(C_inputs, dim=1).float() # (batch, node, 1)
        #print(weights.shape, biases.shape, C_inputs.shape)
        
        C_embedded_num = ((C_inputs.unsqueeze(-1) * weights.unsqueeze(0)).sum(-1) + biases.unsqueeze(0))
        
        if num_CAT > 0:
            catagorical_filed_nodes = sum(field[-num_CAT:]) # pick catagory fields
            C_embedded_cat = self.Catagory_embedding_cat(C_input[:,-catagorical_filed_nodes:].squeeze(2).long()).float() 
            C_embedded = torch.cat([C_embedded_num, C_embedded_cat], dim = 1)
        else:
            C_embedded = C_embedded_num
        print_checkpoint_time('get C_embedded')
        
        F_embedded = self.Field_embedding(F_input.long()).squeeze(2).float()
        print_checkpoint_time('get F_embedded')
        
        
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
                S_embedded = self.transformer_decoder['L2S'] (S_embedded,L_embedded, origin_S,
                                                    memory_mask = self.tmpmask_L2S[:,:S_,:L])# + S_embedded
                C_embedded = self.transformer_decoder['S2C'] (C_embedded,S_embedded, origin_C,
                                                    memory_mask = masks['S2C'][:,:C,:S_])# + C_embedded   
                F_embedded = self.transformer_decoder['C2F'] (F_embedded,C_embedded, origin_F,
                                                    memory_mask = masks['C2F'][:,:F,:C])# + F_embedded
                C_embedded = self.transformer_decoder['F2C'] (C_embedded,F_embedded, origin_C,
                                                    memory_mask = Tensor.contiguous(masks['C2F'].transpose(1, 2))[:,:C,:F])# + C_embedded
                S_embedded = self.transformer_decoder['C2S'] (S_embedded,C_embedded, origin_S,
                                                    memory_mask = Tensor.contiguous(masks['S2C'].transpose(1, 2))[:,:S_,:C])# + S_embedded
                # L_embedded = self.transformer_decoder['S2L'] (L_embedded,S_embedded, origin_L,
                #                                     memory_mask = Tensor.contiguous(self.tmpmask_L2S.clone().detach().transpose(1, 2))[:,:L,:S_])# + L_embedded
        print_checkpoint_time('propagate')
        output = self.MLP(S_embedded)
        outputs = output[np.arange(len(query_indexs)), query_indexs]
        print_checkpoint_time('MLP')
        return outputs
  