import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
from typing import Callable, Optional

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
import attention
import utils
from anchor_patch_samplers import *

PAD_VALUE = 0

class SG_MPN(MessagePassing):
    '''
    A single subgraph-level message passing layer

    Messages are passed from anchor patch to connected component and weighted by the channel-specific similarity between the two.
    The resulting messages for a single component are aggregated and used to update the embedding for the component.
    '''

    def __init__(self, node_embed_size, hparams):
        super(SG_MPN, self).__init__(aggr='add')  # "Add" aggregation.
        self.hparams = hparams
#         self.device = device
        self.linear =  nn.Linear(node_embed_size * 2, node_embed_size)
        self.linear_position = nn.Linear(node_embed_size,1)

    def create_patch_embedding_matrix(self,cc_embeds, cc_embed_mask, anchor_embeds, anchor_mask):
        '''
        Concatenate the connected component and anchor patch embeddings into a single matrix.
        This will be used an input for the pytorch geometric message passing framework.
        '''
        batch_sz, max_n_cc, cc_hidden_dim = cc_embeds.shape
        anchor_hidden_dim = anchor_embeds.shape[-1]

        # reshape connected component & anchor patch embedding matrices
        reshaped_cc_embeds = cc_embeds.view(-1, cc_hidden_dim) #(batch_sz * max_n_cc , hidden_dim)
        reshaped_anchor_embeds =  anchor_embeds.view(-1, anchor_hidden_dim) #(batch_sz * max_n_cc * n_sampled_patches, hidden_dim)

        # concatenate the anchor patch and connected component embeddings into single matrix
        patch_embedding_matrix = torch.cat([reshaped_anchor_embeds, reshaped_cc_embeds])
        return patch_embedding_matrix

    def create_edge_index(self, reshaped_cc_ids, reshaped_anchor_patch_ids, anchor_mask, n_anchor_patches):
        '''
        Create edge matrix of shape (2, # edges) where edges exist between connected components and their associated anchor patches

        Note that edges don't exist between components or between anchor patches
        '''
        # get indices into patch matrix corresponding to anchor patches
        anchor_inds = torch.tensor(range(reshaped_anchor_patch_ids.shape[0]))
        
        # get indices into patch matrix corresponding to connected components
        cc_inds = torch.tensor(range(reshaped_cc_ids.shape[0])) + reshaped_anchor_patch_ids.shape[0] 
        
        # repeat CC indices n_anchor_patches times
        cc_inds_matched = cc_inds.repeat_interleave(n_anchor_patches)
        
        # stack together two indices to create (2,E) edge matrix
        edge_index = torch.stack((anchor_inds, cc_inds_matched))
        mask_inds = anchor_mask.view(-1, anchor_mask.shape[-1])[:,0]

        return edge_index[:,mask_inds], mask_inds

    def get_similarities(self, edge_index, sims, cc_ids, anchor_ids, anchors_sim_index):
        '''
        Reshape similarities tensor of shape (n edges, 1) that contains similarity value for each edge in the edge index

        sims: (batch_size, max_n_cc, n possible anchor patches)
        edge_index: (2, number of edges between components and anchor patches)
        anchors_sim_index: indices into sims matrix for the structure channel that specify which anchor patches we're using
        '''
        n_cc = cc_ids.shape[0] 
        n_anchor_patches = anchor_ids.shape[0]
        
        batch_sz, max_n_cc, n_patch_options = sims.shape
        sims = sims.view(batch_sz * max_n_cc, n_patch_options)

        if anchors_sim_index != None: anchors_sim_index = anchors_sim_index * torch.unique(edge_index[1,:]).shape[0] # n unique CC
        
        # NOTE: edge_index contains stacked anchor, cc embeddings
        if anchors_sim_index == None: # neighborhood, position channels
            anchor_indices = anchor_ids[edge_index[0,:],:] - 1 # get the indices into the similarity matrix of which anchors were sampled
            cc_indices = edge_index[1,:] - n_anchor_patches  # get indices of the conneced components into the similarity matrix
            similarities = sims[cc_indices, anchor_indices.squeeze()]
        else: #structure channel

            # get indices of the conneced components into the similarity matrix
            cc_indices = edge_index[1,:] - n_anchor_patches #indexing into edge index is different than indexing into sims because patch matrix from which edge index was derived stacks anchor paches before the cc embeddings
            similarities = sims[cc_indices, torch.tensor(anchors_sim_index)] # anchors_sim_index provides indexing into the big similarity matrix - it tells you which anchors we actually sampled

        if len(similarities.shape) == 1: similarities = similarities.unsqueeze(-1)

        return similarities

    def generate_pos_struc_embeddings(self, raw_msgs, cc_ids, anchor_ids, edge_index, edge_index_mask):
        '''
        Generates the property aware position/structural embeddings for each connected component
        '''
        # Generate position/structure embeddings
        n_cc = cc_ids.shape[0]
        n_anchor_patches = anchor_ids.shape[0]
        embed_sz = raw_msgs.shape[1]
        n_anchors_per_cc = int(n_anchor_patches/n_cc)

        # 1) add masked CC back in & reshape
        # raw_msgs doesn't include padding so we need to add padding back in
        # NOTE: while these are named as position embeddings, these apply to structure channel as well
        pos_embeds = torch.zeros((n_cc * n_anchors_per_cc, embed_sz)).to(cc_ids.device) + PAD_VALUE
        pos_embeds[edge_index_mask] = raw_msgs # raw_msgs doesn't include padding so we need to add padding back in
        pos_embeds_reshaped = pos_embeds.view(-1, n_anchors_per_cc, embed_sz)

        # 2) linear layer + normalization
        position_out = self.linear_position(pos_embeds_reshaped).squeeze(-1)

        # optionally normalize the output of the linear layer (this is what P-GNN paper did) 
        if 'norm_pos_struc_embed' in self.hparams and self.hparams['norm_pos_struc_embed']:
            position_out = F.normalize(position_out, p=2, dim=-1) 
        else: # otherwise, just push through a relu
            position_out = F.relu(position_out) 

        return position_out #(n subgraphs * n_cc, n_anchors_per_cc )

    def forward(self, sims, cc_ids, cc_embeds, cc_embed_mask, anchor_patches, anchor_embeds, anchor_mask, anchors_sim_index): 
        '''
        Performs a single message passing layer

        Returns:
            - cc_embed_matrix_reshaped: order-invariant hidden representation (batch_sz, max_n_cc, node embed dim)
            - position_struc_out_reshaped: property aware cc representation (batch_sz, max_n_cc, n_anchor_patches)
        '''
        
        # reshape anchor patches & CC embeddings & stack together
        # NOTE: anchor patches then CC stacked in matrix
        patch_matrix = self.create_patch_embedding_matrix(cc_embeds, cc_embed_mask, anchor_embeds, anchor_mask)

        # reshape cc & anchor patch id matrices
        batch_sz, max_n_cc, max_size_cc = cc_ids.shape
        cc_ids = cc_ids.view(-1, max_size_cc) # (batch_sz * max_n_cc, max_size_cc)

        anchor_ids = anchor_patches.contiguous().view(-1, anchor_patches.shape[-1]) # (batch_sz * max_n_cc * n_sampled_patches, anchor patch size)
        n_anchor_patches_sampled = anchor_ids.shape[0]

        # create edge index
        edge_index, edge_index_mask = self.create_edge_index(cc_ids, anchor_ids, anchor_mask, anchor_patches.shape[2])

        # get similarity values for each edge index
        similarities = self.get_similarities(edge_index, sims, cc_ids, anchor_ids, anchors_sim_index)

        # Perform Message Passing
        # propagated_msgs: (length of concatenated anchor patches & cc, node dim size)
        propagated_msgs, raw_msgs =  self.propagate(edge_index.to(similarities.device), x=patch_matrix, similarity=similarities) 
        
        # Generate Position/Structure Embeddings
        position_struc_out = self.generate_pos_struc_embeddings(raw_msgs, cc_ids, anchor_ids, edge_index, edge_index_mask)

        # index resulting propagated messagaes to get updated CC embeddings & reshape
        cc_embed_matrix = propagated_msgs[n_anchor_patches_sampled:,:]
        cc_embed_matrix_reshaped = cc_embed_matrix.view(batch_sz , max_n_cc ,-1)

        # reshape property aware position/structure embeddings
        position_struc_out_reshaped = position_struc_out.view(batch_sz, max_n_cc, -1)
        
        return cc_embed_matrix_reshaped, position_struc_out_reshaped

    def propagate(self, edge_index, size=None, **kwargs):
        # We need to reimplement propagate instead of relying on base class implementation because we need 
        # to return the raw messages to generate the position/structure embeddings. 
        # Everything else is identical to propagate function from Pytorch Geometric.
         
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # run both functions in separation.
        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                        kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        msg_out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(msg_out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        out = self.update(out, **update_kwargs)

        return out, msg_out


    def message(self, x_j, similarity): #default is source to target
        '''
        The message is the anchor patch representation weighted by the similarity between the patch and the component
        '''
        return similarity * x_j

    def update(self, aggr_out, x):
        '''
        Update the connected component embedding from the result of the aggregation. The default is to 'use_mpn_projection',
        i.e. concatenate the aggregated messages with the previous cc embedding and push through a relu
        '''
        if self.hparams['use_mpn_projection']:
            return F.relu(self.linear(torch.cat([x, aggr_out], dim=1)))
        else:
            return aggr_out

class LSTM(nn.Module):
    '''
    bidirectional LSTM with linear head
    '''
    def __init__(self, n_features, h, dropout=0.0, num_layers=1, batch_first=True, aggregator='sum'):
        super().__init__()

        # number of LSTM layers
        self.num_layers = num_layers

        # type of aggregation('sum' or 'last')
        self.aggregator = aggregator

        self.lstm = nn.LSTM(n_features, h, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(h * 2, n_features)
    
    def forward(self, input):
        #input: (batch_sz, seq_len, hidden_dim )
        lstm_out, last_hidden = self.lstm(input)
        batch, seq_len, _ = lstm_out.shape

        # either take last hidden state or sum all hidden states
        if self.aggregator == 'last':
            lstm_agg = lstm_out[:,-1,:]
        elif self.aggregator == 'sum':
            lstm_agg = torch.sum(lstm_out, dim=1)
        elif self.aggregator == 'max':
            lstm_agg, _ = torch.max(lstm_out, dim=1)
        elif self.aggregator == 'avg':
            lstm_agg = torch.mean(lstm_out, dim=1)
        else:
            raise NotImplementedError
        return self.linear(lstm_agg)

class SubgEncoder(nn.Module):
    def __init__(
        self, node_embed_size, n_anchor_patches_structure, subg_n_layers, subg_hidden_dim, dropout_prob, lstm_n_layers, lstm_aggregator, hparams):
        super(SubgEncoder, self).__init__()
        self.hparams = hparams  
        self.subg_n_layers = subg_n_layers
        self.structure_mpns = nn.ModuleList()
        hid_dim = node_embed_size
        hid_dim += 2 * n_anchor_patches_structure * subg_n_layers
        for l in range(self.subg_n_layers):
            curr_layer = nn.ModuleDict() 
            curr_layer['internal'] = SG_MPN(node_embed_size, self.hparams)
            curr_layer['border'] = SG_MPN(node_embed_size, self.hparams)
            # optionally add batch_norm
            if 'batch_norm' in self.hparams and self.hparams['batch_norm']:
                curr_layer['batch_norm'] = nn.BatchNorm1d(node_embed_size)
                curr_layer['batch_norm_out'] = nn.BatchNorm1d(node_embed_size)

            self.structure_mpns.append(curr_layer)

        # initialize FF layers on top of MPN layers
        self.lin =  nn.Linear(hid_dim, subg_hidden_dim)

        # optional dropout on the linear layers
        self.lin_dropout = nn.Dropout(p=dropout_prob)

        # initialize LSTM - this is used in the structure channel for embedding anchor patches
        self.lstm = LSTM(
            node_embed_size, 
            node_embed_size,
            dropout=dropout_prob, 
            num_layers=lstm_n_layers,
            aggregator=lstm_aggregator,
        )

        # attention 
        self.attn_vector = torch.nn.Parameter(torch.zeros((hid_dim,1), dtype=torch.float), requires_grad=True)   
        nn.init.xavier_uniform_(self.attn_vector)
        self.attention = attention.AdditiveAttention(hid_dim, hid_dim)
        
    def initialize_cc_embeddings(self, node_embeddings, cc_id_list, aggregator='sum'):
        '''
        Initialize connected component embeddings as either the sum or max of node embeddings in the connected component
        Input:
            - cc_id_list: 3D tensor of shape (n subgraphs, max n CC, max length CC)
        Output:
            - 3D tensor of shape (n_subgraphs, max n_cc, node embedding dim)
        '''
        if aggregator == 'sum':
            return torch.sum(node_embeddings(cc_id_list), dim=2)
        elif aggregator == 'max':
            return torch.max(node_embeddings(cc_id_list), dim=2)[0]
        
    def run_mpn_layer(
        self, 
        node_embeddings,
        anchors_structure,
        mpn_fn, 
        subgraph_idx, 
        cc_ids,
        cc_embeds, 
        cc_embed_mask, 
        sims, 
        layer_num, 
        channel, 
        inside=True):
        '''
        Perform a single message-passing layer for the specified 'channel' and internal/border
        Returns:
            - cc_embed_matrix: updated connected component embedding matrix
            - position_struc_out: property aware embedding matrix (for position & structure channels)
        '''
        # Get Anchor Patches
        anchor_patches, anchor_mask, anchor_embeds = get_anchor_patches(
            None, self.hparams, node_embeddings, subgraph_idx, cc_ids, cc_embed_mask, self.lstm, 
            None, None, None, None, anchors_structure, layer_num, channel, inside)

        # for the structure channel, we need to also pass in indices into larger matrix of pre-sampled structure AP
        if channel == 'structure': anchors_sim_index = anchors_structure[layer_num][1]
        else: anchors_sim_index = None
        
        # one layer of message passing
        cc_embed_matrix, position_struc_out = mpn_fn(
            sims, cc_ids, cc_embeds, cc_embed_mask, anchor_patches, anchor_embeds, anchor_mask, anchors_sim_index)

        return cc_embed_matrix, position_struc_out
    
    def forward(
        self,
        node_embeddings,
        anchors_structure,
        S_I_cc_embed, 
        S_B_cc_embed, 
        cc_ids, 
        subgraph_idx, 
        I_S_sim, 
        B_S_sim):

        init_cc_embeds = self.initialize_cc_embeddings(node_embeddings, cc_ids)
        S_in_cc_embeds = init_cc_embeds.clone()
        S_out_cc_embeds = init_cc_embeds.clone()
        batch_sz, max_n_cc, _ = init_cc_embeds.shape
        
        # get mask for cc_embeddings
        cc_embed_mask = (cc_ids != PAD_VALUE)[:,:,0] # only take first element bc only need mask over n_cc, not n_nodes in cc

        # for each layer in SubgEncoder:
        outputs = []
        for l in range(self.subg_n_layers):
            # message passing layer for S internal and border
            S_in_cc_embeds, S_in_struc_embed = self.run_mpn_layer(
                node_embeddings, 
                anchors_structure, 
                self.structure_mpns[l]['internal'], 
                subgraph_idx, 
                cc_ids, 
                S_in_cc_embeds, 
                cc_embed_mask, 
                I_S_sim, 
                layer_num=l, 
                channel='structure', 
                inside=True)
            S_out_cc_embeds, S_out_struc_embed = self.run_mpn_layer(
                node_embeddings, 
                anchors_structure, 
                self.structure_mpns[l]['border'], 
                subgraph_idx, 
                cc_ids, 
                S_out_cc_embeds, 
                cc_embed_mask, 
                B_S_sim, 
                layer_num=l, 
                channel='structure', 
                inside=False)
            if 'batch_norm' in self.hparams and self.hparams['batch_norm']:  #optional batch norm
                S_in_cc_embeds = self.structure_mpns[l]['batch_norm'](S_in_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
                S_out_cc_embeds = self.structure_mpns[l]['batch_norm_out'](S_out_cc_embeds.view(batch_sz*max_n_cc,-1)).view(batch_sz,max_n_cc, -1 )
            outputs.extend([S_in_struc_embed, S_out_struc_embed])

        # concatenate all layers
        all_cc_embeds = torch.cat([init_cc_embeds] + outputs, dim=-1)
        
        # attention
        batched_attn = self.attn_vector.squeeze().unsqueeze(0).repeat(all_cc_embeds.shape[0],1)
        attn_weights = self.attention(batched_attn, all_cc_embeds, cc_embed_mask)
        subgraph_embedding = utils.weighted_sum(all_cc_embeds, attn_weights)
        subgraph_embedding_out = torch.relu(self.lin(subgraph_embedding))
        return subgraph_embedding_out
    
    
class PositionalEncoder(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
class StructuralEncoder(nn.Module):
    def __init__(self, input_dim):
        super(StructuralEncoder, self).__init__()
        self.input_dim = input_dim
        self.attn_vector = Parameter(torch.zeros((self.input_dim, 1), dtype=torch.float), requires_grad=True)   
        self.attention = attention.AdditiveAttention(self.input_dim, self.input_dim)
        self.xavier_init()

    def forward(self, inputs):
        attn_vectors = self.attn_vector.squeeze().unsqueeze(0).repeat(inputs.shape[0], 1)
        attn_weights = self.attention(attn_vectors, inputs)
        outputs = utils.weighted_sum(inputs, attn_weights)
        return outputs
    
    def xavier_init(self):
        nn.init.xavier_uniform_(self.attn_vector)

class TemporalEncoder(nn.Module):
    '''
    It is similar to https://github.com/FeiGSSS/DySAT_pytorch/blob/c66da8b0677b528f9661e4e1adfe17474fd76261/models/layers.py
    '''
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                dropout_prob, 
                residual):
        super(TemporalEncoder, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        self.pe = PositionalEncoder(input_dim, dropout=dropout_prob)

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(dropout_prob)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        temporal_inputs = self.pe(inputs) # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class CrossEntropyLossList(_WeightedLoss):
    def __init__(self, weight: Optional[Tensor] = None, eps=1e-7):
        super(CrossEntropyLossList, self).__init__(weight, eps)
        self.eps = eps
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        preds_smax = F.softmax(input, dim=1)
        true_smax = F.softmax(target, dim=1)
        preds_smax = preds_smax + self.eps
        preds_log = torch.log(preds_smax)
        if self.weight is not None:
            loss = torch.mean(-torch.sum(true_smax * preds_log * self.weight, dim=1))
        else:
            loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
        return loss