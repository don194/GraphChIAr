import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [B, N, H, M] (Batched) or [N, H, M] (Unbatched)
    ks: key tensor   [B, L, H, M] (Batched) or [L, H, M] (Unbatched)
    vs: value tensor [B, L, H, D] (Batched) or [L, H, D] (Unbatched)

    return output [B, N, H, D] or [N, H, D]
    '''
    
    # Check if input is batched (4D)
    is_batched = qs.dim() == 4

    if kernel == 'simple':
        # Normalize input
        qs = qs / torch.norm(qs, p=2, dim=-1, keepdim=True)
        ks = ks / torch.norm(ks, p=2, dim=-1, keepdim=True)

        if is_batched:
            # Batched implementation
            B, N, H, M = qs.shape

            # numerator
            kvs = torch.einsum("blhm,blhd->bhmd", ks, vs) # [B, H, M, D]
            attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs) # [B, N, H, D]

            all_ones = torch.ones([vs.shape[1]]).to(vs.device) # [L]
            vs_sum = torch.einsum("l,blhd->bhd", all_ones, vs) # [B, H, D]
            attention_num += vs_sum.unsqueeze(1) # [B, N, H, D]

            # denominator
            all_ones = torch.ones([ks.shape[1]]).to(ks.device) # [L]
            ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones) # [B, H, M]
            attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, N, H]

            # attentive aggregated results
            attention_normalizer = attention_normalizer.unsqueeze(-1)  # [B, N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer # [B, N, H, D]

            if output_attn:
                attention = torch.einsum("bnhm,blhm->bnlh", qs, ks) / attention_normalizer # [B, N, L, H]
        else:
            # Original unbatched implementation
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
            attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
            attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer # [N, H, D]

            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer # [N, L, H]

    elif kernel == 'sigmoid':
        if is_batched:
            # Batched sigmoid implementation
            attention_num = torch.sigmoid(torch.einsum("bnhm,blhm->bnlh", qs, ks)) # [B, N, L, H]

            all_ones = torch.ones([ks.shape[1]]).to(ks.device)
            attention_normalizer = torch.einsum("bnlh,l->bnh", attention_num, all_ones)
            attention_normalizer = attention_normalizer.unsqueeze(2) # [B, N, 1, H]

            attention = attention_num / (attention_normalizer + 1e-6)
            attn_output = torch.einsum("bnlh,blhd->bnhd", attention, vs)
        else:
            # Original unbatched implementation
            attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
            attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]

            # compute attention and attentive aggregated results
            attention = attention_num / attention_normalizer
            attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, edge_index, edge_weight):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple',
               use_graph=True,
               use_weight=True):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # Check for batched input [B, L, D]
        is_batched = query_input.dim() == 3

        # feature transformation
        query = self.Wq(query_input)
        key = self.Wk(source_input)
        if self.use_weight:
            value = self.Wv(source_input)
        else:
            if is_batched:
                value = source_input.reshape(source_input.shape[0], source_input.shape[1], 1, self.out_channels)
            else:
                value = source_input.reshape(-1, 1, self.out_channels)

        if is_batched:
            B, L, _ = query_input.shape
            query = query.reshape(B, L, self.num_heads, self.out_channels)
            key = key.reshape(B, L, self.num_heads, self.out_channels)
            if self.use_weight:
                value = value.reshape(B, L, self.num_heads, self.out_channels)
        else:
            query = query.reshape(-1, self.num_heads, self.out_channels)
            key = key.reshape(-1, self.num_heads, self.out_channels)
            if self.use_weight:
                value = value.reshape(-1, self.num_heads, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)
        else:
            attention_output = full_attention_conv(query,key,value,self.kernel)

        # use input graph for gcn conv
        if self.use_graph:
            # GCN expects [N, H, D], so flatten batched input then reshape back.
            if is_batched:
                value_flat = value.reshape(-1, self.num_heads, self.out_channels)
                gcn_out = gcn_conv(value_flat, edge_index, edge_weight)
                gcn_out = gcn_out.reshape(B, L, self.num_heads, self.out_channels)
                final_output = attention_output + gcn_out
            else:
                final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output

        # Mean over head dimension (works for batched and unbatched)
        final_output = final_output.mean(dim=-2)

        if output_attn:
            return final_output, attn
        else:
            return final_output

class DIFFormer(nn.Module):
    '''
    DIFFormer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True):
        super(DIFFormer, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel, use_graph=use_graph, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with DIFFormer layer
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)
        return x_out

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0) # [layer num, N, N]
