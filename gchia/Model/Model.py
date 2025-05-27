import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from gchia.Model.Blocks import PositionalEncoding, ConvBlock, ResBlockDilated, GCNLayer
from gchia.Model.difformer import DIFFormer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class GraphChIAr(nn.Module):
    def __init__(
        self,
        resolution,
        window_size,
        filter_size = 5,
        n_difformer_layers=2,
        num_heads=4,
        dropout=0.1,
        hidden_dim=32,
        feature_dim=1
    ):
        super().__init__()
        self.matrix_size = window_size // resolution
        self.resolution = resolution
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim + 5
        self.conv_blocks = self.get_conv_blocks(filter_size, resolution,self.hidden_dim)
        
        # Original DIFFormer component
        self.difformer = DIFFormer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=n_difformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=True,
            use_residual=True
        )
        
        self.pos_encoder = PositionalEncoding(self.hidden_dim // 2)
        self.scale_conv = nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, 1)
        self.res_blocks = self.get_res_blocks(3, self.hidden_dim)  # Increased input channels for additional features
        
        # Adjacency matrix feature extraction
        self.adj_conv = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Final prediction layers
        self.mlp = nn.Sequential(
            nn.Conv2d(self.hidden_dim , self.hidden_dim , 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim//2, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, 1, 1),
            nn.ReLU()
        )
    
    def get_conv_blocks(self, filter, resolution,hidden_dim):
        if(self.feature_dim < 8):
            hidden = [self.feature_dim, 8, 8]
        else:
            hidden = [self.feature_dim, 16, 16]
        tmp = resolution
        blocks = []
        for i in range(3, math.floor(math.log10(resolution)) + 1):
            hidden.append(hidden_dim // 2)
        for i in range(len(hidden) - 1):
            blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[i], hidden=hidden[i+1]))
            tmp /= 10
        if not math.log10(resolution).is_integer():
            if tmp == 2:
                blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[-1], hidden=hidden_dim // 2, Maxpool=False))
            elif tmp == 5:
                blocks.append(ConvBlock(filter, stride=1, hidden_in=hidden[-1], hidden=hidden_dim // 2))
        return nn.Sequential(*blocks)
        
    def get_res_blocks(self, n, hidden):
        blocks = []
        blocks.append(ResBlockDilated(3, hidden=hidden, dil=1))
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(3, hidden=hidden, dil=dilation))
        return nn.Sequential(*blocks)
    
    def create_adjacency_matrix(self, edge_index, edge_weights, batch_size, seq_len):
        """Convert edge_index and weights to adjacency matrix using vectorized operations"""
        # Create batch adjacency matrices
        adj_matrices = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        
        # Calculate which graph each edge belongs to
        batch_idx = edge_index[0] // seq_len
        
        # Calculate node indices within their respective graphs
        node_idx_in_batch = edge_index - (batch_idx * seq_len).repeat(2, 1)
        
        # Use scatter_add_ to batch update all adjacency matrices
        adj_matrices.index_put_(
            (batch_idx, node_idx_in_batch[0], node_idx_in_batch[1]),
            edge_weights,
            # accumulate=True
        )
        
        return adj_matrices.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
    
    def forward(self, data):
        x, seq, edge_index, edge_attr, batch = data.features, data.seq, data.edge_index, data.edge_attr, data.batch
        
        # Original feature processing pipeline
        
        x = torch.cat([x, seq], dim=1)
        
        x = self.conv_blocks(x)
        x = self.pos_encoder(x)
        
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.size(2))
        
        edge_weights = edge_attr.float() if edge_attr is not None else torch.ones(edge_index.size(1), device=edge_index.device)
        x = self.difformer(x, edge_index, edge_weights)
        
        graph_nums = len(torch.unique(batch))
        batch_size = x.size(0)
        x = x.view(graph_nums, batch_size // graph_nums, -1).transpose(1, 2)
        
        # Create pair features
        node_i = x.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        node_j = x.unsqueeze(3).expand(-1, -1, -1, x.size(2))
        pair_features = torch.cat([node_i, node_j], dim=1)
        # logging.info(f"Pair features shape: {pair_features.shape}")
        # Process adjacency matrix features using optimized create_adjacency_matrix
        adj_matrix = self.create_adjacency_matrix(edge_index, edge_weights, graph_nums, batch_size // graph_nums)
        adj_features = self.adj_conv(adj_matrix)  # [batch, hidden_dim, seq_len, seq_len]
        # logging.info(f"Adjacency features shape: {adj_features.shape}")
        # Combine features
        
        combined_features = torch.cat([pair_features, adj_features], dim=1)
        combined_features = self.scale_conv(combined_features)
        # Apply residual blocks
        combined_features = self.res_blocks(combined_features)
        
        # Final prediction
        matrix_pred = self.mlp(combined_features).squeeze(1)
        # matrix_pred = matrix_pred + matrix_pred.transpose(1, 2)
        
        return matrix_pred
    
    

class GraphChIArNoSeq(nn.Module):
    def __init__(
        self,
        resolution,
        window_size,
        filter_size = 5,
        n_difformer_layers=2,
        num_heads=4,
        dropout=0.1,
        hidden_dim=32,
        feature_dim=1
    ):
        super().__init__()
        self.matrix_size = window_size // resolution
        self.resolution = resolution
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.conv_blocks = self.get_conv_blocks(filter_size, resolution,self.hidden_dim)
        
        # Original DIFFormer component
        self.difformer = DIFFormer(
            in_channels=self.hidden_dim // 2,
            hidden_channels=self.hidden_dim // 2,
            out_channels=self.hidden_dim // 2,
            num_layers=n_difformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=True,
            use_residual=True
        )
        
        # self.pos_encoder = PositionalEncoding(self.hidden_dim // 2)
        self.scale_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1)
        self.res_blocks = self.get_res_blocks(3, self.hidden_dim)  # Increased input channels for additional features
        
        # Adjacency matrix feature extraction
        self.adj_conv = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Final prediction layers
        self.mlp = nn.Sequential(
            nn.Conv2d(self.hidden_dim , self.hidden_dim , 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim//2, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, 1, 1),
            nn.ReLU()
        )
    
    def get_conv_blocks(self, filter, resolution,hidden_dim):
        if(self.feature_dim < 8):
            hidden = [self.feature_dim, 8, 8]
        else:
            hidden = [self.feature_dim, 16, 16]
        tmp = resolution
        blocks = []
        for i in range(3, math.floor(math.log10(resolution)) + 1):
            hidden.append(hidden_dim // 2)
        for i in range(len(hidden) - 1):
            blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[i], hidden=hidden[i+1]))
            tmp /= 10
        if not math.log10(resolution).is_integer():
            if tmp == 2:
                blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[-1], hidden=hidden_dim // 2, Maxpool=False))
            elif tmp == 5:
                blocks.append(ConvBlock(filter, stride=1, hidden_in=hidden[-1], hidden=hidden_dim // 2))
        return nn.Sequential(*blocks)
        
    def get_res_blocks(self, n, hidden):
        blocks = []
        blocks.append(ResBlockDilated(3, hidden=hidden, dil=1))
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(3, hidden=hidden, dil=dilation))
        return nn.Sequential(*blocks)
    
    def create_adjacency_matrix(self, edge_index, edge_weights, batch_size, seq_len):
        """Convert edge_index and weights to adjacency matrix using vectorized operations"""
        # Create batch adjacency matrices
        adj_matrices = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        
        # Calculate which graph each edge belongs to
        batch_idx = edge_index[0] // seq_len
        
        # Calculate node indices within their respective graphs
        node_idx_in_batch = edge_index - (batch_idx * seq_len).repeat(2, 1)
        
        # Use scatter_add_ to batch update all adjacency matrices
        adj_matrices.index_put_(
            (batch_idx, node_idx_in_batch[0], node_idx_in_batch[1]),
            edge_weights,
            # accumulate=True
        )
        
        return adj_matrices.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
    
    def forward(self, data):
        x, seq, edge_index, edge_attr, batch = data.features, data.seq, data.edge_index, data.edge_attr, data.batch
        
        # Original feature processing pipeline
        
        # x = torch.cat([x, seq], dim=1)
        
        x = self.conv_blocks(x)
        # x = self.pos_encoder(x)
        
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.size(2))
        
        edge_weights = edge_attr.float() if edge_attr is not None else torch.ones(edge_index.size(1), device=edge_index.device)
        x = self.difformer(x, edge_index, edge_weights)
        
        graph_nums = len(torch.unique(batch))
        batch_size = x.size(0)
        x = x.view(graph_nums, batch_size // graph_nums, -1).transpose(1, 2)
        
        # Create pair features
        node_i = x.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        node_j = x.unsqueeze(3).expand(-1, -1, -1, x.size(2))
        pair_features = torch.cat([node_i, node_j], dim=1)
        # logging.info(f"Pair features shape: {pair_features.shape}")
        # Process adjacency matrix features using optimized create_adjacency_matrix
        adj_matrix = self.create_adjacency_matrix(edge_index, edge_weights, graph_nums, batch_size // graph_nums)
        adj_features = self.adj_conv(adj_matrix)  # [batch, hidden_dim, seq_len, seq_len]

        # Combine features
        
        combined_features = torch.cat([pair_features, adj_features], dim=1)
        combined_features = self.scale_conv(combined_features)
        # Apply residual blocks
        combined_features = self.res_blocks(combined_features)
        
        # Final prediction
        matrix_pred = self.mlp(combined_features).squeeze(1)

        
        return matrix_pred


class Caesar_pytorch(nn.Module):
    def __init__(
        self,
        resolution,
        window_size,
        n_gc_layers = 2,
        gc_dim=32,
        n_conv_layers=2,
        conv_dim=32,
        pe_dim=8,
        feature_dim=1
    ):
        super().__init__()
        self.matrix_size = window_size // resolution
        self.resolution = resolution
        self.gc_dim = gc_dim
        self.conv_dim = conv_dim
        self.feature_dim = feature_dim
        self.avgpool = nn.AvgPool1d(kernel_size=resolution, stride=resolution)
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(GCNLayer(self.feature_dim + pe_dim, self.gc_dim))
        for _ in range(n_gc_layers - 1):
            self.gc_layers.append(GCNLayer(self.gc_dim, self.gc_dim))
        self.pos_encoder = PositionalEncoding(pe_dim)
        self.node_conv_layers = nn.ModuleList()
        self.node_conv_layers.append(
            nn.Sequential(
                nn.Conv1d(self.feature_dim + pe_dim, self.conv_dim, kernel_size=15,padding=7),
                nn.BatchNorm1d(self.conv_dim),
                nn.ReLU()
            )
        )
        for _ in range(n_conv_layers - 1):
            self.node_conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=15,padding=7),
                    nn.BatchNorm1d(self.conv_dim),
                    nn.ReLU()
                )
            )
        mlp_input_dim =(self.conv_dim + self.gc_dim) * 2
        # Final prediction layers
        self.mlp = nn.Sequential(
            nn.Conv2d( mlp_input_dim,mlp_input_dim , 1),
            nn.ReLU(),
            nn.Conv2d( mlp_input_dim, mlp_input_dim//2, 1),
            nn.ReLU(),
            nn.Conv2d( mlp_input_dim//2, 1, 1),
            nn.ReLU()
        )
    
    
    def forward(self, data):
        x, seq, edge_index, edge_attr, batch = data.features, data.seq, data.edge_index, data.edge_attr, data.batch
        
        
        x = self.avgpool(x)
        
        x = self.pos_encoder(x)
        conv_out = x
        for i, node_conv_layer in enumerate(self.node_conv_layers):
            conv_out = node_conv_layer(conv_out)
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.size(2))
        
        edge_weights = edge_attr.float() if edge_attr is not None else torch.ones(edge_index.size(1), device=edge_index.device)
        for i, gc_layer in enumerate(self.gc_layers):
            x = gc_layer(x, edge_index, edge_weights)
        
        graph_nums = len(torch.unique(batch))
        batch_size = x.size(0)
        x = x.view(graph_nums, batch_size // graph_nums, -1).transpose(1, 2)
        x = torch.cat([x, conv_out], dim=1)
        # Create pair features
        node_i = x.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        node_j = x.unsqueeze(3).expand(-1, -1, -1, x.size(2))
        pair_features = torch.cat([node_i, node_j], dim=1)
        
        # Final prediction
        matrix_pred = self.mlp(pair_features).squeeze(1)
        # matrix_pred = matrix_pred + matrix_pred.transpose(1, 2)
        
        return matrix_pred
class GraphChIAr_super(nn.Module):
    def __init__(
        self,
        resolution,
        window_size,
        filter_size = 5,
        n_difformer_layers=2,
        num_heads=4,
        dropout=0.1,
        hidden_dim=32,
        feature_dim=1
    ):
        super().__init__()
        self.matrix_size = window_size // resolution
        self.resolution = resolution
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim + 5
        self.conv_blocks = self.get_conv_blocks(filter_size, resolution,self.hidden_dim)
        
        # Original DIFFormer component
        self.difformer = DIFFormer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=n_difformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bn=True,
            use_residual=True
        )
        
        self.pos_encoder = PositionalEncoding(self.hidden_dim // 2)
        self.scale_conv = nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, 1)
        # self.res_blocks = self.get_res_blocks(3, self.hidden_dim)  # Increased input channels for additional features
        
        # Adjacency matrix feature extraction
        self.adj_conv = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Final prediction layers
        self.mlp = nn.Sequential(
            nn.Conv2d(self.hidden_dim , self.hidden_dim , 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim//2, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim//2, 1, 1),
            nn.ReLU()
        )
    
    def get_conv_blocks(self, filter, resolution,hidden_dim):
        hidden = [self.feature_dim, hidden_dim//2]
        tmp = resolution
        blocks = []
        for i in range(2, math.floor(math.log10(resolution)) + 1):
            hidden.append(hidden_dim // 2)
        for i in range(len(hidden) - 1):
            blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[i], hidden=hidden[i+1]))
            tmp /= 10
        if not math.log10(resolution).is_integer():
            if tmp == 2:
                blocks.append(ConvBlock(filter, stride=2, hidden_in=hidden[-1], hidden=hidden[-1], Maxpool=False))
            elif tmp == 5:
                blocks.append(ConvBlock(filter, stride=1, hidden_in=hidden[-1], hidden=hidden[-1]))
        return nn.Sequential(*blocks)
        
    def get_res_blocks(self, n, hidden):
        blocks = []
        blocks.append(ResBlockDilated(3, hidden=hidden, dil=1))
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(3, hidden=hidden, dil=dilation))
        return nn.Sequential(*blocks)
    
    def create_adjacency_matrix(self, edge_index, edge_weights, batch_size, seq_len):
        """Convert edge_index and weights to adjacency matrix using vectorized operations"""
        # Create batch adjacency matrices
        adj_matrices = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        
        # Calculate which graph each edge belongs to
        batch_idx = edge_index[0] // seq_len
        
        # Calculate node indices within their respective graphs
        node_idx_in_batch = edge_index - (batch_idx * seq_len).repeat(2, 1)
        
        # Use scatter_add_ to batch update all adjacency matrices
        adj_matrices.index_put_(
            (batch_idx, node_idx_in_batch[0], node_idx_in_batch[1]),
            edge_weights,
            # accumulate=True
        )
        
        return adj_matrices.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
    
    def forward(self, data):
        x, seq, edge_index, edge_attr, batch = data.features, data.seq, data.edge_index, data.edge_attr, data.batch
        
        # Original feature processing pipeline
        
        x = torch.cat([x, seq], dim=1)
        # logging.info(f"X shape: {x.shape}")
        x = self.conv_blocks(x)
        x = self.pos_encoder(x)
        # logging.info(f"X shape: {x.shape}")
        x = x.transpose(1, 2)
        x = x.reshape(-1, x.size(2))
        
        edge_weights = edge_attr.float() if edge_attr is not None else torch.ones(edge_index.size(1), device=edge_index.device)
        # logging.info(f"X shape: {x.shape}")
        x = self.difformer(x, edge_index, edge_weights)
        
        graph_nums = len(torch.unique(batch))
        batch_size = x.size(0)
        x = x.view(graph_nums, batch_size // graph_nums, -1).transpose(1, 2)
        adj_matrix = self.create_adjacency_matrix(edge_index, edge_weights, graph_nums, batch_size // graph_nums)
        # logging.info(f"adj shape: {adj_matrix.shape}")
        # Create pair features
        node_i = x.unsqueeze(2).expand(-1, -1, x.size(2), -1)
        node_j = x.unsqueeze(3).expand(-1, -1, -1, x.size(2))
        pair_features = torch.cat([node_i, node_j], dim=1)
        # logging.info(f"Pair features shape: {pair_features.shape}")
        # Process adjacency matrix features using optimized create_adjacency_matrix
        
        adj_features = self.adj_conv(adj_matrix)  # [batch, hidden_dim, seq_len, seq_len]
        # logging.info(f"Adjacency features shape: {adj_features.shape}")
        # Combine features
        
        combined_features = torch.cat([pair_features, adj_features], dim=1)
        combined_features = self.scale_conv(combined_features)
        # Apply residual blocks
        #combined_features = self.res_blocks(combined_features)
        
        # Final prediction
        matrix_pred = self.mlp(combined_features).squeeze(1)
        # matrix_pred = matrix_pred + matrix_pred.transpose(1, 2)
        
        return matrix_pred
    