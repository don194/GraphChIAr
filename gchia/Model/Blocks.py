import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import logging
class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64, Maxpool = True):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.Maxpool = Maxpool
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        if self.Maxpool:
            self.pool = nn.MaxPool1d(5)
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        if self.Maxpool:
            out = self.pool(out)
        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(0)  # [1, max_len]
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(1, hidden, max_len)  # [1, hidden, max_len]
        pe[0, 0::2, :] = torch.sin(position * div_term.unsqueeze(1))
        pe[0, 1::2, :] = torch.cos(position * div_term.unsqueeze(1))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim, seq_len]
        """
        
        # x = x + self.pe[:, :, :x.size(2)]
        # cat the positional encoding to the input
        # reshape pe to match the input
        # logging.info(f"PE shape: {self.pe.shape}")
        # logging.info(f"Input shape: {x.shape}")
        x = torch.cat((x, self.pe[:, :, :x.size(2)].expand(x.size(0), -1, -1)), dim=1)
        return self.dropout(x)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weights):
        
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        if edge_weights is not None:
            edge_weights = edge_weights[mask]
            
        x = self.conv(x, edge_index, edge_weights)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    


class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden , stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

