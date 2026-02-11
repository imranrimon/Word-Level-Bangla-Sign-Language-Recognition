
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(A.shape), requires_grad=True) # Learnable Graph
        self.num_subsets = 3 
        
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, kernel_size=1)
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = (self.A + self.B).to(x.device) # 3, V, V (Original + Learned)
        
        x_res = self.down(x)
        
        x = self.conv(x)
        x = x.view(N, self.num_subsets, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return self.relu(self.bn(x) + x_res)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_point):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn = self.conv1(x) 
        attn = self.sigmoid(attn)
        return x * attn

class GCN_Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GCN_Backbone, self).__init__()
        
        self.gcn1 = GraphConv(in_channels, 64, A)
        self.gcn2 = GraphConv(64, 64, A)
        self.gcn3 = GraphConv(64, 128, A)
        self.gcn4 = GraphConv(128, out_channels, A)
        
        self.s_attn = SpatialAttention(out_channels, A.shape[1])

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.gcn4(x)
        x = self.s_attn(x)
        
        # Pool over vertices -> (N, C, T, 1)
        x = F.avg_pool2d(x, (1, x.size(3))) 
        return x.squeeze(-1) # N, C, T

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: T, N, E
        return x + self.pe[:x.size(0), :]

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=27, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 1. Spatial Feature Extractor (GCN)
        self.gcn_out_dim = 256
        self.spatial_encoder = GCN_Backbone(in_channels, self.gcn_out_dim, A)

        # 2. Temporal Feature Extractor (Transformer)
        self.d_model = 256
        self.num_heads = 4
        self.num_layers = 2
        self.dim_feedforward = 1024
        self.dropout = 0.5
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=300)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.num_heads, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            batch_first=False # PyTorch Transformer default is (T, N, E) usually better
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 3. Classifier
        self.fc = nn.Linear(self.d_model, num_class) 

    def forward(self, x, keep_prob=1.0):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 1, 2).contiguous().view(N, C, T, V * M)

        if M > 1:
            x = x.view(N, M, C, T, V)
            x = x.mean(dim=1)
        else:
            x = x.view(N, C, T, V)

        # 1. Spatial Encoding
        x_spatial = self.spatial_encoder(x) # N, C, T
        
        # Prepare for Transformer: Require (T, N, E)
        x_seq = x_spatial.permute(2, 0, 1) # T, N, C
        x_seq = x_seq * math.sqrt(self.d_model)
        x_seq = self.pos_encoder(x_seq)
        
        # 2. Temporal Encoding
        x_trans = self.transformer_encoder(x_seq) # T, N, E
        
        # 3. Classification
        # Mean pool over time
        x_pool = torch.mean(x_trans, dim=0) # N, E
        
        out = self.fc(x_pool)
        return out
