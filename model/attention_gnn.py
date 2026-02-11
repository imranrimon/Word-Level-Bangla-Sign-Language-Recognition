import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph import tools

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

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_point):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: N, C, T, V
        attn = self.conv1(x) # N, 1, T, V
        attn = self.sigmoid(attn)
        return x * attn

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1)) # over V
        self.conv1 = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: N, C, T, V
        N, C, T, V = x.size()
        y = self.avg_pool(x).squeeze(-1) # N, C, T
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y).unsqueeze(-1) # N, C, T, 1
        return x * y

        y = self.sigmoid(y).unsqueeze(-1) # N, C, T, 1
        return x * y

class GlobalSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(GlobalSelfAttention, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.bn = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        # x: N, C, T, V
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous() # N, T, V, C
        x = x.view(N, T * V, C) # Flatten spatial-temporal
        
        # Attention
        res = x
        x_att, _ = self.att(x, x, x)
        x = self.bn(x + x_att)
        
        x = x.view(N, T, V, C).permute(0, 3, 1, 2).contiguous() # N, C, T, V
        return x

class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dilation=1, residue=True):
        super(ST_GCN_Block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride, dilation=dilation)
        self.s_attn = SpatialAttention(out_channels, A.shape[1])
        self.t_attn = TemporalAttention(out_channels)
        self.residue = residue
        
        if residue and (in_channels != out_channels or stride != 1):
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1))
            self.res_bn = nn.BatchNorm2d(out_channels)
        else:
            self.res_conv = None

    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.s_attn(x)
        x = self.t_attn(x)
        
        if self.residue:
            if self.res_conv:
                res = self.res_bn(self.res_conv(res))
            x = x + res
            
        return F.relu(x)

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

        # Layers
        # Increasing dilation to capture long sequences
        self.layers = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, A, stride=1, dilation=1),
            ST_GCN_Block(64, 64, A, stride=1, dilation=1),
            ST_GCN_Block(64, 64, A, stride=1, dilation=1),
            ST_GCN_Block(64, 128, A, stride=2, dilation=2), # Downsample T
            ST_GCN_Block(128, 128, A, stride=1, dilation=2),
            ST_GCN_Block(128, 128, A, stride=1, dilation=2),
            ST_GCN_Block(128, 256, A, stride=2, dilation=4), # Downsample T
            ST_GCN_Block(256, 256, A, stride=1, dilation=4),
            ST_GCN_Block(256, 256, A, stride=1, dilation=4),
            ST_GCN_Block(256, 256, A, stride=1, dilation=8)  # High dilation for long range
        ])
        
        # Global Attention at Bottleneck
        self.global_attn = GlobalSelfAttention(256, num_heads=4)

        self.fc = nn.Linear(256, num_class)

    def forward(self, x, keep_prob=1.0):
        # x: N, C, T, V, M
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 1, 2).contiguous().view(N, C, T, V * M)

        for layer in self.layers:
            x = layer(x)

        # Apply Global Attention
        x = self.global_attn(x)


        # Global Pooling
        x = F.avg_pool2d(x, x.size()[2:]) # N, C, 1, 1
        x = x.view(N, -1)
        x = self.fc(x)

        return x
