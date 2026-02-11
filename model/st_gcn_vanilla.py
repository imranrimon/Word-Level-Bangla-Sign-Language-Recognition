import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class GraphConv(nn.Module):
    """Vanilla Graph Convolution - No learnable adjacency"""
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subsets = A.shape[0]
        
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
        A = self.A.to(x.device)
        
        x_res = self.down(x)
        
        x = self.conv(x)
        x = x.view(N, self.num_subsets, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return self.relu(self.bn(x) + x_res)

class TemporalConv(nn.Module):
    """Vanilla Temporal Convolution - No dilation"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1), 
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ST_GCN_Block(nn.Module):
    """Vanilla ST-GCN Block - No attention, no dilations"""
    def __init__(self, in_channels, out_channels, A, stride=1, residue=True):
        super(ST_GCN_Block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride)
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
        
        if self.residue:
            if self.res_conv:
                res = self.res_bn(self.res_conv(res))
            x = x + res
            
        return F.relu(x)

class Model(nn.Module):
    """
    Vanilla ST-GCN Baseline
    Pure Spatial-Temporal Graph Convolution
    No attention, no learnable graph, no dilations
    Based on original ST-GCN paper
    """
    def __init__(self, num_class=60, num_point=27, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # Standard ST-GCN architecture
        self.layers = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, A, stride=1),
            ST_GCN_Block(64, 64, A, stride=1),
            ST_GCN_Block(64, 64, A, stride=1),
            ST_GCN_Block(64, 128, A, stride=2),
            ST_GCN_Block(128, 128, A, stride=1),
            ST_GCN_Block(128, 128, A, stride=1),
            ST_GCN_Block(128, 256, A, stride=2),
            ST_GCN_Block(256, 256, A, stride=1),
            ST_GCN_Block(256, 256, A, stride=1),
        ])

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

        # Global Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        x = self.fc(x)

        return x
