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

class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution (MS-TCN)
    Captures temporal dependencies at multiple scales using different dilations.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2], residual=True):
        super(MultiScaleTemporalConv, self).__init__()
        
        # Branch 1: 1x1 Conv (Dimensionality reduction / preservation)
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))

        # Branch 2: k x 1 Conv, Dilation = d1
        pad2 = (kernel_size + (kernel_size - 1) * (dilations[0] - 1) - 1) // 2
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                                 padding=(pad2, 0), stride=(stride, 1), dilation=(dilations[0], 1))

        # Branch 3: k x 1 Conv, Dilation = d2
        pad3 = (kernel_size + (kernel_size - 1) * (dilations[1] - 1) - 1) // 2
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                                 padding=(pad3, 0), stride=(stride, 1), dilation=(dilations[1], 1))
        
        # Branch 4: Max Pooling + 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels), # BN before 1x1? Usually after. Let's stick closer to Inception.
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )

        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()
        
        # Project back to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.residual = residual
        if residual and (in_channels != out_channels or stride != 1):
             self.res_conv = nn.Sequential(
                 nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                 nn.BatchNorm2d(out_channels)
             )
        else:
             self.res_conv = lambda x: x

    def forward(self, x):
        res = self.res_conv(x)
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)
        x_out = self.project(self.relu(self.bn(x_concat)))
        
        if self.residual:
            x_out = x_out + res
            
        return F.relu(x_out)


class AdaptiveGraphConv(nn.Module):
    """
    Stabilized Adaptive Graph Convolution
    A_final = (A_physical + B_learnable) + alpha * C_data_dependent
    Maintains the 3-subset partition structure.
    """
    def __init__(self, in_channels, out_channels, A):
        super(AdaptiveGraphConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. Structural Path (Physical Graph + Learnable Bias)
        self.num_subsets = A.shape[0] # Usually 3
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(A.shape), requires_grad=True) # Learnable global graph
        
        # 2. Data-Dependent Path (Correlation)
        inter_channels = max(in_channels // 8, 2)
        self.conv_theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_phi = nn.Conv2d(in_channels, inter_channels, 1)
        
        # Learnable scalar to gate the adaptive attention (START AT ZERO for stability)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Main Convolution
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, kernel_size=1)
        
        # Residual
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        nn.init.constant_(self.B, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        
        # --- Compute Data-Dependent Graph ---
        # Theta: N, C', T, V -> NT, V, C'
        theta = self.conv_theta(x).permute(0, 2, 3, 1).contiguous().view(N*T, V, -1)
        # Phi: N, C', T, V -> NT, C', V
        phi = self.conv_phi(x).permute(0, 2, 3, 1).contiguous().view(N*T, -1, V)
        
        # Correlation: (NT, V, C') @ (NT, C', V) -> (NT, V, V)
        C_data = torch.matmul(theta, phi)
        # Normalize (Softmax over V)
        C_data = F.softmax(C_data, dim=-1) # NT, V, V
        
        # Reshape to (N, T, V, V)
        C_data = C_data.view(N, T, V, V)
        
        # --- Prepare Combined Graph ---
        # A_static: (3, V, V) -> broadcast to (N, 3, T, V, V)
        A_static = self.A + self.B 
        
        # We want A_final: (N, 3, T, V, V)
        # A_static part: (1, 3, 1, V, V)
        # C_data part: (N, 1, T, V, V)
        # Formula: A_total = (A+B) + alpha * C_data
        # C_data is broadcasted across the 3 subsets (shared adaptive graph)
        A_final = A_static.view(1, 3, 1, V, V) + self.alpha * C_data.unsqueeze(1) # Broadcast addition
        
        # --- Convolution ---
        x_res = self.down(x)
        
        x_conv = self.conv(x) # N, 3*C_out, T, V
        x_conv = x_conv.view(N, self.num_subsets, -1, T, V) # N, 3, C_out, T, V
        
        # --- Graph Application ---
        # Einsum: 
        # x: (N, K, C, T, V) where K=3
        # A: (N, K, T, V, W) where V=target, W=source (neighbor)
        # Result: (N, C, T, V)
        # Contraction: Sum over K (subsets) and W (neighbors)
        # 'nkctv, nktvw -> nctv'
        # Wait, standard graph mult is x @ A.T. 
        # x is vectors at nodes. A is adjacency.
        # x: (..., V). A: (V, V). -> (..., V) @ (V, V) = (..., V)
        # Here x_conv has structure (N, K, C, T, V).
        # A_final has structure (N, K, T, V, W)
        
        # Correct einsum for graph convolution:
        # For each subset k, and time t:
        # y[v] = sum_w (x[w] * A[v, w])
        # Indices:
        # n: batch
        # k: subset
        # c: channel
        # t: time
        # v: target node
        # w: source node
        
        # x: n k c t w
        # A: n k t v w
        # -> n k c t v (before summing k)
        # -> n c t v (after summing k)
        
        # Let's align dimensions for einsum:
        # x_conv: n k c t w
        # A_final: n k t v w
        y = torch.einsum('nkctw, nktvw -> nctv', (x_conv, A_final))
        
        return self.relu(self.bn(y) + x_res)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_point):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: N, C, T, V
        # pool time -> N, C, 1, V
        attn = self.conv1(x.mean(dim=2, keepdim=True)) 
        attn = self.sigmoid(attn)
        return x * attn

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1)) 
        self.conv1 = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V = x.size()
        y = self.avg_pool(x).squeeze(-1) # N, C, T
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y).unsqueeze(-1) # N, C, T, 1
        return x * y

class AdaptiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dilation=1, residue=True):
        super(AdaptiveBlock, self).__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)
        
        # Use Multi-Scale TCN instead of simple TCN
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels, kernel_size=3, stride=stride, dilations=[1, 2])
        
        # Attention Modules
        self.s_attn = SpatialAttention(out_channels, A.shape[1])
        self.t_attn = TemporalAttention(out_channels)
        self.residue = residue
        
        if residue and (in_channels != out_channels or stride != 1):
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
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
                res = self.res_conv(res)
            x = x + res
            
        return F.relu(x)

class Model(nn.Module):
    """
    Improved Adaptive GNN with MS-TCN and Stabilized Adapative Graph
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

        # Layers with Improved Adaptive Blocks
        self.layers = nn.ModuleList([
            AdaptiveBlock(in_channels, 64, A, stride=1),
            AdaptiveBlock(64, 64, A, stride=1),
            AdaptiveBlock(64, 64, A, stride=1),
            AdaptiveBlock(64, 128, A, stride=2),
            AdaptiveBlock(128, 128, A, stride=1),
            AdaptiveBlock(128, 128, A, stride=1),
            AdaptiveBlock(128, 256, A, stride=2),
            AdaptiveBlock(256, 256, A, stride=1),
            AdaptiveBlock(256, 256, A, stride=1),
            AdaptiveBlock(256, 256, A, stride=1) 
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

        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        x = self.fc(x)

        return x
