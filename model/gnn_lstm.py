
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
        # x: N, C, T, V
        attn = self.conv1(x) # N, 1, T, V
        attn = self.sigmoid(attn)
        return x * attn

class GCN_Backbone(nn.Module):
    """
    Extracts spatial features from every frame independently using GCNs.
    """
    def __init__(self, in_channels, out_channels, A):
        super(GCN_Backbone, self).__init__()
        
        # Spatial Graph Convolutions
        self.gcn1 = GraphConv(in_channels, 64, A)
        self.gcn2 = GraphConv(64, 64, A)
        self.gcn3 = GraphConv(64, 128, A)
        self.gcn4 = GraphConv(128, out_channels, A)
        
        self.s_attn = SpatialAttention(out_channels, A.shape[1])

    def forward(self, x):
        # x: N, C, T, V
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.gcn4(x)
        
        x = self.s_attn(x)
        
        # Pool over vertices -> (N, C, T, 1)
        x = F.avg_pool2d(x, (1, x.size(3))) 
        return x.squeeze(-1) # N, C, T

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

        # 2. Temporal Feature Extractor (Bi-LSTM)
        self.lstm_hidden_dim = 256
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=self.gcn_out_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        # 3. Classifier
        self.fc = nn.Linear(self.lstm_hidden_dim * 2, num_class) # *2 for bidirectional

    def forward(self, x, keep_prob=1.0):
        # x: N, C, T, V, M
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 1, 2).contiguous().view(N, C, T, V * M)

        # Need to process all frames. 
        # But GCN expects (N, C, T, V). 
        # Since M=1 usually, we reuse the N dimension if M>1 or just assume V*M is the graph?
        # The standard layout handles M inside V usually, or we avg M.
        # Let's align with the previous Attention GNN:
        # It did: view(N, C, T, V * M) -> GraphConv.
        # So we treat all people as nodes in one big graph if M>1, OR we sum M.
        # AttentionGNN code assumes V is the graph nodes. 
        # If we passed M nodes into V, the graph adjacency A must match.
        # The data loader typically returns fixed V=27.
        # If M=2, we usually process them separately and max-pool, or the graph is 54 nodes.
        # The current config uses 'graph.bdsl.Graph' which typically defines 27 nodes.
        # So if M=2, we should probably average or max over M at start.
        
        # Taking max over M (num_person)
        if M > 1:
            x = x.view(N, M, C, T, V)
            x = x.mean(dim=1) # N, C, T, V
        else:
            x = x.view(N, C, T, V)

        # 1. Spatial Encoding
        # x: N, C, T, V
        x_spatial = self.spatial_encoder(x) # N, C, T
        
        # Prepare for LSTM: (N, T, C)
        x_seq = x_spatial.permute(0, 2, 1) # N, T, C

        # 2. Temporal Encoding (LSTM)
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x_seq) # N, T, 2*hidden
        
        # 3. Classification
        # Take the last time step? Or Avg pooling over time?
        # Standard for action recognition is often Max or Avg pool over T
        x_pool = torch.mean(x_lstm, dim=1) # N, 2*hidden
        
        out = self.fc(x_pool)
        return out
