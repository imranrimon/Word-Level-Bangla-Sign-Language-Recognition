import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    """
    Pose-LSTM Baseline
    Flattens pose keypoints and processes as a temporal sequence with Bi-LSTM.
    No graph structure, purely sequential modeling.
    """
    def __init__(self, num_class=60, num_point=27, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        
        # Flatten spatial features: V * C
        self.input_dim = num_point * in_channels  # 27 * 3 = 81
        
        # Data normalization
        self.data_bn = nn.BatchNorm1d(self.input_dim)
        
        # Bi-Directional LSTM
        self.lstm_hidden = 256
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Classifier
        self.fc = nn.Linear(self.lstm_hidden * 2, num_class)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, keep_prob=1.0):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()
        
        # Average over multiple persons if M > 1
        if M > 1:
            x = x.mean(dim=4)  # (N, C, T, V)
        else:
            x = x.squeeze(4)  # (N, C, T, V)
        
        # Flatten spatial dimensions: (N, C, T, V) -> (N, T, C*V)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, T, C, V)
        x = x.view(N, T, C * V)  # (N, T, 81)
        
        # Normalize per frame
        x = x.permute(0, 2, 1).contiguous()  # (N, 81, T)
        x = self.data_bn(x)
        x = x.permute(0, 2, 1).contiguous()  # (N, T, 81)
        
        # LSTM
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (N, T, 512)
        
        # Global average pooling over time
        x = torch.mean(lstm_out, dim=1)  # (N, 512)
        
        # Dropout and classification
        x = self.dropout(x)
        out = self.fc(x)
        
        return out
