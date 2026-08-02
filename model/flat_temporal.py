"""A deliberately-plain temporal transformer over flattened per-frame features.

Exists so that pose-based inputs (MediaPipe 27-keypoint, C=3, V=27) and
DINOv2 hand/face-crop inputs (C=384, V=3) can be fed through the SAME
architecture for a clean feature-isolation experiment (Option B). No graph
structure is assumed — per-frame inputs are just flattened into a token.

Input contract matches the rest of the repo:
    forward(x, keep_prob=1.0)  where x is (N, C, T, V, M)
                               -> (N, num_class) logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (N, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=27,
        num_person=1,
        in_channels=3,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_frames=300,
        graph=None,            # accepted and ignored; preserves YAML compatibility
        graph_args=None,       # idem
    ):
        super().__init__()
        del graph, graph_args
        self.num_point = int(num_point)
        self.num_person = int(num_person)
        self.in_channels = int(in_channels)
        input_dim = self.num_point * self.num_person * self.in_channels

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _SinusoidalPositionalEmbedding(max_frames, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_class)

    def forward(self, x, keep_prob=1.0):
        # x: (N, C, T, V, M)
        del keep_prob
        if x.dim() != 5:
            raise ValueError(f"expected 5D (N,C,T,V,M), got {x.dim()}D")
        N, C, T, V, M = x.shape
        # (N, C, T, V, M) -> (N, T, V*M*C)
        flat = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, V * M * C)
        h = self.input_proj(flat)                      # (N, T, d_model)
        h = self.pos_enc(h)
        # Build a key_padding_mask for all-zero time steps (the feeder pads
        # short clips with zeros). A time step is "padding" if every input
        # feature is exactly zero. Sparse but useful when T_max is large.
        with torch.no_grad():
            padding = (flat.abs().sum(dim=-1) == 0)   # (N, T) bool
        h = self.encoder(h, src_key_padding_mask=padding)
        h = self.norm(h)
        # mean pool over non-padding time steps
        mask = (~padding).float().unsqueeze(-1)       # (N, T, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (h * mask).sum(dim=1) / denom
        return self.fc(pooled)
