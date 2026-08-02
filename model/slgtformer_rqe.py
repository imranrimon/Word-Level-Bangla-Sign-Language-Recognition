"""SLGTFormer (twin_attention) with Relative Quantization Encoding injected.

Wraps `model.twin_attention.Model` and adds RQE as an additive positional prior
at the input, parallel to `model.block_gcn_rqe`. Separate from LGRPE — the
backbone's `use_grpe` flag can be True (RQE + LGRPE) or False (RQE only).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.twin_attention import Model as SLGTFormer
from model.rqe import RelativeQuantizationEncoding


class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=27,
        num_person=1,
        graph=None,
        graph_args=None,
        in_channels=3,
        rqe_embed_dim=64,
        rqe_num_buckets=16,
        rqe_right_shoulder_idx=1,
        rqe_left_shoulder_idx=2,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = SLGTFormer(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args or {},
            in_channels=in_channels,
            **backbone_kwargs,
        )
        self.rqe = RelativeQuantizationEncoding(
            embed_dim=rqe_embed_dim,
            num_buckets=rqe_num_buckets,
            right_shoulder_idx=rqe_right_shoulder_idx,
            left_shoulder_idx=rqe_left_shoulder_idx,
        )
        self.rqe_proj = nn.Conv2d(rqe_embed_dim, in_channels, kernel_size=1)

    def forward(self, x, keep_prob=0.9):
        rqe = self.rqe(x)              # (N, D, T, V)
        rqe = self.rqe_proj(rqe)       # (N, C, T, V)
        x = x + rqe.unsqueeze(-1)      # broadcast across M
        return self.backbone(x, keep_prob=keep_prob)
