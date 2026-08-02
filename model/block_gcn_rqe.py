"""BlockGCN backbone with Relative Quantization Encoding injected at the input.

The RQE module (see model.rqe) produces a (N, D, T, V) positional encoding
from the raw keypoint coordinates. We project it down to `in_channels` via
a 1x1 conv and add it to the input, broadcast across persons, before the
standard BlockGCN forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.block_gcn import Model as BlockGCN
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
        self.backbone = BlockGCN(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args,
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

    def forward(self, x, keep_prob=1.0):
        # x: (N, C, T, V, M). keep_prob forwarded to backbone for harness
        # compatibility (currently a no-op inside BlockGCN).
        rqe = self.rqe(x)                        # (N, D, T, V)
        rqe = self.rqe_proj(rqe)                 # (N, C, T, V)
        x = x + rqe.unsqueeze(-1)                # broadcast across M
        return self.backbone(x, keep_prob=keep_prob)
