"""Relative Quantization Encoding (RQE) for skeleton-based sign recognition.

Inspired by the BdSLW401 paper (arXiv 2503.02360). Each joint's (x, y)
position is re-expressed relative to the shoulder midpoint, normalised by
shoulder width, then quantised into discrete buckets per axis and embedded.
The result is an approximately view-invariant positional encoding that can
be added to any per-token feature map of shape (N, D, T, V).

Assumes the 27-joint BdSL skeleton layout from `graph.sign_27`:
    index 1 -> right shoulder (MediaPipe 12)
    index 2 -> left  shoulder (MediaPipe 11)
Override via constructor args for other skeleton topologies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeQuantizationEncoding(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_buckets=16,
        right_shoulder_idx=1,
        left_shoulder_idx=2,
        clip_radius=2.0,
        eps=1e-6,
    ):
        super().__init__()
        if num_buckets < 2:
            raise ValueError("num_buckets must be >= 2")
        self.num_buckets = int(num_buckets)
        self.clip_radius = float(clip_radius)
        self.eps = float(eps)
        self.right_shoulder_idx = int(right_shoulder_idx)
        self.left_shoulder_idx = int(left_shoulder_idx)
        self.embed_x = nn.Embedding(num_buckets, embed_dim)
        self.embed_y = nn.Embedding(num_buckets, embed_dim)
        nn.init.trunc_normal_(self.embed_x.weight, std=0.02)
        nn.init.trunc_normal_(self.embed_y.weight, std=0.02)

    def forward(self, x):
        # x: (N, C, T, V, M), C >= 2 (x, y[, conf]); uses person 0 only.
        if x.dim() != 5:
            raise ValueError(f"expected 5D (N,C,T,V,M) input, got {x.dim()}D")
        if x.size(1) < 2:
            raise ValueError(f"RQE requires C >= 2, got C={x.size(1)}")

        pos = x[:, :2, :, :, 0]                                          # (N, 2, T, V)
        rs = pos[:, :, :, self.right_shoulder_idx:self.right_shoulder_idx + 1]
        ls = pos[:, :, :, self.left_shoulder_idx:self.left_shoulder_idx + 1]
        anchor = 0.5 * (rs + ls)                                         # (N, 2, T, 1)
        width = ((ls - rs) ** 2).sum(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
        rel = (pos - anchor) / (width + self.eps)                        # (N, 2, T, V)

        r = self.clip_radius
        rel = rel.clamp(-r, r)
        scaled = (rel + r) / (2.0 * r) * (self.num_buckets - 1)
        buckets = scaled.round().long().clamp(0, self.num_buckets - 1)
        bx = buckets[:, 0]                                               # (N, T, V)
        by = buckets[:, 1]

        emb = self.embed_x(bx) + self.embed_y(by)                        # (N, T, V, D)
        return emb.permute(0, 3, 1, 2).contiguous()                      # (N, D, T, V)
