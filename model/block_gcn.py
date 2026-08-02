"""BlockGCN backbone for skeleton-based sign language recognition.

Faithful minimum implementation of the two core ideas from Zhou et al.,
"BlockGCN: Redefine Topology Awareness for Skeleton-Based Action
Recognition" (CVPR 2024):

  * Topological Encoding (TE): pairwise shortest-path distances over the
    skeleton graph are embedded and injected as a per-head additive bias
    on the affinity matrix. Shared across all blocks (block-level rather
    than per-layer topology learning).
  * Block graph convolution: static physical adjacency is combined with a
    CTR-GCN-style channel-wise refinement plus the shared TE, and used as
    the affinity applied to a 1x1-projected node feature map.

Input/output contract matches the other models in this repo:
    forward(x) : x is (N, C, T, V, M) -> logits of shape (N, num_class).

Setting return_features=True returns the pooled backbone feature (N, C_f)
instead of class logits, which is useful for building classifier heads on
top of a pretrained backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def _shortest_path_distances(A):
    """Unweighted shortest-path distances on the union of partition adjacencies."""
    A = np.asarray(A)
    if A.ndim == 3:
        adj = (A.sum(axis=0) > 0).astype(np.int64)
    else:
        adj = (A > 0).astype(np.int64)
    # ensure symmetric for undirected distances
    adj = np.maximum(adj, adj.T)
    V = adj.shape[-1]
    INF = 10 ** 6
    dist = np.where(adj > 0, 1, INF).astype(np.int64)
    np.fill_diagonal(dist, 0)
    # Floyd-Warshall (V is small, ~27)
    for k in range(V):
        dist = np.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist


class TopologicalEncoding(nn.Module):
    """Embed shortest-path distances on the skeleton as a per-head affinity bias."""

    def __init__(self, A, num_heads=1, max_distance=8):
        super().__init__()
        dist = _shortest_path_distances(A)
        dist = np.clip(dist, 0, max_distance)
        self.register_buffer("dist", torch.from_numpy(dist).long())
        self.embed = nn.Embedding(max_distance + 1, num_heads)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self):
        # dist: (V, V) -> embed -> (V, V, H) -> (H, V, V)
        return self.embed(self.dist).permute(2, 0, 1).contiguous()


class BlockGraphConv(nn.Module):
    """aff = softmax(A_phys + alpha * CTR(x) + TE); y = conv1x1(x) @ aff."""

    def __init__(self, in_channels, out_channels, A, max_distance=8, rel_reduction=8):
        super().__init__()
        A = np.asarray(A)
        A_first = A[0] if A.ndim == 3 else A
        self.register_buffer("A_phys", torch.from_numpy(A_first.astype(np.float32)))
        reduced = max(in_channels // rel_reduction, 2)
        self.theta = nn.Conv2d(in_channels, reduced, 1)
        self.phi = nn.Conv2d(in_channels, reduced, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.te = TopologicalEncoding(A, num_heads=1, max_distance=max_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (N, C, T, V)
        theta = self.theta(x).mean(dim=2)              # (N, C', V)
        phi = self.phi(x).mean(dim=2)                  # (N, C', V)
        ctr = torch.matmul(theta.transpose(1, 2), phi)  # (N, V, V)
        te = self.te().squeeze(0)                      # (V, V)
        aff = self.A_phys.unsqueeze(0) + self.alpha * ctr + te.unsqueeze(0)
        aff = F.softmax(aff, dim=-1)
        y = torch.einsum("nctw, nvw -> nctv", self.conv(x), aff)
        return F.relu(self.bn(y) + self.residual(x))


class MultiScaleTCN(nn.Module):
    """Two-branch temporal conv: dilations 1 and 2, concatenated."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("MultiScaleTCN requires out_channels divisible by 2")
        half = out_channels // 2
        pad1 = (kernel_size - 1) // 2
        pad2 = (kernel_size - 1)  # dilation=2
        self.branch1 = nn.Conv2d(
            in_channels, half, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(pad1, 0), dilation=(1, 1),
        )
        self.branch2 = nn.Conv2d(
            in_channels, half, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(pad2, 0), dilation=(2, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return F.relu(self.bn(y))


class BlockGCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, max_distance=8):
        super().__init__()
        self.gcn = BlockGraphConv(in_channels, out_channels, A, max_distance=max_distance)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x))


class Model(nn.Module):
    """BlockGCN backbone with (N, C, T, V, M) -> (N, num_class) contract."""

    def __init__(
        self,
        num_class=60,
        num_point=27,
        num_person=1,
        graph=None,
        graph_args=None,
        in_channels=3,
        num_blocks=(1, 2, 3, 3),
        stage_channels=(64, 128, 256, 256),
        max_distance=8,
        return_features=False,
        stride_between_stages=True,
    ):
        super().__init__()
        if graph is None:
            raise ValueError("BlockGCN requires a `graph` argument.")
        graph_args = graph_args or {}
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A

        if len(num_blocks) != len(stage_channels):
            raise ValueError("num_blocks and stage_channels must have same length")

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        layers = []
        prev_c = in_channels
        for stage_idx, (n_blocks, out_c) in enumerate(zip(num_blocks, stage_channels)):
            for b in range(n_blocks):
                # Inter-stage stride defaults to 2 for classification (T halves at
                # each stage transition). SSL pretraining sets this False so that
                # the backbone's output T matches the input T — required by the
                # SHuBERT-style per-time masked prediction loss, which masks at
                # input resolution.
                if stride_between_stages and b == 0 and stage_idx > 0:
                    stride = 2
                else:
                    stride = 1
                layers.append(
                    BlockGCNUnit(prev_c, out_c, A, stride=stride, max_distance=max_distance)
                )
                prev_c = out_c
        self.layers = nn.ModuleList(layers)
        self.feat_dim = prev_c
        self.return_features = return_features
        self.fc = nn.Linear(prev_c, num_class)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, keep_prob=1.0):
        # x: (N, C, T, V, M). keep_prob is accepted for harness compatibility
        # with the SLGTFormer-family forward signature; BlockGCN has no
        # DropBlock_Ske/DropBlockT_1d inside its blocks so it is ignored.
        del keep_prob
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for layer in self.layers:
            x = layer(x)

        c = x.size(1)
        x = x.view(N, M, c, -1).mean(dim=-1).mean(dim=1)  # (N, c)
        if self.return_features:
            return x
        return self.fc(x)
