"""BlockGCN backbone with two classification heads for joint co-training.

Implements the co-training alternative to the BPT recipe (supervised
pretrain on BdSLW401 -> head swap -> low-LR finetune on BdSLW60-SI):
one shared BlockGCN backbone optimized jointly on the small target
dataset (60-way head) and the large auxiliary dataset (401-way head).
Motivated by the Logos finding (EMNLP 2025) that multi-head joint
co-training beats sequential transfer for low-resource targets — the
BPT-vs-co-training ablation preempts that review question.

Trained by `main_cotrain.py`; not part of the single-dataset `main.py`
pipeline (forward returns a tuple of logits, one per head).
"""

from __future__ import annotations

import torch.nn as nn

from model.block_gcn import Model as BlockGCN


class Model(nn.Module):
    """Two-headed BlockGCN. forward: (N, C, T, V, M) -> (logits_target, logits_aux)."""

    def __init__(
        self,
        num_class=60,
        num_class_aux=401,
        num_point=27,
        num_person=1,
        graph=None,
        graph_args=None,
        in_channels=3,
        num_blocks=(2, 2, 2, 2),
        stage_channels=(64, 128, 256, 256),
        max_distance=8,
    ):
        super().__init__()
        self.backbone = BlockGCN(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args,
            in_channels=in_channels,
            num_blocks=num_blocks,
            stage_channels=stage_channels,
            max_distance=max_distance,
            return_features=True,
        )
        # Drop the backbone's own (unused) classifier so the optimizer sees
        # no dead parameters and the exported backbone state dict is clean.
        self.backbone.fc = nn.Identity()

        feat_dim = self.backbone.feat_dim
        self.fc_target = nn.Linear(feat_dim, num_class)
        self.fc_aux = nn.Linear(feat_dim, num_class_aux)
        for fc in (self.fc_target, self.fc_aux):
            nn.init.trunc_normal_(fc.weight, std=0.02)
            nn.init.zeros_(fc.bias)

    def forward(self, x, keep_prob=1.0):
        del keep_prob  # harness-signature compatibility; BlockGCN ignores it
        feat = self.backbone(x)  # (N, feat_dim) via return_features=True
        return self.fc_target(feat), self.fc_aux(feat)
