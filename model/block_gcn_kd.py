"""BlockGCN with an auxiliary KD head for handshape-teacher distillation.

Wraps `model.block_gcn.Model` (set with `return_features=True`) and adds a
single linear projection from the student's pre-classifier pooled feature
to the teacher's feature dimension. Forward returns *both* class logits and
the projected student feature so the training loop can compute classification
cross-entropy plus a KD loss against pooled teacher features.

Compared with vanilla BlockGCN this adds:
    student_proj : (N, feat_dim) -> (N, teacher_dim)

and otherwise leaves the backbone's input/output contract unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.block_gcn import Model as _BlockGCN


class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=27,
        num_person=1,
        graph=None,
        graph_args=None,
        in_channels=3,
        num_blocks=(2, 2, 2, 2),
        stage_channels=(64, 128, 256, 256),
        max_distance=8,
        teacher_dim=768,
        proj_hidden=None,
    ):
        super().__init__()
        # Build the backbone in feature-extractor mode; we add our own head.
        self.backbone = _BlockGCN(
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
        feat_dim = self.backbone.feat_dim
        self.fc = nn.Linear(feat_dim, num_class)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

        # KD projection head: maps student pooled feature -> teacher dim.
        if proj_hidden is None:
            self.kd_proj = nn.Linear(feat_dim, teacher_dim)
        else:
            self.kd_proj = nn.Sequential(
                nn.Linear(feat_dim, int(proj_hidden)),
                nn.GELU(),
                nn.Linear(int(proj_hidden), teacher_dim),
            )
        self.teacher_dim = int(teacher_dim)

    def forward(self, x, keep_prob=1.0):
        """x: (N, C, T, V, M)  ->  (logits, student_proj).

        Returning a tuple is intentional — main.py consumers won't accept
        this contract, so this model is invoked only by `path2_handshape_kd/
        train_kd.py`.
        """
        pooled = self.backbone(x, keep_prob=keep_prob)   # (N, feat_dim)
        logits = self.fc(pooled)
        student_proj = self.kd_proj(pooled)              # (N, teacher_dim)
        return logits, student_proj
