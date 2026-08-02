"""Feeder that yields (pose, teacher_feature, label) per sample.

Pairs the pose NPY (the BlockGCN input) with a teacher-feature NPY produced
by Path 1's `path1_bangla_dinov2/extract_features.py` (or Option B's stock
DINOv2 extractor). Both must be aligned row-by-row — same number of samples,
same sample order. We sanity-check by comparing label files.

Teacher feature layout follows the convention from
`preprocessing/extract_dinov2_features.py`:
    teacher_data.npy : (N, D_teacher, T_max, V_regions=3, M=1)
where region index 0 = left hand, 1 = right hand, 2 = face.

For handshape KD we pool **only the hand regions** (indices 0 and 1) into a
single concatenated 2*D_teacher-dim vector per frame, then mean over time.
The face region is intentionally dropped — KD here is *handshape* distillation.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from feeders.feeder import Feeder as _PoseFeeder


class KDFeeder(Dataset):
    def __init__(
        self,
        pose_data_path,
        pose_label_path,
        teacher_data_path,
        teacher_label_path=None,
        # standard pose-feeder kwargs:
        random_choose=False, random_shift=False, random_move=False,
        window_size=120, normalization=False, debug=False, use_mmap=True,
        random_mirror=False, random_mirror_p=0.5, is_vector=False,
        lap_pe=False,
        # KD-specific:
        teacher_region_indices=(0, 1),     # 0=L hand, 1=R hand
        pool_teacher_over_time=True,
    ):
        # Reuse the existing pose feeder for crop/normalisation/etc.
        self.pose = _PoseFeeder(
            data_path=pose_data_path, label_path=pose_label_path,
            random_choose=random_choose, random_shift=random_shift,
            random_move=random_move, window_size=window_size,
            normalization=normalization, debug=debug, use_mmap=use_mmap,
            random_mirror=random_mirror, random_mirror_p=random_mirror_p,
            is_vector=is_vector, lap_pe=lap_pe,
        )
        # mmap teacher features for cheap row access
        self.teacher = np.load(teacher_data_path, mmap_mode="r")
        if self.teacher.ndim != 5:
            raise ValueError(f"teacher data must be 5D (N,D,T,V,M); got shape {self.teacher.shape}")
        if self.teacher.shape[0] != len(self.pose):
            raise ValueError(
                f"length mismatch: pose has {len(self.pose)} samples, "
                f"teacher has {self.teacher.shape[0]}"
            )
        # Optional: cross-check sample names if a teacher label pkl is provided.
        if teacher_label_path is not None and os.path.exists(teacher_label_path):
            with open(pose_label_path, "rb") as f:
                pose_names, _ = pickle.load(f)
            with open(teacher_label_path, "rb") as f:
                teacher_names, _ = pickle.load(f)
            if list(pose_names) != list(teacher_names):
                # Common-mode failure: same files but produced in a different
                # walk order. Fail loud rather than silently misalign.
                raise ValueError(
                    "pose and teacher sample names differ at the same index; "
                    "regenerate one of the bundles so they share row order"
                )

        self.region_indices = tuple(int(i) for i in teacher_region_indices)
        self.pool_teacher_over_time = bool(pool_teacher_over_time)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):
        pose_item = self.pose[idx]
        # Existing pose Feeder returns (data, label, index) per sample.
        if len(pose_item) == 3:
            pose, label, _index = pose_item
        elif len(pose_item) == 2:
            pose, label = pose_item
        else:
            raise RuntimeError(f"unexpected pose feeder output len={len(pose_item)}")

        # Teacher slice: (D, T, V, M)
        t = self.teacher[idx]                         # mmap view
        # Select region indices, drop person dim
        sel = np.stack([t[:, :, r, 0] for r in self.region_indices], axis=2)
        # sel shape: (D, T, len(region_indices))
        # Concatenate selected regions along the channel dim -> (D*R, T)
        teacher_concat = sel.transpose(2, 0, 1).reshape(-1, sel.shape[1])

        if self.pool_teacher_over_time:
            # Use only frames where teacher signal is non-zero (drop padding).
            valid = (np.abs(teacher_concat).sum(axis=0) > 0)
            denom = max(int(valid.sum()), 1)
            teacher_vec = teacher_concat[:, valid].sum(axis=1) / denom
            teacher_out = torch.from_numpy(teacher_vec.astype(np.float32))
        else:
            teacher_out = torch.from_numpy(teacher_concat.astype(np.float32))

        return pose, teacher_out, label
