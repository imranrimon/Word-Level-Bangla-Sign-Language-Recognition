"""k-means pretraining targets, v2 (P1 fix program 2026-10).

Thin wrapper around preprocessing/compute_pretrain_targets.py (unchanged, still the reference
implementation) that adds ONE feature mode:

  * `pose_motion_masked` -- like `pose_motion` (pose concat first difference, 162-dim) but the
    motion of joint v at frame t is set to 0 unless v is detected ((x,y) != (0,0)) in BOTH t-1
    and t. MediaPipe stores a missing hand as zeros, so in `pose_motion` every hand
    detect/lose event writes a ~0.3-1.0 jump into 20 motion dims (normal motion ~0.01), and
    those transition frames can claim their own clusters. The pose half is unchanged (it still
    carries the presence/absence of each part).

All other modes and every CLI flag behave exactly as in compute_pretrain_targets.py.

    python preprocessing/compute_pretrain_targets_v2.py --manifest data/ssl_pool_manifest_bdsl_asl_v2.json \
        --output data/pretrain_kmeans_targets_bdsl_asl_v2.npz --num-clusters 64 --fit-sample 0.1 \
        --seed 0 --per-source-cap 200000 --feature-mode pose_motion_masked
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import preprocessing.compute_pretrain_targets as base  # noqa: E402

_orig_dim = base._feature_dim
_orig_feat = base._clip_to_per_frame_features


def _feature_dim(feature_mode, num_channels=base.DEFAULT_NUM_CHANNELS, num_joints=base.DEFAULT_NUM_JOINTS):
    if feature_mode == "pose_motion_masked":
        return 2 * num_channels * num_joints
    return _orig_dim(feature_mode, num_channels, num_joints)


def masked_pose_motion(data):
    """(C, T, V, M) -> (T, 2*C*V): pose | motion, motion zeroed unless detected at t-1 and t."""
    C, T, V, M = data.shape
    x = data[:, :, :, 0].astype(np.float32)                    # (C, T, V)
    det = np.abs(x[:2]).sum(0) > 0                              # (T, V)
    mot = np.zeros_like(x)
    if T > 1:
        ok = det[1:] & det[:-1]                                 # (T-1, V)
        mot[:, 1:] = (x[:, 1:] - x[:, :-1]) * ok[None]
    pose = x.transpose(1, 0, 2).reshape(T, C * V)
    mot = mot.transpose(1, 0, 2).reshape(T, C * V)
    return np.concatenate([pose, mot], axis=1)


def _clip_to_per_frame_features(data, feature_mode):
    if feature_mode == "pose_motion_masked":
        return masked_pose_motion(data)
    return _orig_feat(data, feature_mode)


base._feature_dim = _feature_dim
base._clip_to_per_frame_features = _clip_to_per_frame_features

if __name__ == "__main__":
    base.main()
