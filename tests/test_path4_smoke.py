"""Smoke tests for Path 4 (RGB baseline): dataset parity + model head surgery."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_ROOT = os.path.join(_ROOT, "Word_level_Bangla_Sign_Language_Dataset", "BdSLW30")
_CLASSES_JSON = os.path.join(_ROOT, "data", "bdsl_si", "classes.json")

needs_data = pytest.mark.skipif(
    not (os.path.isdir(_DATASET_ROOT) and os.path.exists(_CLASSES_JSON)),
    reason="raw BdSLW60 videos / classes.json not on this machine",
)


def test_segment_indices_bounds_and_count():
    from path4_rgb_baseline.rgb_dataset import segment_indices

    for n_avail in (1, 5, 16, 300):
        for train in (False, True):
            idx = segment_indices(n_avail, 16, train)
            assert len(idx) == 16
            assert all(0 <= i < n_avail for i in idx)
    # eval sampling is deterministic
    assert segment_indices(300, 8, False) == segment_indices(300, 8, False)


def test_s3d_head_replacement_forward():
    from path4_rgb_baseline.train_rgb import build_model

    model, norm = build_model("s3d", num_classes=60, pretrained=False)
    assert norm == "kinetics"
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 16, 224, 224))
    assert out.shape == (1, 60)


def test_i3d_head_replacement_forward():
    from path4_rgb_baseline.train_rgb import build_model

    model, norm = build_model("i3d", num_classes=60, pretrained=False)
    assert norm == "pm1"
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 16, 224, 224))
        if out.dim() > 2:  # vendored I3D emits (N, C, T')
            out = out.mean(dim=2)
    assert out.shape == (1, 60)


@needs_data
def test_rgb_dataset_labels_match_pose_bundle():
    import pickle

    from path4_rgb_baseline.rgb_dataset import BdSLW60RGBDataset

    ds = BdSLW60RGBDataset(
        root=_DATASET_ROOT, split="val", classes_json=_CLASSES_JSON,
        num_frames=8, size=224, train=False, max_clips=5,
    )
    clip, label, index = ds[0]
    assert clip.shape == (3, 8, 224, 224)
    assert clip.dtype == torch.float32
    assert 0 <= label < 60

    # Same (filename -> label) pairs as the pose bundle for the val split.
    with open(os.path.join(_ROOT, "data", "bdsl_si", "val_label.pkl"), "rb") as f:
        pose_names, pose_labels = pickle.load(f)
    pose_map = dict(zip(pose_names, pose_labels))
    for name, label in zip(ds.sample_name, ds.label):
        key = os.path.splitext(name)[0] + ".mp4" if not name.endswith(".mp4") else name
        assert key in pose_map, f"{key} missing from pose bundle"
        assert pose_map[key] == label
