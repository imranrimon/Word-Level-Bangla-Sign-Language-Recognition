"""Smoke tests for Option B: DINOv2 feature pipeline + flat temporal model.

Verifies:
  * The flat temporal transformer forwards correctly on *both* shapes we
    plan to compare: pose (C=3, V=27) and DINOv2 (C=384, V=3).
  * Gradient flows through the transformer on both shape regimes.
  * The attention's key-padding mask correctly ignores all-zero frames
    without producing NaNs.
  * `_crop_square_region` in the DINOv2 extractor crops the expected patch
    on a synthetic image.

DINOv2 model download and MediaPipe holistic instantiation are *not* tested
here; those are integration concerns triggered by the real preprocessing run.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


def _set_seed(seed=0):
    torch.manual_seed(seed)


def test_flat_temporal_forward_on_pose_shape():
    from model.flat_temporal import Model

    _set_seed()
    m = Model(num_class=5, num_point=27, in_channels=3, d_model=64,
              num_heads=4, num_layers=2, dim_feedforward=128,
              max_frames=32).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_flat_temporal_forward_on_dinov2_shape():
    from model.flat_temporal import Model

    _set_seed()
    m = Model(num_class=5, num_point=3, in_channels=384, d_model=64,
              num_heads=4, num_layers=2, dim_feedforward=128,
              max_frames=32).eval()
    x = torch.randn(2, 384, 16, 3, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_flat_temporal_backward_flows_gradients():
    from model.flat_temporal import Model

    _set_seed()
    m = Model(num_class=5, num_point=27, in_channels=3, d_model=64,
              num_heads=4, num_layers=2, dim_feedforward=128, max_frames=32)
    x = torch.randn(2, 3, 16, 27, 1)
    loss = m(x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_flat_temporal_padding_mask_does_not_produce_nan():
    from model.flat_temporal import Model

    _set_seed()
    m = Model(num_class=5, num_point=27, in_channels=3, d_model=64,
              num_heads=4, num_layers=2, dim_feedforward=128,
              max_frames=32).eval()
    # Left 6 frames real, last 10 frames zero-padded -> should be ignored.
    x = torch.zeros(2, 3, 16, 27, 1)
    x[:, :, :6, :, :] = torch.randn(2, 3, 6, 27, 1)
    with torch.no_grad():
        y = m(x)
    assert torch.isfinite(y).all()


def test_flat_temporal_accepts_graph_kwargs_and_ignores_them():
    """YAML compatibility: graph/graph_args present in config must not crash."""
    from model.flat_temporal import Model

    _set_seed()
    m = Model(num_class=5, num_point=27, in_channels=3,
              graph="graph.sign_27.Graph", graph_args={"labeling_mode": "spatial"},
              d_model=32, num_heads=4, num_layers=1,
              dim_feedforward=64, max_frames=16).eval()
    x = torch.randn(1, 3, 8, 27, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 5)


# ---------------------------------------------------------------------------
# DINOv2 extractor helpers (pure-numpy, test without running MediaPipe/DINOv2)
# ---------------------------------------------------------------------------

def test_crop_square_region_returns_correct_shape_and_nonempty():
    from preprocessing.extract_dinov2_features import _crop_square_region

    # 480x640 BGR frame, uint8
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a bright square in the region [0.3..0.5]x[0.4..0.7] normalized
    y1, y2 = int(0.3 * 480), int(0.5 * 480)
    x1, x2 = int(0.4 * 640), int(0.7 * 640)
    frame[y1:y2, x1:x2] = 200
    xs = [0.4, 0.5, 0.6, 0.7]
    ys = [0.3, 0.35, 0.45, 0.5]
    crop = _crop_square_region(frame, xs, ys, pad_ratio=0.2, target=224)
    assert crop is not None
    assert crop.shape == (224, 224, 3)
    # The resized crop should still contain the bright patch (mean > 50).
    assert crop.mean() > 50.0


def test_crop_square_region_returns_none_on_empty_landmarks():
    from preprocessing.extract_dinov2_features import _crop_square_region

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = _crop_square_region(frame, [], [], pad_ratio=0.2, target=224)
    assert crop is None
