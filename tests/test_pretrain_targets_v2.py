"""preprocessing/compute_pretrain_targets_v2.py: the dropout-masked pose_motion feature mode."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import compute_pretrain_targets_v2 as V2  # noqa: E402


def _clip(T=12, V=27, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(0.01 * rng.standard_normal((3, T, V, 1)), axis=1).astype(np.float32) + 0.5
    return x


def test_shape_and_feature_dim():
    x = _clip()
    f = V2.masked_pose_motion(x)
    assert f.shape == (12, 2 * 3 * 27)
    assert V2._feature_dim("pose_motion_masked") == 2 * 3 * 27
    assert np.array_equal(V2._clip_to_per_frame_features(x, "pose_motion_masked"), f)


def test_motion_is_zero_across_detect_lose_transitions():
    x = _clip()
    x[:, 5, 9] = 0.0                       # joint 9 undetected in frame 5
    f = V2.masked_pose_motion(x)
    mot = f[:, 3 * 27:].reshape(12, 3, 27)
    assert np.all(mot[5, :, 9] == 0) and np.all(mot[6, :, 9] == 0)   # 0 -> value and value -> 0 jumps masked
    assert np.all(mot[0] == 0)                                       # no motion at t = 0
    assert np.any(mot[7, :2, 9] != 0)                                # real motion kept


def test_unmasked_mode_would_see_the_jump():
    """The plain pose_motion mode (v1) turns a detect/lose event into a large motion value."""
    x = _clip()
    x[:, 5, 9] = 0.0
    f1 = V2._clip_to_per_frame_features(x, "pose_motion")
    f2 = V2.masked_pose_motion(x)
    assert np.abs(f1[:, 3 * 27:]).max() > 10 * np.abs(f2[:, 3 * 27:]).max()
