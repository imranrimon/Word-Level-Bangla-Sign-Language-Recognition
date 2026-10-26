"""tools/compute_lang_distance_v2.py descriptors and the unbiased MMD on synthetic clips."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import compute_lang_distance_v2 as D  # noqa: E402


def _clip(rng, T, fps, amp=0.05, freq=1.5, pad=0):
    """(3, T+pad, 27, 1): shoulders fixed 0.2 apart, other joints oscillate at `freq` Hz."""
    t = np.arange(T) / fps
    xy = np.zeros((2, T + pad, 27), np.float32)
    base = rng.uniform(0.3, 0.7, size=(2, 27))
    base[:, 1], base[:, 2] = (0.4, 0.5), (0.6, 0.5)
    xy[:, :T] = base[:, None, :]
    osc = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28, size=(27, 1))).T
    osc[:, [1, 2]] = 0.0
    xy[0, :T] += osc
    c = np.zeros((3, T + pad, 27, 1), np.float32)
    c[:2, :, :, 0] = xy
    return c


def test_v2_descriptor_is_frame_rate_invariant():
    rng = np.random.default_rng(0)
    a = D.descriptor(_clip(rng, 50, 25.0), "v2", 25.0)
    b = D.descriptor(_clip(np.random.default_rng(0), 100, 50.0), "v2", 50.0)
    assert a.shape == (54,) and b.shape == (54,)
    assert np.allclose(a[:27], b[:27], rtol=0.15, atol=1e-3)       # mean speeds per second agree


def test_v2_descriptor_ignores_padding_and_missing_joints():
    rng = np.random.default_rng(1)
    c = _clip(rng, 50, 25.0, pad=30)
    full = D.descriptor(c, "v2", 25.0)
    c2 = c.copy()
    c2[:2, 10:12, 9] = 0.0                                          # joint 9 lost for two frames
    d = D.descriptor(c2, "v2", 25.0)
    # the masked descriptor must not turn the loss into a speed spike
    assert d[9] < 2 * full[9]
    # the masked-frame variant also stays bounded, while the raw frame-difference speed explodes
    raw = np.linalg.norm(np.diff(c2[:2, :50, 9, 0], axis=1), axis=0).max()
    assert raw > 0.3 and D.descriptor(c2, "masked_frame", 25.0)[9] < 0.05


def test_scale_normalisation_by_shoulder_width():
    rng = np.random.default_rng(2)
    c = _clip(rng, 50, 25.0)
    a = D.descriptor(c, "v2", 25.0)
    c2 = c.copy()
    c2[:2] *= 2.0                                                   # same motion, camera twice as close
    assert np.allclose(a, D.descriptor(c2, "v2", 25.0), rtol=1e-4, atol=1e-6)


def test_unbiased_mmd():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 5))
    Y = rng.standard_normal((200, 5))
    Z = rng.standard_normal((200, 5)) + 1.0
    s2 = D.median_sq(np.vstack([X, Z]))
    same, diff = D.mmd2_u(X, Y, s2), D.mmd2_u(X, Z, s2)
    assert abs(same) < 0.02 and diff > 0.1
