"""tools/check_pose_geometry.py on synthetic 27-keypoint skeletons (no data needed).

A correctly laid-out signer passes all six assertions; the WLASL-v1-style scramble (right-hand
landmarks in the body slots) and a left/right hand swap are caught.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_pose_geometry import BODY, LH, RH, check_clips  # noqa: E402


def _signer(rng, T=60):
    """(3, T, 27, 1) clip: upright body, hands at the wrists, hands moving more than the trunk."""
    t = np.arange(T)[:, None]
    xy = np.zeros((2, T, 27), np.float32)
    body = {0: (0.50, 0.30), 1: (0.40, 0.45), 2: (0.60, 0.45), 3: (0.37, 0.58),
            4: (0.63, 0.58), 5: (0.36, 0.70), 6: (0.64, 0.70)}
    for v, (x, y) in body.items():
        xy[0, :, v] = x + 0.002 * rng.standard_normal(T)
        xy[1, :, v] = y + 0.002 * rng.standard_normal(T)
    wave = 0.04 * np.sin(2 * np.pi * t / 15.0)[:, 0]
    for wrist_body, hand in ((5, RH), (6, LH)):
        for k, v in enumerate(hand):
            dx, dy = 0.012 * (k % 5), -0.02 * (k // 5 + (k > 0))
            xy[0, :, v] = xy[0, :, wrist_body] + dx + wave
            xy[1, :, v] = xy[1, :, wrist_body] + dy + 0.5 * wave
        xy[0, :, wrist_body] += wave          # body wrist follows the hand
        xy[1, :, wrist_body] += 0.5 * wave
    clip = np.zeros((3, T, 27, 1), np.float32)
    clip[:2, :, :, 0] = xy
    clip[2] = 1.0
    return clip


def test_correct_layout_passes():
    rng = np.random.default_rng(0)
    r = check_clips([_signer(rng) for _ in range(20)])
    assert r["verdict"] == "PASS", r["fails"]
    assert r["nose_up"] == 1.0 and r["spd_ratio"] > 1.5


def test_scrambled_body_slots_fail():
    """Right-hand landmarks stored in the body slots (the WLASL v1 converter bug)."""
    rng = np.random.default_rng(1)
    clips = []
    for _ in range(20):
        c = _signer(rng)
        c[:, :, BODY] = c[:, :, RH[:7]]
        clips.append(c)
    r = check_clips(clips)
    assert r["verdict"] == "FAIL"


def test_swapped_hands_fail_side_check():
    rng = np.random.default_rng(2)
    clips = []
    for _ in range(20):
        c = _signer(rng)
        c[:, :, RH + LH] = c[:, :, LH + RH]
        clips.append(c)
    r = check_clips(clips)
    assert r["verdict"] == "FAIL"
    assert any(f.startswith("A3") or f.startswith("A2") for f in r["fails"])


def test_missing_joints_are_ignored_not_counted_as_motion():
    """Zeroed (undetected) hand frames must not create hand speed from the 0 <-> value jumps."""
    rng = np.random.default_rng(3)
    clips = []
    for _ in range(20):
        c = _signer(rng)
        c[:2, ::4, :, :][:, :, RH + LH] = 0.0      # drop the hands in every 4th frame
        clips.append(c)
    r = check_clips(clips)
    assert r["verdict"] == "PASS", r["fails"]
    assert 0.2 < r["miss_R"] < 0.3
