"""Convert WLASL MediaPipe landmarks (Kaggle `abd0kamel/mutemotion-output`, landmarks_V3.npz)
into a per-clip 27-keypoint pose cache -- CORRECTED layout (v2, P1 fix program 2026-10).

WHY v2. The v1 converter (convert_wlasl_landmarks_to_pose_cache.py) assumed the V3 553-point
order was [pose 33, face 468, left hand 21, right hand 21, extras 10]. The dataset card states,
and the data confirm, that the order is

    [ Right Hand (21) | Left Hand (21) | Pose (33) | Face (478) ]   = 553
       0..20            21..41           42..74      75..552

so v1's 7 "body" slots were right-hand landmarks and its 20 "hand" slots were face-mesh points.
Empirical verification on 150 random clips (docs/p1_fix_2026-10/LOG.md):
  * block 0..20 wrist -> pose[16] (subject R wrist) median 0.020, -> pose[15] 0.321
  * block 21..41 wrist -> pose[15] 0.022, -> pose[16] 0.225
  * pose[11]-pose[12] shoulder width 0.294; nose above shoulders in 100% of frames
  * face point 75+1 (MediaPipe face 1, nose tip) 0.012 from pose nose
  * each block is all-or-none detected per frame (missing part stored as zeros)

Output format is identical to v1 (and to data/bdsl_cache): `data` = (3, T, 27, 1) float32,
channels x, y, binary detection flag; 27-kp order of graph/sign_27.py:
  0-6  nose, R-sh, L-sh, R-elb, L-elb, R-wrist, L-wrist  = pose [0, 12, 11, 14, 13, 16, 15]
  7-16 right hand [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];  17-26 left hand (same local indices)
(the same mapping preprocess_bdsl.py applies to MediaPipe Holistic's pose/right/left results).

GEOMETRY ASSERTIONS (fail loudly, nothing is written if the pre-check fails):
  * pre-check on the first --check-n converted clips, post-check on a random sample of the
    written files, both with tools/check_pose_geometry.py's verdict (shoulder width band, hand
    wrist on the correct-side pose wrist, hand span, hands faster than shoulders, upright);
  * plus a per-clip check: among clips with >=5 frames where a hand and its pose wrist are both
    detected, the median hand-wrist -> pose-wrist distance must be < 0.5 shoulder widths in
    >= 95% of clips.

Optionally also writes the bone-modality mirror (bone[v2] = joint[v2] - joint[v1] over
graph.sign_27 inward edges, exactly preprocessing/make_bone_pool.py's transform).

    python preprocessing/convert_wlasl_landmarks_to_pose_cache_v2.py \
        --landmarks data/raw_wlasl_v3/landmarks_V3.npz \
        --output-dir data/wlasl_pose_cache_v2 --bone-dir data/wlasl_pose_cache_bone_v2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.check_pose_geometry import check_clips  # noqa: E402

RH0, LH0, POSE0 = 0, 21, 42
_BODY = [POSE0 + i for i in (0, 12, 11, 14, 13, 16, 15)]
_HAND_LOCAL = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
V3_TO_OUR27_V2 = _BODY + [RH0 + i for i in _HAND_LOCAL] + [LH0 + i for i in _HAND_LOCAL]
assert len(V3_TO_OUR27_V2) == 27 and len(set(V3_TO_OUR27_V2)) == 27


def convert_clip(v3):
    """(T, 553, 3) -> (3, T, 27, 1) float32; channel 2 = 1 iff the point is non-zero."""
    if v3.ndim != 3 or v3.shape[1] != 553 or v3.shape[2] < 2:
        raise ValueError(f"unexpected V3 shape {v3.shape}")
    sel = v3[:, V3_TO_OUR27_V2, :]
    out = np.zeros((3, sel.shape[0], 27, 1), dtype=np.float32)
    out[0, :, :, 0] = sel[:, :, 0]
    out[1, :, :, 0] = sel[:, :, 1]
    out[2, :, :, 0] = (np.abs(sel[:, :, :2]).sum(-1) > 0).astype(np.float32)
    return out


def per_clip_wrist_ok(clip, min_frames=5, tol=0.5):
    """None if not checkable, else True iff each checkable hand's wrist sits on its pose wrist."""
    xy = clip[:2, :, :, 0]
    det = np.abs(xy).sum(0) > 0
    body = det[:, 1] & det[:, 2]
    if body.sum() < min_frames:
        return None
    s = np.median(np.linalg.norm(xy[:, body, 1] - xy[:, body, 2], axis=0))
    res = []
    for hw, pw in ((7, 5), (17, 6)):
        m = det[:, hw] & det[:, pw]
        if m.sum() >= min_frames:
            res.append(np.median(np.linalg.norm(xy[:, m, hw] - xy[:, m, pw], axis=0)) < tol * s)
    return all(res) if res else None


def _assert_geometry(clips, tag):
    r = check_clips(clips)
    print(f"[geometry {tag}] n={r['n_clips']} shoulder_w={r['shoulder_w']:.3f} "
          f"Rwrist={r['Rwrist_d']:.2f}s (wrong side {r['Rx_d']:.2f}s) Lwrist={r['Lwrist_d']:.2f}s "
          f"(wrong side {r['Lx_d']:.2f}s) span={r['hand_span']:.2f}s speed ratio={r['spd_ratio']:.1f} "
          f"nose_up={r['nose_up']:.2f} -> {r['verdict']}", flush=True)
    if r["verdict"] != "PASS":
        raise AssertionError(f"geometry check FAILED ({tag}): {r['fails']}")
    oks = [per_clip_wrist_ok(c) for c in clips]
    oks = [o for o in oks if o is not None]
    frac = float(np.mean(oks)) if oks else float("nan")
    print(f"[geometry {tag}] per-clip wrist check: {frac:.3f} of {len(oks)} checkable clips OK", flush=True)
    if not oks or frac < 0.95:
        raise AssertionError(f"per-clip wrist check FAILED ({tag}): {frac:.3f} < 0.95")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landmarks", default="data/raw_wlasl_v3/landmarks_V3.npz")
    ap.add_argument("--output-dir", default="data/wlasl_pose_cache_v2")
    ap.add_argument("--bone-dir", default=None, help="also write the bone mirror here")
    ap.add_argument("--check-n", type=int, default=400)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if os.path.abspath(a.output_dir).rstrip("/").endswith("wlasl_pose_cache"):
        raise SystemExit("refusing to overwrite the v1 cache in place")

    bundle = np.load(a.landmarks, allow_pickle=False)
    keys = list(bundle.keys())
    print(f"{a.landmarks}: {len(keys)} clips", flush=True)

    # ---- pre-check: nothing is written unless the converted sample passes ----
    pre = [convert_clip(bundle[k]) for k in keys[: a.check_n]]
    _assert_geometry(pre, "pre-write")

    inward = None
    if a.bone_dir:
        from graph.sign_27 import Graph
        inward = Graph(labeling_mode="spatial").inward
        os.makedirs(a.bone_dir, exist_ok=True)
    os.makedirs(a.output_dir, exist_ok=True)
    n_w = n_s = 0
    for i, k in enumerate(keys):
        op = os.path.join(a.output_dir, f"{k}.npz")
        bp = os.path.join(a.bone_dir, f"{k}.npz") if a.bone_dir else None
        if a.skip_existing and os.path.exists(op) and (bp is None or os.path.exists(bp)):
            n_s += 1
            continue
        out = convert_clip(bundle[k])
        np.savez_compressed(op, data=out)
        if bp:
            bone = np.zeros_like(out)
            for v1, v2 in inward:
                bone[:, :, v2, :] = out[:, :, v2, :] - out[:, :, v1, :]
            np.savez_compressed(bp, data=bone)
        n_w += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(keys)}", flush=True)
    print(f"written {n_w}, skipped existing {n_s}", flush=True)

    # ---- post-check on a random sample of the files actually written ----
    rng = np.random.default_rng(1)
    samp = rng.choice(len(keys), size=min(a.check_n, len(keys)), replace=False)
    post = []
    for j in samp:
        with np.load(os.path.join(a.output_dir, f"{keys[j]}.npz")) as h:
            post.append(np.array(h["data"]))
    _assert_geometry(post, "post-write")
    print("OK", flush=True)


if __name__ == "__main__":
    main()
