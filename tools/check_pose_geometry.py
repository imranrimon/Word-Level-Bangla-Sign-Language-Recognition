#!/usr/bin/env python
"""Geometry sanity checker for 27-keypoint pose data (P1 fix program, 2026-10).

Catches scrambled-skeleton bugs like the WLASL V3 index mix-up (body slots holding hand
landmarks, hand slots holding face-mesh points), which no training/eval code notices.

27-kp convention (graph/sign_27.py, preprocessing/preprocess_bdsl.py):
  0 nose, 1 R-shoulder, 2 L-shoulder, 3 R-elbow, 4 L-elbow, 5 R-wrist, 6 L-wrist  (body)
  7-16  right hand: wrist, thumb-tip, index-MCP, index-tip, mid-MCP, mid-tip, ring-MCP, ring-tip,
        pinky-MCP, pinky-tip
  17-26 left hand: same order
A joint is "detected" in a frame iff (x, y) != (0, 0) (MediaPipe stores a missing part as zeros).

Per corpus (a random sample of clips) it reports, all in units of the median shoulder width s
where scale-free:
  shoulder_w      median |p1 - p2| in raw coordinates (normalised MediaPipe coords: ~0.1-0.5)
  Rwrist_d/Rx_d   median |hand-R wrist (7) - body R-wrist (5)| / s, and to the WRONG body wrist (6)
  Lwrist_d/Lx_d   same for the left hand (17 vs 6, and vs 5)
  hand_span       median |wrist - mid-MCP| / s (a real hand: ~0.2-0.5 of shoulder width)
  spd_hand/spd_sh median per-frame speed of hand joints / of the shoulders (consecutive detected
                  frames only), ratio = hands faster than the trunk
  nose_up         fraction of body frames with nose above the shoulder midline (image y down)
  miss_body/R/L   fraction of frames where the body / right hand / left hand is not detected
  hand_side       fraction of frames where the detected R hand is nearer body-R than body-L wrist

Assertions (verdict FAIL if any is violated; thresholds chosen from the in-house MediaPipe
corpora with wide margins):
  A1  0.02 <= shoulder_w <= 1.0 when coordinates are normalised (max |coord| <= 2); skipped for
      pixel coordinates (reported only)
  A2  Rwrist_d < 0.35 and Lwrist_d < 0.35            (hand wrist sits on the pose wrist)
  A3  Rwrist_d < Rx_d and Lwrist_d < Lx_d            (on the correct side)
  A4  0.08 <= hand_span <= 0.9                       (hand-sized hand)
  A5  spd_hand / spd_sh >= 1.5                       (hands move more than the shoulders)
  A6  nose_up >= 0.8                                 (upright: nose above shoulders)

Usage:
  python tools/check_pose_geometry.py --npy data/bdsl_si/train_data.npy --name BdSL
  python tools/check_pose_geometry.py --npz-dir data/wlasl_pose_cache_v2 --name WLASL-v2
  python tools/check_pose_geometry.py --all [--n 300] [--json out.json] [--markdown out.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BODY = list(range(7))
RH = list(range(7, 17))
LH = list(range(17, 27))

# Every corpus used by P1 (targets = SI train bundles; pool sources = per-clip caches).
CORPORA = {
    "BdSL (BdSLW60 si train)": ("npy", "data/bdsl_si/train_data.npy"),
    "BdSLW60 pretrain split": ("npy", "data/bdsl_si/pretrain_data.npy"),
    "BdSLW401 front cache (pool)": ("npz", "data/bdslw401_pose_cache_front"),
    "BdSLW401 si train": ("npy", "data/bdslw401_si/train_data.npy"),
    "BdSLW102a cache (pool)": ("npz", "data/bdslw102_a_pose_cache"),
    "WLASL cache v1 (pool, OLD)": ("npz", "data/wlasl_pose_cache"),
    "WLASL cache v2 (pool, FIXED)": ("npz", "data/wlasl_pose_cache_v2"),
    "LSA64 si train": ("npy", "data/lsa64_si/train_data.npy"),
    "AUTSL si train": ("npy", "data/autsl_si/train_data.npy"),
    "KArSL si train": ("npy", "data/karsl_si/train_data.npy"),
    "GSL si train": ("npy", "data/gsl_si/train_data.npy"),
    "LSFB si train": ("npy", "data/lsfb_si/train_data.npy"),
    "LIBRAS si train": ("npy", "data/libras_si/train_data.npy"),
    "BISINDO si train": ("npy", "data/bisindo_si/train_data.npy"),
    "ASL-Citizen-60 si train": ("npy", "data/aslcitizen60_si/train_data.npy"),
}


def _xy(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    return a[:2, :, :, 0]  # (2, T, V)


def clip_stats(arr):
    """Per-clip raw quantities (lists of per-frame values) for pooling across a corpus."""
    xy = _xy(arr)
    det = np.abs(xy).sum(0) > 0                      # (T, V)
    keep = det.any(1)                                # drop all-zero padding frames
    xy, det = xy[:, keep], det[keep]
    T = det.shape[0]
    if T < 2:
        return None
    out = {"T": T}
    body = det[:, 1] & det[:, 2]
    rh, lh = det[:, 7], det[:, 17]
    out["miss_body"] = 1.0 - body.mean()
    out["miss_body7_all"] = float((~det[:, :7]).all(1).mean())
    out["miss_R"] = 1.0 - rh.mean()
    out["miss_L"] = 1.0 - lh.mean()
    out["maxabs"] = float(np.abs(xy).max())
    P = lambda v, m: xy[:, m, v]                     # (2, n)
    out["sw"] = np.linalg.norm(P(1, body) - P(2, body), axis=0)
    m = body & det[:, 0]
    mid_y = 0.5 * (xy[1, m, 1] + xy[1, m, 2])
    out["nose_up"] = (xy[1, m, 0] < mid_y)
    for tag, hw, own, other in (("R", 7, 5, 6), ("L", 17, 6, 5)):
        m = det[:, hw] & det[:, own] & det[:, other]
        out[f"{tag}_own"] = np.linalg.norm(P(hw, m) - P(own, m), axis=0)
        out[f"{tag}_oth"] = np.linalg.norm(P(hw, m) - P(other, m), axis=0)
        mm = det[:, hw] & det[:, hw + 4]
        out[f"{tag}_span"] = np.linalg.norm(P(hw, mm) - P(hw + 4, mm), axis=0)
    # speeds between consecutive frames where the joint is detected in both
    sp = np.linalg.norm(np.diff(xy, axis=1), axis=0)  # (T-1, V)
    ok = det[1:] & det[:-1]
    out["spd_hand"] = sp[:, RH + LH][ok[:, RH + LH]]
    out["spd_sh"] = sp[:, [1, 2]][ok[:, [1, 2]]]
    # share of raw speed mass that comes from detect/lose transitions (0 <-> value jumps)
    trans = det[1:] ^ det[:-1]
    out["dropout_speed_share"] = float(sp[trans].sum() / max(sp.sum(), 1e-12))
    return out


def summarize(stats):
    stats = [s for s in stats if s is not None]
    cat = lambda k: np.concatenate([np.atleast_1d(s[k]).astype(np.float64) for s in stats])
    sw = cat("sw")
    s = float(np.median(sw)) if sw.size else float("nan")
    r = {"n_clips": len(stats), "median_T": float(np.median([x["T"] for x in stats])),
         "maxabs": float(np.max([x["maxabs"] for x in stats])), "shoulder_w": s}
    med = lambda k: float(np.median(cat(k))) if cat(k).size else float("nan")
    for tag in ("R", "L"):
        r[f"{tag}wrist_d"] = med(f"{tag}_own") / s
        r[f"{tag}x_d"] = med(f"{tag}_oth") / s
        r[f"{tag}_span"] = med(f"{tag}_span") / s
        own, oth = cat(f"{tag}_own"), cat(f"{tag}_oth")
        r[f"{tag}_side_ok"] = float((own < oth).mean()) if own.size else float("nan")
    r["hand_span"] = float(np.nanmean([r["R_span"], r["L_span"]]))
    r["spd_hand"] = med("spd_hand") / s
    r["spd_sh"] = med("spd_sh") / s
    r["spd_ratio"] = r["spd_hand"] / max(r["spd_sh"], 1e-12)
    nu = cat("nose_up")
    r["nose_up"] = float(nu.mean()) if nu.size else float("nan")
    for k in ("miss_body", "miss_body7_all", "miss_R", "miss_L", "dropout_speed_share"):
        r[k] = float(np.mean([x[k] for x in stats]))
    return r


def verdict(r):
    fails = []
    normalised = r["maxabs"] <= 2.0
    if normalised and not (0.02 <= r["shoulder_w"] <= 1.0):
        fails.append(f"A1 shoulder width {r['shoulder_w']:.3f} outside [0.02,1.0]")
    for t in ("R", "L"):
        if not (r[f"{t}wrist_d"] < 0.35):
            fails.append(f"A2 {t}-hand wrist {r[f'{t}wrist_d']:.2f}s from pose wrist (>=0.35s)")
        if not (r[f"{t}wrist_d"] < r[f"{t}x_d"]):
            fails.append(f"A3 {t}-hand nearer the WRONG body wrist ({r[f'{t}wrist_d']:.2f} vs {r[f'{t}x_d']:.2f})")
    if not (0.08 <= r["hand_span"] <= 0.9):
        fails.append(f"A4 hand span {r['hand_span']:.2f}s outside [0.08,0.9]")
    if not (r["spd_ratio"] >= 1.5):
        fails.append(f"A5 hand/shoulder speed ratio {r['spd_ratio']:.2f} < 1.5")
    if not (r["nose_up"] >= 0.8):
        fails.append(f"A6 nose above shoulders in only {r['nose_up']:.2f} of frames")
    return ("PASS" if not fails else "FAIL"), fails


def sample_npy(path, n, rng):
    a = np.load(path, mmap_mode="r")
    idx = np.sort(rng.choice(len(a), size=min(n, len(a)), replace=False))
    return [np.array(a[i]) for i in idx]


def sample_npz(root, n, rng):
    fs = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    if not fs:
        raise FileNotFoundError(f"no .npz under {root}")
    idx = rng.choice(len(fs), size=min(n, len(fs)), replace=False)
    out = []
    for i in idx:
        with np.load(fs[i]) as h:   # I/O errors propagate (never silently skip)
            out.append(np.array(h["data"]))
    return out


def check_clips(clips):
    r = summarize([clip_stats(c) for c in clips])
    v, fails = verdict(r)
    r["verdict"], r["fails"] = v, fails
    return r


def fmt_row(name, r):
    return (f"| {name} | {r['n_clips']} | {r['median_T']:.0f} | {r['shoulder_w']:.3f} | "
            f"{r['Rwrist_d']:.2f}/{r['Rx_d']:.2f} | {r['Lwrist_d']:.2f}/{r['Lx_d']:.2f} | "
            f"{r['hand_span']:.2f} | {r['spd_ratio']:.1f} | {r['nose_up']:.2f} | "
            f"{r['miss_body']:.2f}/{r['miss_R']:.2f}/{r['miss_L']:.2f} | "
            f"{r['dropout_speed_share']:.2f} | **{r['verdict']}** {'; '.join(r['fails'])} |")


HEADER = ("| corpus | clips | med T | shoulder w | R wrist own/wrong (s) | L wrist own/wrong (s) | "
          "hand span (s) | hand/shoulder speed | nose up | miss body/R/L | dropout speed share | verdict |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy"); ap.add_argument("--npz-dir"); ap.add_argument("--name", default="corpus")
    ap.add_argument("--all", action="store_true", help="check every P1 corpus in CORPORA")
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json"); ap.add_argument("--markdown")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any corpus FAILs")
    a = ap.parse_args()
    todo = {}
    if a.all:
        todo.update(CORPORA)
    if a.npy:
        todo[a.name] = ("npy", a.npy)
    if a.npz_dir:
        todo[a.name] = ("npz", a.npz_dir)
    res = {}
    print(HEADER)
    for name, (kind, path) in todo.items():
        if not os.path.exists(path):
            print(f"| {name} | MISSING {path} |"); continue
        rng = np.random.default_rng(a.seed)
        clips = sample_npy(path, a.n, rng) if kind == "npy" else sample_npz(path, a.n, rng)
        r = check_clips(clips); r["path"] = path
        res[name] = r
        print(fmt_row(name, r), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(HEADER + "\n" + "\n".join(fmt_row(k, v) for k, v in res.items()) + "\n")
    if a.strict and any(r["verdict"] == "FAIL" for r in res.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
