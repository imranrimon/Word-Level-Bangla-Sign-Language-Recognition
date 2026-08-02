"""Audit fix #10: per-signer hand-keypoint / hand-crop detection rates.

Cross-signer variance in MediaPipe detection success is a hidden identity-
shortcut: if signer A's hands are detected 95 % of frames and signer B's
only 60 %, then ANY downstream feature (pose, DINOv2-crop, kd-teacher)
already encodes per-signer info before any learning. The model can shortcut
"signer identity" from this alone.

This tool walks a per-clip pose cache OR DINOv2 feature cache and reports:

  * total frames per signer,
  * hand-detection rate per signer (fraction of frames with non-zero hand
    landmarks, per left/right separately),
  * face-detection rate per signer,
  * per-signer outliers (>= 2 SD from cohort mean — likely identity-
    shortcut sources).

Modes (auto-detected from cache dir contents):
  pose       — per-clip .npz with key 'data' shape (3, T, 27, 1).
               Treats wrist+MCP+fingertip subset (idx 7..26) as the hand region.
  dino       — per-clip .npz with key 'features' shape (D, T, 3, 1)
               (region 0 = left hand, 1 = right hand, 2 = face).

Usage:
    python tools/diagnose_hand_crop_quality.py \\
        --cache-dir data/bdsl_cache \\
        --output results/hand_detection_by_signer_pose.md

    python tools/diagnose_hand_crop_quality.py \\
        --cache-dir data/bdsl_dino_cache \\
        --output results/hand_detection_by_signer_dino.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_SIGNER_RE = re.compile(r"U(\d+)W", re.IGNORECASE)


def _signer_of(name):
    m = _SIGNER_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def _detect_mode(sample_npz):
    """Look at one npz to decide pose vs dino layout."""
    with np.load(sample_npz) as h:
        keys = list(h.keys())
        if "data" in keys:
            shape = h["data"].shape
            if len(shape) == 4 and shape[0] == 3 and shape[2] == 27:
                return "pose"
        if "features" in keys:
            shape = h["features"].shape
            if len(shape) == 4 and shape[2] == 3:
                return "dino"
    raise RuntimeError(f"could not detect cache layout from {sample_npz}: "
                       f"keys={keys}")


def _process_pose_clip(npz_path):
    """For a pose .npz: return (T, frame_has_lhand, frame_has_rhand)."""
    with np.load(npz_path) as h:
        data = h["data"]   # (3, T, 27, 1)
    # Right hand: indices 7..16  (10 joints). Left hand: 17..26.
    rh = data[:, :, 7:17, 0]
    lh = data[:, :, 17:27, 0]
    # A hand is "detected" in a frame iff any of its 10 joints has non-zero coords.
    rh_active = (np.abs(rh).sum(axis=(0, 2)) > 0)   # (T,)
    lh_active = (np.abs(lh).sum(axis=(0, 2)) > 0)
    T = data.shape[1]
    return T, lh_active, rh_active, None  # no face in pose mode


def _process_dino_clip(npz_path):
    """For a DINOv2 .npz: return (T, lhand_active, rhand_active, face_active)."""
    with np.load(npz_path) as h:
        feat = h["features"]   # (D, T, 3, 1)
    # An active crop is one whose feature vector is non-zero (the extractor
    # writes a zero vector for missing crops).
    norm = np.linalg.norm(feat[:, :, :, 0], axis=0)   # (T, 3)
    lh = norm[:, 0] > 1e-6
    rh = norm[:, 1] > 1e-6
    fc = norm[:, 2] > 1e-6
    return feat.shape[1], lh, rh, fc


def _process_clip(npz_path, mode):
    if mode == "pose":
        return _process_pose_clip(npz_path)
    if mode == "dino":
        return _process_dino_clip(npz_path)
    raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="directory of per-clip .npz files (pose or dino layout)")
    ap.add_argument("--output", default="results/hand_detection_by_signer.md")
    ap.add_argument("--mode", choices=["auto", "pose", "dino"], default="auto")
    args = ap.parse_args()

    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"cache dir not found: {args.cache_dir} — "
                         "did you forget to run pose / DINOv2 extraction?")

    # Walk for .npz files.
    npz_paths = []
    for dirpath, _dirs, files in os.walk(args.cache_dir):
        for fn in files:
            if fn.lower().endswith(".npz"):
                npz_paths.append(os.path.join(dirpath, fn))
    if not npz_paths:
        raise SystemExit(f"no .npz files in {args.cache_dir}")
    print(f"found {len(npz_paths)} clips in {args.cache_dir}")

    mode = args.mode
    if mode == "auto":
        mode = _detect_mode(npz_paths[0])
    print(f"mode: {mode}")

    per_signer_T = defaultdict(int)
    per_signer_lh = defaultdict(int)
    per_signer_rh = defaultdict(int)
    per_signer_face = defaultdict(int)
    per_signer_clips = defaultdict(int)
    skipped = 0

    for path in npz_paths:
        signer = _signer_of(path)
        if signer is None:
            skipped += 1
            continue
        try:
            T, lh, rh, face = _process_clip(path, mode)
        except Exception as e:
            print(f"[warn] failed {path}: {e}")
            skipped += 1
            continue
        per_signer_T[signer] += int(T)
        per_signer_lh[signer] += int(lh.sum())
        per_signer_rh[signer] += int(rh.sum())
        if face is not None:
            per_signer_face[signer] += int(face.sum())
        per_signer_clips[signer] += 1

    if skipped:
        print(f"[warn] skipped {skipped} clips (no signer ID or load failure)")

    signers = sorted(per_signer_T.keys())
    if not signers:
        raise SystemExit("no clips with parseable signer ID")

    # Aggregate stats
    lh_rates = np.asarray([per_signer_lh[s] / max(1, per_signer_T[s]) for s in signers])
    rh_rates = np.asarray([per_signer_rh[s] / max(1, per_signer_T[s]) for s in signers])
    face_rates = (np.asarray([per_signer_face[s] / max(1, per_signer_T[s])
                              for s in signers])
                  if mode == "dino" else None)

    # Cohort stats (use train-signer subset only — pretrain/val/test signers
    # might genuinely differ for non-shortcut reasons, e.g. different recording).
    try:
        from preprocessing.bdsl_signer_split import SIGNER_SPLIT
        train_signers = set(SIGNER_SPLIT["train"])
    except Exception:
        train_signers = set(signers)

    train_lh = [lh_rates[i] for i, s in enumerate(signers) if s in train_signers]
    train_rh = [rh_rates[i] for i, s in enumerate(signers) if s in train_signers]
    lh_mean, lh_std = float(np.mean(train_lh)), float(np.std(train_lh, ddof=0))
    rh_mean, rh_std = float(np.mean(train_rh)), float(np.std(train_rh, ddof=0))

    def _flag(rate, mean, std):
        if std == 0:
            return ""
        z = (rate - mean) / std
        if abs(z) >= 2.0:
            return f" **{z:+.1f}SD**"
        if abs(z) >= 1.5:
            return f" {z:+.1f}SD"
        return ""

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Hand/face detection by signer ({mode} cache)\n\n")
        f.write(f"Cache: `{args.cache_dir}`  \n")
        f.write(f"Train-signer cohort statistics (excludes val/test/pretrain):\n")
        f.write(f"- Left-hand detection rate:  mean={lh_mean*100:.1f}%, std={lh_std*100:.1f}\n")
        f.write(f"- Right-hand detection rate: mean={rh_mean*100:.1f}%, std={rh_std*100:.1f}\n\n")
        f.write("Outlier flag: |z| >= 1.5 = mild, >= 2.0 = strong (potential identity-shortcut source)\n\n")
        f.write("| signer | split | clips | total frames | L-hand % | R-hand % |")
        if mode == "dino":
            f.write(" face % |")
        f.write("\n|---:|---|---:|---:|---:|---:|")
        if mode == "dino":
            f.write("---:|")
        f.write("\n")

        for i, s in enumerate(signers):
            try:
                split = next(name for name, group in SIGNER_SPLIT.items() if s in group)
            except (NameError, StopIteration):
                split = "?"
            lh_pct = lh_rates[i] * 100
            rh_pct = rh_rates[i] * 100
            row = (f"| U{s:02d} | {split} | {per_signer_clips[s]} | "
                   f"{per_signer_T[s]} | "
                   f"{lh_pct:.1f}%{_flag(lh_rates[i], lh_mean, lh_std)} | "
                   f"{rh_pct:.1f}%{_flag(rh_rates[i], rh_mean, rh_std)} |")
            if mode == "dino":
                face_pct = face_rates[i] * 100
                row += f" {face_pct:.1f}% |"
            f.write(row + "\n")

    # Console summary
    print(f"\n=== Hand-detection summary (mode={mode}) ===")
    print(f"train signers: L-hand mean={lh_mean*100:.1f}±{lh_std*100:.1f}%; "
          f"R-hand mean={rh_mean*100:.1f}±{rh_std*100:.1f}%")
    print(f"per-signer rates:")
    for i, s in enumerate(signers):
        try:
            split = next(name for name, group in SIGNER_SPLIT.items() if s in group)
        except (NameError, StopIteration):
            split = "?"
        flag_lh = _flag(lh_rates[i], lh_mean, lh_std)
        flag_rh = _flag(rh_rates[i], rh_mean, rh_std)
        print(f"  U{s:02d} [{split:<8}] clips={per_signer_clips[s]:>4} "
              f"frames={per_signer_T[s]:>7} "
              f"L={lh_rates[i]*100:5.1f}%{flag_lh}  "
              f"R={rh_rates[i]*100:5.1f}%{flag_rh}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
