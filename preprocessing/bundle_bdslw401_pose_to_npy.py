"""Bundle BdSLW401 pose-cache .npz files into a single NPY per split.

BdSLW401 (arXiv:2503.02360, Kaggle hasanssl/bdslw401) ships with author
train/val/test splits already preserved in the pose-cache directory layout:

    data/bdslw401_pose_cache_front/
        train/W<word>S<signer>F_<trial>.npz
        val/  W<word>S<signer>F_<trial>.npz
        test/ W<word>S<signer>F_<trial>.npz

This script walks each split, stacks all clips into a `(N, C, T, V, M)`
NPY array (zero-padded to --max-frames), and writes label.pkl with
sample_names + integer labels. Output layout mirrors data/bdsl_si/:

    data/bdslw401_si/
        train_data.npy   (N, 3, max_frames, 27, 1) float32
        train_label.pkl  tuple (sample_names, labels)
        val_data.npy + val_label.pkl
        test_data.npy + test_label.pkl
        classes.json     {classes: [W001, W002, ...], class_to_idx: {...}}

Note: BdSLW401 class IDs in filenames go from W001 to W401 (1-indexed),
but the NPY labels are 0-indexed internally (W001 -> label 0).

Usage:

    python preprocessing/bundle_bdslw401_pose_to_npy.py \\
        --cache-dir data/bdslw401_pose_cache_front \\
        --output-dir data/bdslw401_si \\
        --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Filename pattern: W<word>S<signer>F_<trial>.npz
# (matches preprocessing/bdslw401_meta.py's _FILENAME_RE but for .npz)
_FILENAME_RE = re.compile(r"^W(\d+)S(\d+)([A-Z]*)_(\d+)\.npz$", re.IGNORECASE)


def _parse_npz(name):
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).upper(), int(m.group(4))


def _scan_split(split_dir):
    """Yield (filepath, word_id, signer_id, view, trial) for every parseable clip."""
    if not split_dir.is_dir():
        return
    for entry in sorted(os.listdir(split_dir)):
        parsed = _parse_npz(entry)
        if parsed is None:
            continue
        word, signer, view, trial = parsed
        yield (split_dir / entry, word, signer, view, trial)


def _bundle_split(split_dir, split_name, max_frames, classes_to_idx,
                  num_channels=3, num_joints=27):
    """Load every clip in split_dir, stack to (N, C, T, V, M), save as NPY."""
    entries = list(_scan_split(split_dir))
    if not entries:
        print(f"[{split_name}] no clips in {split_dir}")
        return None, None
    n = len(entries)
    large = np.zeros((n, num_channels, max_frames, num_joints, 1), dtype=np.float32)
    sample_names = []
    labels = []
    for i, (path, word, signer, view, trial) in enumerate(tqdm(entries, desc=f"[{split_name}] bundle")):
        cls_str = f"W{word:03d}"
        if cls_str not in classes_to_idx:
            print(f"  [warn] unknown class {cls_str} in {path.name}; skipping")
            continue
        with np.load(path) as h:
            data = h["data"]   # (C, T, V, M)
        T = data.shape[1]
        if T > max_frames:
            large[i, :, :max_frames, :, :] = data[:, :max_frames, :, :]
        else:
            large[i, :, :T, :, :] = data
        sample_names.append(path.name.replace(".npz", ".mp4"))
        labels.append(classes_to_idx[cls_str])
    return large, (sample_names, labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/bdslw401_pose_cache_front")
    ap.add_argument("--output-dir", default="data/bdslw401_si")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    cache_root = Path(args.cache_dir)
    if not cache_root.is_dir():
        raise SystemExit(f"--cache-dir not found: {cache_root}")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # First pass: discover the set of word classes across ALL splits.
    class_set = set()
    per_split_counts = defaultdict(int)
    for split in args.splits:
        for _, word, _, _, _ in _scan_split(cache_root / split):
            class_set.add(f"W{word:03d}")
            per_split_counts[split] += 1
    classes = sorted(class_set)
    classes_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"discovered {len(classes)} word classes (W{min(int(c[1:]) for c in classes):03d}"
          f" to W{max(int(c[1:]) for c in classes):03d})")
    for split in args.splits:
        print(f"  {split}: {per_split_counts[split]} clips")

    # Write classes.json once.
    with open(output_root / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_idx": classes_to_idx}, f, indent=2)
    print(f"wrote {output_root / 'classes.json'}")

    # Second pass: bundle each split.
    for split in args.splits:
        large, label = _bundle_split(
            cache_root / split, split, args.max_frames, classes_to_idx,
        )
        if large is None:
            continue
        np.save(output_root / f"{split}_data.npy", large)
        with open(output_root / f"{split}_label.pkl", "wb") as f:
            pickle.dump(label, f)
        print(f"  [{split}] shape={large.shape} -> "
              f"{output_root / f'{split}_data.npy'}")

    # Sanity: signer breakdown.
    print("\n=== Signer breakdown ===")
    for split in args.splits:
        signers = defaultdict(int)
        for _, _, signer, _, _ in _scan_split(cache_root / split):
            signers[signer] += 1
        print(f"  {split}: {len(signers)} signers, "
              f"{sum(signers.values())} clips total")


if __name__ == "__main__":
    main()
