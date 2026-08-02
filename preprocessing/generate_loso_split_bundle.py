"""Generate one BdSLW60 LOSO (leave-one-signer-out) data bundle.

For audit fix #2. Reads the per-clip MediaPipe pose cache at <cache-dir>/
<class>/<basename>.npz (built once by generate_signer_split_npy.py) and
re-bundles it into <split>_data.npy + <split>_label.pkl files under a fold-
specific output directory, using a custom (test_signer, val_signer) assignment
in place of the canonical SIGNER_SPLIT.

By only re-bundling cached pose tensors (no MediaPipe), one LOSO fold takes
~30-60 seconds vs the 2-3 hours of the original extraction.

Usage (one fold):

    python preprocessing/generate_loso_split_bundle.py \\
        --cache-dir data/bdsl_cache \\
        --output-dir data/bdsl_si_loso/test_U02_val_U15 \\
        --test-signer 2 --val-signer 15

The companion driver `tools/run_loso.py` iterates this script + main.py over
every LOSO fold.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.bdsl_signer_split import (  # noqa: E402
    SIGNER_SPLIT, parse_filename,
)


def build_loso_assignment(test_signer, val_signer):
    """Build a {signer_id: split_name} dict for one LOSO fold.

    All other 'full' signers (the 11 in the union of train+val+test of the
    canonical SIGNER_SPLIT) go to TRAIN. Pretrain-pool signers (partial
    vocabulary coverage) remain in PRETRAIN.

    Raises if test_signer == val_signer or either is in the pretrain pool.
    """
    full_signers = set(SIGNER_SPLIT["train"]) | set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])
    pretrain = set(SIGNER_SPLIT["pretrain"])
    if test_signer == val_signer:
        raise ValueError("test_signer and val_signer must differ")
    for label, sid in (("test", test_signer), ("val", val_signer)):
        if sid not in full_signers:
            raise ValueError(
                f"{label}_signer U{sid:02d} is not in the 11 full-vocabulary "
                f"signers {sorted(full_signers)}"
            )
    assignment = {}
    for s in full_signers - {test_signer, val_signer}:
        assignment[s] = "train"
    assignment[val_signer] = "val"
    assignment[test_signer] = "test"
    for s in pretrain:
        assignment[s] = "pretrain"
    return assignment


def _scan_cache(cache_dir):
    """Yield (cache_path, class_name, signer) for every parseable cached clip."""
    for class_name in sorted(os.listdir(cache_dir)):
        cls_dir = os.path.join(cache_dir, class_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith(".npz"):
                continue
            # Cache files are named after the .mp4: U<s>W<w>F[_<sess>]_trial_<n>_[LR].npz.
            # Reuse parse_filename by appending .mp4 for the regex.
            parsed = parse_filename(fn.replace(".npz", ".mp4"))
            if parsed is None:
                continue
            signer = parsed[0]
            yield (os.path.join(cls_dir, fn), class_name, signer)


def _save_split(results, split_name, output_dir, max_frames=300,
                num_channels=3, num_joints=27):
    if not results:
        print(f"[{split_name}] no clips, skipping")
        return 0
    n = len(results)
    large = np.zeros((n, num_channels, max_frames, num_joints, 1), dtype=np.float32)
    sample_names = []
    labels = []
    for i, (data, label, _label_name, cache_path) in enumerate(results):
        t = data.shape[1]
        if t > max_frames:
            large[i, :, :max_frames, :, :] = data[:, :max_frames, :, :]
        else:
            large[i, :, :t, :, :] = data
        labels.append(label)
        # Use the basename WITHOUT .npz so label.pkl stays comparable to the
        # canonical bundle (which stores .mp4 basenames).
        sample_names.append(os.path.basename(cache_path).replace(".npz", ".mp4"))
    np.save(os.path.join(output_dir, f"{split_name}_data.npy"), large)
    with open(os.path.join(output_dir, f"{split_name}_label.pkl"), "wb") as f:
        pickle.dump((sample_names, labels), f)
    print(f"[{split_name}] wrote {n} clips, shape={large.shape}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/bdsl_cache",
                    help="directory of per-clip pose .npz files "
                         "(built once by generate_signer_split_npy.py)")
    ap.add_argument("--output-dir", required=True,
                    help="per-fold output dir (e.g. data/bdsl_si_loso/test_U02_val_U15/)")
    ap.add_argument("--test-signer", type=int, required=True)
    ap.add_argument("--val-signer", type=int, required=True)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test", "pretrain"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, don't write")
    args = ap.parse_args()

    if not os.path.isdir(args.cache_dir):
        raise FileNotFoundError(f"cache dir missing: {args.cache_dir}")

    assignment = build_loso_assignment(args.test_signer, args.val_signer)
    print(f"LOSO assignment: test=U{args.test_signer:02d}, val=U{args.val_signer:02d}")
    print(f"  train: {sorted(s for s,sp in assignment.items() if sp=='train')}")
    print(f"  val:   [{args.val_signer}]")
    print(f"  test:  [{args.test_signer}]")
    print(f"  pretrain (unchanged): {sorted(SIGNER_SPLIT['pretrain'])}")

    # Group cached clips by split via the LOSO assignment.
    per_split = defaultdict(list)
    class_set = set()
    for cache_path, class_name, signer in _scan_cache(args.cache_dir):
        split = assignment.get(signer)
        if split is None:
            continue
        per_split[split].append((cache_path, class_name))
        class_set.add(class_name)

    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"per-split clip counts:")
    for name in ("train", "val", "test", "pretrain"):
        n = len(per_split.get(name, []))
        print(f"  {name:<8}: {n}")

    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx,
                   "loso": {"test_signer": args.test_signer,
                            "val_signer": args.val_signer}}, f, indent=2)

    totals = {}
    for split_name in args.splits:
        tasks = per_split.get(split_name, [])
        if not tasks:
            continue
        results = []
        for cache_path, cls in tqdm(tasks, desc=f"[{split_name}] load-cache"):
            try:
                with np.load(cache_path) as h:
                    data = h["data"]
            except Exception as e:
                print(f"[warn] failed to load {cache_path}: {e}")
                continue
            results.append((data, class_to_idx[cls], cls, cache_path))
        totals[split_name] = _save_split(results, split_name, args.output_dir,
                                         max_frames=args.max_frames)

    print(f"done. wrote to {args.output_dir}; totals: {totals}")


if __name__ == "__main__":
    main()
