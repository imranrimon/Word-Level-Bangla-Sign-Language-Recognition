"""Bundle BdSLW102_A SENTENCE-level pose-cache .npz files into NPY per split.

IMPORTANT — this dataset is **sentence-level**, NOT word-level. The
"Word Label.xlsx" file is a vocabulary listing of the words used across
sentences; it is NOT a per-word clip index. The pose cache lives at:

    data/bdslw102_a_pose_cache/Sentence/
        Masking Data/With Background/<sentence_id>/<signer_id>/<n>_sentence<S>_withBg.npz
        Masking Data/Without Background/...
        Raw Video Data/...

This bundler treats each clip as labelled by its `sentence_id` (folder
name at depth-1 under "Masking Data" or "Raw Video Data"). There are 20
sentence classes (per Sentence Label.xlsx).

Because BdSLW102_A is sentence-level, it is **not usable for the word-
level cross-dataset benchmark** (paper 2's main intent). It is instead
suitable for:
  * a separate sentence-recognition benchmark, or
  * a granularity-comparison study (word- vs sentence-level Bangla SLR).

Our default split policy (since BdSLW102_A ships with no documented
train/val/test partition): random 70/15/15 across clips, seeded.

Usage:

    python preprocessing/bundle_bdslw102a_sentence_pose_to_npy.py \\
        --cache-dir data/bdslw102_a_pose_cache/Sentence \\
        --output-dir data/bdslw102a_sentence \\
        --max-frames 300 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
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


# Filename pattern: <idx>_sentence<id>[_withBg|_withoutBg|_raw].npz
_FILENAME_RE = re.compile(
    r"^(\d+)_sentence(\d+)(?:_(withBg|withoutBg|raw))?\.npz$",
    re.IGNORECASE,
)


def _parse_npz(name):
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), (m.group(3) or "raw").lower()


def _scan_cache(cache_root):
    """Yield (npz_path, sentence_id, signer_id_str, sample_index, variant)."""
    for dirpath, _dirs, files in os.walk(cache_root):
        rel = Path(dirpath).relative_to(cache_root)
        parts = rel.parts
        # We expect ...<variant_branch>/<sentence_id>/<signer_id>/
        if len(parts) < 2:
            continue
        for fn in files:
            parsed = _parse_npz(fn)
            if parsed is None:
                continue
            sample_idx, sentence_in_name, variant = parsed
            sentence_from_dir = parts[-2]
            signer_from_dir = parts[-1]
            # Trust the directory's sentence id (filename's sentence id is
            # typically the same but the dir is the canonical label source).
            try:
                sentence_id = int(sentence_from_dir)
            except ValueError:
                sentence_id = sentence_in_name
            yield (
                Path(dirpath) / fn,
                sentence_id,
                signer_from_dir,
                sample_idx,
                variant,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/bdslw102_a_pose_cache/Sentence")
    ap.add_argument("--output-dir", default="data/bdslw102a_sentence")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument(
        "--variant-filter", default=None,
        choices=["withBg", "withoutBg", "raw"],
        help="restrict to one masking variant (default: keep all)",
    )
    args = ap.parse_args()

    cache_root = Path(args.cache_dir)
    if not cache_root.is_dir():
        raise SystemExit(f"--cache-dir not found: {cache_root}")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Scan all clips.
    entries = list(_scan_cache(cache_root))
    if args.variant_filter:
        entries = [e for e in entries if e[4] == args.variant_filter.lower()]
    if not entries:
        raise SystemExit(f"no clips found in {cache_root} "
                         f"(filter={args.variant_filter})")

    # Build class index from sentence IDs present.
    sentence_ids = sorted({e[1] for e in entries})
    class_to_idx = {sid: i for i, sid in enumerate(sentence_ids)}
    print(f"found {len(entries)} clips covering {len(sentence_ids)} sentence classes "
          f"({sentence_ids[0]}..{sentence_ids[-1]})")

    # Per-sentence count diagnostic.
    per_sent = defaultdict(int)
    per_signer = defaultdict(int)
    per_variant = defaultdict(int)
    for _, sid, signer, _, variant in entries:
        per_sent[sid] += 1
        per_signer[signer] += 1
        per_variant[variant] += 1
    print(f"per-variant: {dict(per_variant)}")
    print(f"unique signer folders: {len(per_signer)}")

    # Deterministic random split.
    rng = random.Random(args.seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "test":  shuffled[:n_test],
        "val":   shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }
    print(f"split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # classes.json
    classes_serialised = [str(s) for s in sentence_ids]
    with open(output_root / "classes.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes_serialised,
            "class_to_idx": {str(k): v for k, v in class_to_idx.items()},
            "note": "BdSLW102_A is sentence-level; class IDs are sentence "
                    "folder names (integers). See Sentence Label.xlsx for "
                    "Bangla sentence text.",
        }, f, indent=2, ensure_ascii=False)
    print(f"wrote {output_root / 'classes.json'}")

    for split_name, items in splits.items():
        if not items:
            continue
        nn = len(items)
        large = np.zeros((nn, 3, args.max_frames, 27, 1), dtype=np.float32)
        sample_names = []
        labels = []
        for i, (path, sid, signer, sample_idx, variant) in enumerate(
            tqdm(items, desc=f"[{split_name}] bundle")
        ):
            with np.load(path) as h:
                data = h["data"]
            T = data.shape[1]
            if T > args.max_frames:
                large[i, :, :args.max_frames, :, :] = data[:, :args.max_frames, :, :]
            else:
                large[i, :, :T, :, :] = data
            sample_names.append(path.name)
            labels.append(class_to_idx[sid])

        np.save(output_root / f"{split_name}_data.npy", large)
        with open(output_root / f"{split_name}_label.pkl", "wb") as f:
            pickle.dump((sample_names, labels), f)
        print(f"  [{split_name}] shape={large.shape} -> "
              f"{output_root / f'{split_name}_data.npy'}")


if __name__ == "__main__":
    main()
