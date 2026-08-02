"""Build Uni-Sign-format label files for BdSLW60 under the canonical SI split.

Uni-Sign (external/Uni-Sign) expects, per dataset, gzip-pickled dicts
    {clip_id: {"name": str, "gloss": str, "text": str, "video_path": str}}
one file per split (labels.train / labels.dev / labels.test), where for the
ISLR task `text` is the gloss word (here: the romanized BdSLW60 class name).

Output (default):
    external/Uni-Sign/data/BdSLW60/labels.train   (train signers, canonical SI)
    external/Uni-Sign/data/BdSLW60/labels.dev     (val signer 15)
    external/Uni-Sign/data/BdSLW60/labels.test    (test signers 2, 13)

`video_path` is the clip filename (e.g. U11W37F_trial_0_R.mp4); Uni-Sign
resolves it against config.py's rgb_dirs/pose_dirs entries, which for a flat
layout means pointing both at a single directory containing all clips /
pose pkls. Pass --nested to instead emit "<class>/<file>" paths matching the
BdSLW30 on-disk layout.

Usage:
    python preprocessing/build_unisign_bdslw60_labels.py \
        --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
        --output-dir external/Uni-Sign/data/BdSLW60 --nested
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.bdsl_signer_split import scan_dataset  # noqa: E402

# Uni-Sign split naming: train / dev / test. Our SI "val" becomes "dev";
# the pretrain pool is excluded (labeled splits only).
_SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}


def build(dataset_root, output_dir, nested):
    buckets = {v: {} for v in _SPLIT_MAP.values()}
    for filepath, class_name, _signer, split_name in scan_dataset(dataset_root):
        if split_name not in _SPLIT_MAP:
            continue  # pretrain pool
        fn = os.path.basename(filepath)
        clip_id = os.path.splitext(fn)[0]
        video_path = f"{class_name}/{fn}" if nested else fn
        buckets[_SPLIT_MAP[split_name]][clip_id] = {
            "name": clip_id,
            "gloss": "",
            "text": class_name,
            "video_path": video_path,
        }

    os.makedirs(output_dir, exist_ok=True)
    for split, entries in buckets.items():
        out = os.path.join(output_dir, f"labels.{split}")
        with gzip.open(out, "wb") as f:
            pickle.dump(entries, f)
        classes = {e["text"] for e in entries.values()}
        print(f"{out}: {len(entries)} clips, {len(classes)} classes")
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root",
                    default="Word_level_Bangla_Sign_Language_Dataset/BdSLW30")
    ap.add_argument("--output-dir", default="external/Uni-Sign/data/BdSLW60")
    ap.add_argument("--nested", action="store_true",
                    help="emit <class>/<file> video_path (BdSLW30 layout)")
    args = ap.parse_args()
    build(args.dataset_root, args.output_dir, args.nested)


if __name__ == "__main__":
    main()
