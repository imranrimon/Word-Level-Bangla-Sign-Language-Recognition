"""Generate AUTSL train/val/test NPY bundles using AUTSL's OFFICIAL split.

Unlike BdSLW60/LSA64 (where we derive a signer-independent split ourselves),
AUTSL ships an official, already signer-disjoint train/val/test partition
(train 31 signers / val 6 / test 6, no overlap) with per-split label CSVs
(`<sample_id>,<class_id>`, e.g. `signer0_sample1,41`). Clips are
`<split>/signer<S>_sample<N>_color.mp4`. We honour that split directly.

Reuses the dataset-agnostic MediaPipe -> 27-keypoint extractor and the
cache-aware parallel helpers, emitting the same NPY format as the other
datasets so `main.py` consumes it with `num_class: 226`.

    python preprocessing/generate_autsl_split_npy.py \
        --dataset-root data/autsl_download \
        --output-dir   data/autsl_si \
        --cache-dir    data/autsl_cache \
        --splits train val test --jobs 32

Use `--limit N` for a fast smoke test.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.generate_signer_split_npy import _process_split  # noqa: E402
from preprocessing.generate_lsa64_split_npy import _process_split_parallel  # noqa: E402

NUM_CLASSES = 226
SPLITS = ("train", "val", "test")


def _load_labels(csv_path):
    """sample_id -> class_id (int). CSV rows are '<sample_id>,<class_id>' (no header)."""
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            sid, cid = row[0].strip(), row[1].strip()
            if not cid.isdigit():
                continue  # skip any header/blank
            labels[sid] = int(cid)
    return labels


def _build_tasks(root, split):
    """Return list of (filepath, class_name) for a split, labelled from its CSV."""
    labels = _load_labels(os.path.join(root, "{}_labels.csv".format(split)))
    tasks, missing = [], 0
    for fp in sorted(glob.glob(os.path.join(root, split, "*.mp4"))):
        sid = os.path.basename(fp).replace("_color.mp4", "").replace(".mp4", "")
        if sid not in labels:
            missing += 1
            continue
        tasks.append((fp, str(labels[sid])))  # class_name = str(class_id)
    return tasks, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                        help="dir with train/ val/ test/ and *_labels.csv (data/autsl_download)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help=">0: at most N clips/split (smoke)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Identity class map: label strings "0".."225" -> ints 0..225.
    class_to_idx = {str(i): i for i in range(NUM_CLASSES)}
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": [str(i) for i in range(NUM_CLASSES)],
                   "class_to_idx": class_to_idx, "note": "AUTSL official 226-class ids"}, f, indent=2)

    per_split = {}
    print("{:<8} {:>8} {:>8}".format("split", "clips", "no_label"))
    for s in SPLITS:
        tasks, missing = _build_tasks(args.dataset_root, s)
        per_split[s] = tasks
        print("{:<8} {:>8} {:>8}".format(s, len(tasks), missing))
    print("writing to:", args.output_dir)
    if args.dry_run:
        print("--dry-run; skipping MediaPipe")
        return

    totals = {}
    for s in args.splits:
        tasks = per_split.get(s, [])
        if args.limit and len(tasks) > args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print("[{}] no clips, skipping".format(s))
            continue
        if args.jobs > 1:
            totals[s] = _process_split_parallel(s, tasks, class_to_idx, args.output_dir,
                                                args.cache_dir, args.max_frames, args.jobs)
        else:
            totals[s] = _process_split(s, tasks, class_to_idx, args.output_dir,
                                       args.cache_dir, args.max_frames)
    print("done. processed per split:", totals)


if __name__ == "__main__":
    main()
