"""Generate LSA64 train/val/test NPY bundles under a signer-independent split.

Reuses the dataset-agnostic MediaPipe -> 27-keypoint extractor
(`preprocess_bdsl.process_video`, via the cache-aware helpers in
`generate_signer_split_npy`) and the LSA64 signer split
(`lsa64_signer_split`). Emits the same format as the BdSLW60 SI bundle so
`main.py` / `feeders.feeder.Feeder` consume it unchanged (set `num_class: 64`).

Typical invocation (from repo root):

    python preprocessing/generate_lsa64_split_npy.py \
        --dataset-root data/lsa64_raw \
        --output-dir   data/lsa64_si \
        --cache-dir    data/lsa64_cache \
        --splits train val test

MediaPipe extraction is the expensive step; per-clip results are cached as
`<cache-dir>/<class>/<basename>.npz`, so re-splitting is free. Use
`--limit N` for a fast smoke test (processes only the first N clips per split).

Output files per split `S`:
    <output-dir>/<S>_data.npy        (N, 3, max_frames, 27, 1) float32
    <output-dir>/<S>_label.pkl       tuple (sample_names, labels)
    <output-dir>/classes.json        ordered class name -> index map
    <output-dir>/split_manifest.json list of clips contributing to each split
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as _mp
import os
import sys
from collections import defaultdict
from functools import partial

from tqdm import tqdm

# Make the repo root importable regardless of how the script is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.lsa64_signer_split import (  # noqa: E402
    ALL_SPLITS,
    LABELED_SPLITS,
    find_unparseable,
    scan_dataset,
    summarize_splits,
    write_split_json,
)
# Reuse the cache-aware extraction + save helpers from the BdSLW60 generator.
from preprocessing.generate_signer_split_npy import (  # noqa: E402
    _cached_or_process,
    _process_split,
    _save_split,
)


def _process_split_parallel(name, tasks, class_to_idx, output_dir, cache_dir, max_frames, jobs):
    """Same as `_process_split` but extracts clips across `jobs` worker processes.

    Order is not preserved (imap_unordered), which is fine: each result carries
    its own file path, so sample_names/labels stay self-consistent per clip.
    """
    mp_tasks = [(fp, class_to_idx[cls], cls) for fp, cls in tasks]
    worker = partial(_cached_or_process, cache_dir=cache_dir)
    results = []
    with _mp.Pool(jobs) as pool:
        for res in tqdm(pool.imap_unordered(worker, mp_tasks),
                        total=len(mp_tasks), desc="[{}] extract x{}".format(name, jobs)):
            if res is not None:
                results.append(res)
    _save_split(results, name, output_dir, max_frames)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                        help="dir containing LSA64 .mp4 clips (recursively, e.g. data/lsa64_raw)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None,
                        help="optional per-clip .npz cache (skips MediaPipe if present)")
    parser.add_argument("--splits", nargs="+", default=list(LABELED_SPLITS),
                        choices=list(ALL_SPLITS))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, process at most N clips per split (smoke test)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel MediaPipe worker processes (default 1)")
    parser.add_argument("--split-json", default=None,
                        help="optional path to also write the canonical split JSON "
                             "(e.g. splits/lsa64_signer_independent.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="scan and print the plan without running MediaPipe")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Group parseable clips by split.
    per_split_tasks = defaultdict(list)
    class_set = set()
    for filepath, class_name, _signer, split_name in scan_dataset(args.dataset_root):
        per_split_tasks[split_name].append((filepath, class_name))
        class_set.add(class_name)

    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
    manifest = {name: [os.path.basename(fp) for fp, _ in per_split_tasks.get(name, [])]
                for name in ALL_SPLITS}
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if args.split_json:
        write_split_json(args.split_json)

    # Print plan.
    summary = summarize_splits(args.dataset_root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        s = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, s["clips"], len(s["classes"]), len(s["signers"])
        ))
    print("classes: {} | unparseable .mp4: {}".format(
        len(classes), len(find_unparseable(args.dataset_root))))
    print("writing to: {}".format(args.output_dir))
    if args.dry_run:
        print("--dry-run set; skipping MediaPipe extraction")
        return

    totals = {}
    for name in args.splits:
        tasks = per_split_tasks.get(name, [])
        if args.limit and len(tasks) > args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print("[{}] no parseable clips, skipping".format(name))
            continue
        if args.jobs > 1:
            totals[name] = _process_split_parallel(
                name, tasks, class_to_idx, args.output_dir, args.cache_dir,
                args.max_frames, args.jobs,
            )
        else:
            totals[name] = _process_split(
                name, tasks, class_to_idx, args.output_dir, args.cache_dir, args.max_frames,
            )

    print("done. clips processed per split: {}".format(totals))


if __name__ == "__main__":
    main()
