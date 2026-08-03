"""Regenerate BdSLW60 train/val/test NPY files using the signer-independent split.

Typical invocation (from repo root):

    python preprocessing/generate_signer_split_npy.py \
        --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
        --output-dir data/bdsl_si \
        --cache-dir data/bdsl_cache \
        --splits train val test

MediaPipe extraction is the expensive step; results for every clip are
cached as `<cache-dir>/<class>/<basename>.npz`, so subsequent reruns (e.g.
to change the split or re-balance the vocabulary) do not pay the
extraction cost again. Pass `--splits pretrain` to additionally emit an
unlabeled `pretrain_data.npy` for self-supervised pretraining.

Output files per split `S`:
    <output-dir>/<S>_data.npy        (N, 3, max_frames, 27, 1) float32
    <output-dir>/<S>_label.pkl       tuple (sample_names, labels)
    <output-dir>/classes.json        ordered class name -> index map
    <output-dir>/split_manifest.json list of clips contributing to each split
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Make the repo root importable regardless of how the script is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.bdsl_signer_split import (  # noqa: E402
    LABELED_SPLITS,
    ALL_SPLITS,
    scan_dataset,
    summarize_splits,
    find_unparseable,
)


def _cached_or_process(task, cache_dir):
    """Run MediaPipe on one clip, falling back to a cached .npz if present.

    Negative-caching: clips that fail to open/extract (e.g. corrupt/truncated
    downloads that make ffmpeg hang for ~97 s before "moov atom not found")
    get a `<cache-dir>/.failed/<base>.bad` marker, so re-runs skip them
    instantly instead of re-paying the ffmpeg open-timeout. Markers are
    class-independent (keyed by basename), so they can be pre-seeded from a log.
    """
    file_path, label, label_name = task
    cache_path = None
    fail_path = None
    if cache_dir:
        base = os.path.splitext(os.path.basename(file_path))[0]
        cache_path = os.path.join(cache_dir, label_name, base + ".npz")
        fail_path = os.path.join(cache_dir, ".failed", base + ".bad")
        if os.path.exists(cache_path):
            with np.load(cache_path) as handle:
                data = handle["data"]
            return (data, label, label_name, file_path)
        if os.path.exists(fail_path):
            return None  # known-bad clip; skip without invoking ffmpeg

    # Lazy-import so `--dry-run` avoids pulling mediapipe in.
    from preprocessing.preprocess_bdsl import process_video
    try:
        result = process_video(task)
    except Exception:
        traceback.print_exc()
        result = None
    if result is None:
        if fail_path is not None:
            os.makedirs(os.path.dirname(fail_path), exist_ok=True)
            open(fail_path, "a").close()  # negative-cache marker
        return None

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, data=result[0])
    return result


def _save_split(results, split_name, output_dir, max_frames, num_channels=3, num_joints=27):
    if not results:
        print("[{}] no clips, skipping".format(split_name))
        return

    n = len(results)
    large = np.zeros((n, num_channels, max_frames, num_joints, 1), dtype=np.float32)
    sample_names = []
    labels = []
    for i, (data, label, _label_name, file_path) in enumerate(results):
        t = data.shape[1]
        if t > max_frames:
            large[i, :, :max_frames, :, :] = data[:, :max_frames, :, :]
        else:
            large[i, :, :t, :, :] = data
        labels.append(label)
        sample_names.append(os.path.basename(file_path))

    data_path = os.path.join(output_dir, "{}_data.npy".format(split_name))
    label_path = os.path.join(output_dir, "{}_label.pkl".format(split_name))
    np.save(data_path, large)
    with open(label_path, "wb") as f:
        pickle.dump((sample_names, labels), f)
    print("[{}] wrote {} shape={}".format(split_name, data_path, large.shape))


def _process_split(name, tasks, class_to_idx, output_dir, cache_dir, max_frames):
    """Process all clips in `tasks` and save as NPY/PKL for split `name`."""
    mp_tasks = [(fp, class_to_idx[cls], cls) for fp, cls in tasks]
    results = []
    for task in tqdm(mp_tasks, desc="[{}] extract".format(name)):
        res = _cached_or_process(task, cache_dir)
        if res is not None:
            results.append(res)
    _save_split(results, name, output_dir, max_frames)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="folder containing the 60 class subdirectories of .mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where to write <split>_data.npy / <split>_label.pkl",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "optional per-clip cache directory; if present, MediaPipe is skipped "
            "for clips whose cache .npz already exists"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(LABELED_SPLITS),
        choices=list(ALL_SPLITS),
        help="which splits to emit (default: train val test)",
    )
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan the dataset and print the plan without running MediaPipe",
    )
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

    # Deterministic class ordering for the integer label mapping.
    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Persist class ordering and manifest of which clip went where.
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
    manifest = {name: [os.path.basename(fp) for fp, _ in per_split_tasks.get(name, [])]
                for name in ALL_SPLITS}
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Print plan.
    summary = summarize_splits(args.dataset_root)
    unparseable = find_unparseable(args.dataset_root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        s = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, s["clips"], len(s["classes"]), len(s["signers"])
        ))
    print("unparseable .mp4 files: {}".format(len(unparseable)))
    print("writing to: {}".format(args.output_dir))
    if args.cache_dir:
        print("cache dir:  {}".format(args.cache_dir))
    if args.dry_run:
        print("--dry-run set; skipping MediaPipe extraction")
        return

    totals = {}
    for name in args.splits:
        tasks = per_split_tasks.get(name, [])
        if not tasks:
            print("[{}] no parseable clips, skipping".format(name))
            continue
        totals[name] = _process_split(
            name, tasks, class_to_idx, args.output_dir, args.cache_dir, args.max_frames,
        )

    print("done. clips processed per split: {}".format(totals))


if __name__ == "__main__":
    main()
