"""Extract MediaPipe Holistic 27-keypoint poses for every .mp4 under a root.

Output is per-clip cached `.npz` files mirroring the input directory tree:
    <cache_dir>/<rel/path/to/video>.npz   with key 'data' shaped (3, T, 27, 1)

Generic helper used to feed:
  * BdSLW401 (SSL pretraining corpus for Option C)
  * BdSL60-SingleTrial (cross-recording-condition test set for Stage D)
  * any other folder of .mp4 clips.

Usage:
    python preprocessing/extract_pose_cache.py \
        --root data/bdsl60_singletrial \
        --cache-dir data/bdsl60_singletrial_pose_cache

    python preprocessing/extract_pose_cache.py \
        --root data/bdslw401_raw/Front \
        --cache-dir data/bdslw401_pose_cache_front
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.preprocess_bdsl import process_video  # noqa: E402


def _walk_videos(root):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                yield os.path.join(dirpath, fn)


def _cache_path(video_path, root, cache_dir):
    rel = os.path.relpath(video_path, root)
    rel_no_ext = os.path.splitext(rel)[0] + ".npz"
    return os.path.join(cache_dir, rel_no_ext)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder containing .mp4 files (recursive)")
    ap.add_argument("--cache-dir", required=True, help="where to write per-clip .npz")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="skip clips whose .npz already exists in the cache (default on)",
    )
    ap.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="re-process even clips whose cache file already exists",
    )
    ap.add_argument("--label", type=int, default=0,
                    help="dummy label passed through process_video; ignored downstream")
    ap.add_argument("--max-files", type=int, default=None,
                    help="optional cap on number of clips to process (debug)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    videos = list(_walk_videos(args.root))
    if args.max_files is not None:
        videos = videos[: args.max_files]
    print(f"found {len(videos)} .mp4 files under {args.root}")

    cached = skipped = 0
    failed = []
    for vp in tqdm(videos, desc="pose extract"):
        cp = _cache_path(vp, args.root, args.cache_dir)
        if args.skip_existing and os.path.exists(cp) and os.path.getsize(cp) > 0:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
        try:
            result = process_video((vp, args.label, "<no-class>"))
            if result is None:
                failed.append(vp)
                continue
            data = result[0]            # (3, T, 27, 1)
            np.savez_compressed(cp, data=data)
            cached += 1
        except Exception as exc:
            print(f"FAIL {vp}: {exc}")
            failed.append(vp)

    print(f"done. cached={cached}  skipped={skipped}  failed={len(failed)}")
    if failed:
        fail_log = os.path.join(args.cache_dir, "_failures.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"failures written to {fail_log}")


if __name__ == "__main__":
    main()
