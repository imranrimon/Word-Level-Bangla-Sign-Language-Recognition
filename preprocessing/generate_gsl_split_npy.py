"""Generate GSL isol. (Greek SL, isolated) train/val/test NPY bundles under a
signer-independent split.

GSL isol. clips are pre-cut per-gloss RGB **frame folders** (JPEGs), not video
files, so unlike LSA64/AUTSL/KArSL we cannot reuse `process_video` (which does
`cv2.VideoCapture` on an .mp4). This module supplies a frame-folder MediaPipe
extractor (`process_frame_dir`) that mirrors `preprocess_bdsl.process_video`'s
keypoint logic EXACTLY (same 27-kpt BODY/HAND indices, same (C,T,V,M) output),
then reuses the cache + `_save_split` machinery so the emitted bundle matches the
BdSLW60/LSA64/KArSL SI format byte-for-byte (set `num_class: 310`).

  python preprocessing/generate_gsl_split_npy.py \
      --iso-root  data/gsl_download/GSL_isolated \
      --csv-dir   data/gsl_download/_split/GSL_split/GSL_isolated \
      --iso-classes data/gsl_download/_split/GSL_split/GSL_isolated/iso_classes.csv \
      --output-dir data/gsl_si --cache-dir data/gsl_cache \
      --splits train val test --jobs 8

MediaPipe extraction is the expensive step; per-clip results are cached as
`<cache-dir>/<class_idx>/<clip_key>.npz`. Use `--limit N` for a smoke test.

Output files per split S:
    <output-dir>/<S>_data.npy   (N, 3, max_frames, 27, 1) float32
    <output-dir>/<S>_label.pkl  tuple (sample_names, labels)
    <output-dir>/classes.json   ordered class name -> index map (310 glosses)
    <output-dir>/split_manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as _mp
import os
import sys
import traceback
from collections import defaultdict
from functools import partial

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.gsl_signer_split import (  # noqa: E402
    ALL_SPLITS,
    LABELED_SPLITS,
    read_classes,
    scan_dataset,
    summarize_splits,
    write_split_json,
)
from preprocessing.generate_signer_split_npy import _save_split  # noqa: E402

# Same 27-keypoint mapping as preprocess_bdsl.process_video.
BODY_INDICES = [0, 12, 11, 14, 13, 16, 15]
HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_frames(clip_dir):
    frames = []
    for ext in _IMG_EXTS:
        frames.extend(glob.glob(os.path.join(clip_dir, "*" + ext)))
        frames.extend(glob.glob(os.path.join(clip_dir, "*" + ext.upper())))
    # sort by numeric suffix if present (frame_0001.jpg), else lexicographic
    def _key(p):
        b = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(ch for ch in b if ch.isdigit())
        return (int(digits) if digits else 0, b)
    return sorted(set(frames), key=_key)


def process_frame_dir(clip_dir):
    """MediaPipe Holistic over a folder of JPEG frames -> (3,T,27,1) float32.

    Mirrors preprocess_bdsl.process_video keypoint extraction exactly, but reads
    an ordered image sequence instead of decoding a video. Returns None on
    failure / empty.
    """
    import cv2
    import mediapipe as mp

    frames = _list_frames(clip_dir)
    if not frames:
        return None

    mp_holistic = mp.solutions.holistic
    try:
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:  # pragma: no cover
        print(f"Error init holistic: {e}")
        return None

    frames_data = []

    def get_kp(landmark_list, idx):
        if landmark_list and idx < len(landmark_list.landmark):
            lm = landmark_list.landmark[idx]
            return [lm.x, lm.y, lm.visibility if hasattr(lm, "visibility") else 1.0]
        return [0, 0, 0]

    for fp in frames:
        image = cv2.imread(fp)
        if image is None:
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame_kps = np.zeros((3, 27))
        if results.pose_landmarks:
            for i, idx in enumerate(BODY_INDICES):
                frame_kps[:, i] = get_kp(results.pose_landmarks, idx)
        if results.right_hand_landmarks:
            for i, idx in enumerate(HAND_INDICES):
                frame_kps[:, 7 + i] = get_kp(results.right_hand_landmarks, idx)
        if results.left_hand_landmarks:
            for i, idx in enumerate(HAND_INDICES):
                frame_kps[:, 17 + i] = get_kp(results.left_hand_landmarks, idx)
        frames_data.append(frame_kps)

    holistic.close()
    if not frames_data:
        return None
    data = np.array(frames_data).transpose(1, 0, 2)  # (T,C,V)->(C,T,V)
    data = data[..., np.newaxis]                       # (C,T,V,1)
    return data


def _clip_key(clip_dir):
    """Stable cache key from the clip's relative dir, e.g.
    'police1_signer1_rep1_glosses/glosses0000' -> 'police1_signer1_rep1_glosses__glosses0000'."""
    parts = os.path.normpath(clip_dir).split(os.sep)
    return "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _cached_or_process(task, cache_dir):
    """task = (clip_dir, label:int, label_name:str). Cache-aware, negative-caching."""
    clip_dir, label, label_name = task
    cache_path = fail_path = None
    key = _clip_key(clip_dir)
    if cache_dir:
        cache_path = os.path.join(cache_dir, str(label), key + ".npz")
        fail_path = os.path.join(cache_dir, ".failed", key + ".bad")
        if os.path.exists(cache_path):
            with np.load(cache_path) as h:
                data = h["data"]
            return (data, label, label_name, clip_dir)
        if os.path.exists(fail_path):
            return None
    try:
        data = process_frame_dir(clip_dir)
    except Exception:
        traceback.print_exc()
        data = None
    if data is None:
        if fail_path is not None:
            os.makedirs(os.path.dirname(fail_path), exist_ok=True)
            open(fail_path, "a").close()
        return None
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, data=data)
    return (data, label, label_name, clip_dir)


def _process_split(name, tasks, output_dir, cache_dir, max_frames, jobs):
    if jobs > 1:
        worker = partial(_cached_or_process, cache_dir=cache_dir)
        results = []
        with _mp.Pool(jobs) as pool:
            for res in tqdm(pool.imap_unordered(worker, tasks),
                            total=len(tasks), desc="[{}] extract x{}".format(name, jobs)):
                if res is not None:
                    results.append(res)
    else:
        results = []
        for t in tqdm(tasks, desc="[{}] extract".format(name)):
            res = _cached_or_process(t, cache_dir)
            if res is not None:
                results.append(res)
    _save_split(results, name, output_dir, max_frames)
    return len(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iso-root", required=True,
                    help="extracted GSL isolated dataset root (has *_glosses/ dirs)")
    ap.add_argument("--csv-dir", required=True,
                    help="dir with {train,val,test}_greek_iso.csv (GSL_split/GSL_isolated)")
    ap.add_argument("--iso-classes", required=True,
                    help="iso_classes.csv (canonical 310-gloss vocabulary)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--splits", nargs="+", default=list(LABELED_SPLITS), choices=list(ALL_SPLITS))
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help=">0: at most N clips/split (smoke)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--split-json", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    csvs = [os.path.join(args.csv_dir, "{}_greek_iso.csv".format(s))
            for s in ("train", "val", "test")]

    # Canonical class ordering from iso_classes.csv (sorted for determinism).
    classes = sorted(read_classes(args.iso_classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx,
                   "note": "GSL isol. 310-gloss vocabulary (iso_classes.csv)"},
                  f, ensure_ascii=False, indent=2)

    # Group clips by our signer-disjoint split.
    per_split = defaultdict(list)
    unknown_gloss = 0
    for clip_dir, gloss, _signer, split_name in scan_dataset(args.iso_root, csvs):
        if gloss not in class_to_idx:
            unknown_gloss += 1
            continue
        per_split[split_name].append((clip_dir, class_to_idx[gloss], gloss))

    manifest = {name: [os.path.basename(os.path.dirname(t[0])) + "/" + os.path.basename(t[0])
                       for t in per_split.get(name, [])]
                for name in ALL_SPLITS}
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if args.split_json:
        write_split_json(args.split_json)

    # Plan (require_exists to report clips actually present on disk).
    summary = summarize_splits(args.iso_root, csvs, require_exists=True)
    print("{:<10} {:>7} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        s = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>7} {:>8} {:>8}".format(
            name, s["clips"], len(s["classes"]), len(s["signers"])))
    print("classes(vocab): {} | rows with unknown gloss: {}".format(len(classes), unknown_gloss))
    print("writing to: {}".format(args.output_dir))
    if args.dry_run:
        print("--dry-run set; skipping MediaPipe extraction")
        return

    totals = {}
    for name in args.splits:
        tasks = per_split.get(name, [])
        # Skip clips whose frame dir is missing on disk (rare partial extract).
        tasks = [t for t in tasks if os.path.isdir(t[0])]
        if args.limit and len(tasks) > args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print("[{}] no clips on disk, skipping".format(name))
            continue
        totals[name] = _process_split(name, tasks, args.output_dir, args.cache_dir,
                                      args.max_frames, args.jobs)
    print("done. clips processed per split: {}".format(totals))


if __name__ == "__main__":
    main()
