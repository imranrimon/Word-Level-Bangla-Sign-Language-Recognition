"""Generate ASL Citizen train/val/test NPY bundles using ASL Citizen's OFFICIAL
participant-disjoint split.

ASL Citizen (Desai et al., NeurIPS D&B 2023) ships an official, already
participant-disjoint train/val/test partition (train 35 signers / val 6 /
test 11, no overlap; 40154 / 10304 / 32941 videos over 2731 signs). The point
of this external validation is their signer-disjoint protocol, so we honour the
official split directly and DO NOT re-split.

The download is RGB videos only (the official ST-GCN baseline extracts
MediaPipe Holistic keypoints itself, ~7 days). We reuse the repo's own
dataset-agnostic MediaPipe -> 27-keypoint extractor (the same one used for
BdSLW60 / AUTSL / LSA64 / ...), which is faster (model_complexity=1, cached,
parallel) and emits the repo's canonical 27-node layout consumed unchanged by
`main.py` / `feeders.feeder.Feeder` / `graph.sign_27` (set `num_class: 2731`).
ASL Citizen's own ST-GCN baseline uses the identical OpenHands 27-keypoint set,
so the skeleton matches.

CSV schema (from microsoft/ASL-citizen-code aslcitizen_dataset_pose.py; header row):
    col 0 = participant / user id
    col 1 = video filename (relative, e.g. seed<...>.mp4 under videos/)
    col 2 = gloss (class label)
We read by header name when present, else positionally (0,1,2).

Sample names are written as ``<user>__<videobasename>`` so the retrieval /
per-signer metric tool can recover the participant id (users are disjoint
across splits, so this doubles as a participant-disjointness check).

    python preprocessing/generate_aslcitizen_split_npy.py \
        --dataset-root data/aslcitizen_download \
        --output-dir   data/aslcitizen_si \
        --cache-dir    data/aslcitizen_cache \
        --splits train val test --jobs 32

Use `--limit N` for a fast smoke test (first N clips/split).
Use `--class-cap K` to keep only the K most frequent glosses (documented cap
for compute-bound runs; default 0 = keep all 2731).

Output files per split S:
    <output-dir>/<S>_data.npy        (N, 3, max_frames, 27, 1) float32
    <output-dir>/<S>_label.pkl       tuple (sample_names, labels)
    <output-dir>/classes.json        ordered gloss -> index map (fixed across splits)
    <output-dir>/split_manifest.json per-split clip + user listing (disjointness proof)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


SPLITS = ("train", "val", "test")

# ASL Citizen CSV header aliases (case-insensitive) -> canonical field.
_USER_KEYS = ("participant", "participant id", "user", "user id", "signer")
_FILE_KEYS = ("video file", "video", "filename", "file", "video file name", "vid")
_GLOSS_KEYS = ("gloss", "label", "sign", "asl gloss", "english gloss")


def _find_split_csv(root, split):
    """Locate the official CSV for a split anywhere under the download root.

    Matches ASL Citizen's shipped names (e.g. aslcitizen_training_set.csv,
    aslcitizen_val_set.csv, aslcitzen_test_set.csv [sic]) and generic
    train/val/test .csv. Returns the single best match or raises.
    """
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation", "dev"),
        "test": ("test",),
    }[split]
    cands = []
    for csv_path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        low = os.path.basename(csv_path).lower()
        # exclude the pose-mapping helper CSVs the repo generates
        if "pose_map" in low or "mapping" in low or "confusion" in low:
            continue
        if any(a in low for a in aliases):
            # guard: 'val' alias must not match 'validation' inside 'train'? fine.
            # avoid 'test' matching nothing else; avoid 'val' inside 'interval' etc.
            cands.append(csv_path)
    # de-dup, prefer shortest path (top-level splits dir over nested)
    cands = sorted(set(cands), key=lambda p: (len(p.split(os.sep)), len(p)))
    if not cands:
        raise FileNotFoundError(
            "no {} split CSV under {} (looked for {})".format(split, root, aliases))
    return cands[0]


def _resolve_video_dir(root):
    """Return the directory that actually contains the .mp4 files."""
    # common layouts: <root>/videos/*.mp4 or <root>/ASL_Citizen/videos/*.mp4
    for pat in ("videos", os.path.join("*", "videos"), os.path.join("*", "*", "videos")):
        for d in glob.glob(os.path.join(root, pat)):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.mp4")):
                return d
    # fallback: deepest dir with the most mp4s
    best, best_n = None, 0
    for d, _sub, files in os.walk(root):
        n = sum(1 for f in files if f.endswith(".mp4"))
        if n > best_n:
            best, best_n = d, n
    if best is None:
        raise FileNotFoundError("no .mp4 files found under {}".format(root))
    return best


def _header_index(header, keys):
    low = [h.strip().lower() for h in header]
    for k in keys:
        if k in low:
            return low.index(k)
    return None


def _read_csv(csv_path):
    """-> list of (user, video_filename, gloss). Robust to header/positional CSVs."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return rows
        ui = _header_index(first, _USER_KEYS)
        fi = _header_index(first, _FILE_KEYS)
        gi = _header_index(first, _GLOSS_KEYS)
        has_header = (ui is not None or fi is not None or gi is not None)
        if not has_header:
            # first row was data; default positional layout 0,1,2
            ui, fi, gi = 0, 1, 2
            rows.append((first[ui].strip(), first[fi].strip(), first[gi].strip()))
        else:
            ui = 0 if ui is None else ui
            fi = 1 if fi is None else fi
            gi = 2 if gi is None else gi
        for row in reader:
            if len(row) <= max(ui, fi, gi):
                continue
            rows.append((row[ui].strip(), row[fi].strip(), row[gi].strip()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True,
                        help="unzipped ASL Citizen root (data/aslcitizen_download)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help=">0: at most N clips/split (smoke)")
    parser.add_argument("--class-cap", type=int, default=0,
                        help=">0: keep only the K most frequent glosses (documented cap)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    video_dir = _resolve_video_dir(args.dataset_root)
    print("video dir:", video_dir)

    # 1) read all split CSVs.
    raw = {}
    for s in SPLITS:
        csv_path = _find_split_csv(args.dataset_root, s)
        raw[s] = _read_csv(csv_path)
        print("[{}] csv={} rows={}".format(s, os.path.relpath(csv_path, args.dataset_root), len(raw[s])))

    # 2) fixed gloss ordering over the UNION of all splits (deterministic).
    all_gloss = set()
    for s in SPLITS:
        for _u, _f, g in raw[s]:
            all_gloss.add(g)
    classes = sorted(all_gloss)

    # optional documented class cap (by global frequency across all splits).
    if args.class_cap and args.class_cap < len(classes):
        freq = Counter()
        for s in SPLITS:
            for _u, _f, g in raw[s]:
                freq[g] += 1
        classes = sorted([g for g, _ in freq.most_common(args.class_cap)])
        print("class-cap: keeping {} most-frequent glosses".format(len(classes)))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx,
                   "note": "ASL Citizen official glosses ({} classes)".format(len(classes))},
                  f, indent=2)

    # 3) build (filepath, class) tasks per split, honouring the official split.
    per_split_tasks = {}
    users_per_split = {}
    missing_video = defaultdict(int)
    dropped_class = defaultdict(int)
    for s in SPLITS:
        tasks = []
        users = set()
        for user, vfile, gloss in raw[s]:
            if gloss not in class_to_idx:
                dropped_class[s] += 1
                continue
            base = os.path.basename(vfile)
            fp = os.path.join(video_dir, base)
            if not os.path.exists(fp):
                # try the path as given (some CSVs include a subdir)
                alt = os.path.join(video_dir, vfile)
                if os.path.exists(alt):
                    fp = alt
                else:
                    missing_video[s] += 1
                    continue
            # encode user id into the sample name for per-signer / disjointness checks.
            tag = "{}__{}".format(user, base)
            tasks.append((fp, gloss, tag))
            users.add(user)
        per_split_tasks[s] = tasks
        users_per_split[s] = sorted(users)

    # 4) participant-disjointness check + manifest.
    u_train, u_val, u_test = (set(users_per_split[s]) for s in SPLITS)
    overlaps = {
        "train_val": sorted(u_train & u_val),
        "train_test": sorted(u_train & u_test),
        "val_test": sorted(u_val & u_test),
    }
    disjoint = all(len(v) == 0 for v in overlaps.values())
    manifest = {
        "classes": len(classes),
        "disjoint_participants": disjoint,
        "overlaps": overlaps,
        "per_split": {s: {"clips": len(per_split_tasks[s]),
                          "users": users_per_split[s],
                          "n_users": len(users_per_split[s]),
                          "missing_video": missing_video[s],
                          "dropped_out_of_cap_class": dropped_class[s]}
                      for s in SPLITS},
    }
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("{:<8} {:>8} {:>8} {:>10} {:>10}".format(
        "split", "clips", "users", "missing", "capdrop"))
    for s in SPLITS:
        print("{:<8} {:>8} {:>8} {:>10} {:>10}".format(
            s, len(per_split_tasks[s]), len(users_per_split[s]),
            missing_video[s], dropped_class[s]))
    print("participant-disjoint across train/val/test: {}".format(disjoint))
    if not disjoint:
        print("  !! OVERLAP DETECTED:", {k: v for k, v in overlaps.items() if v})
    print("classes: {} | writing to: {}".format(len(classes), args.output_dir))
    if args.dry_run:
        print("--dry-run; skipping MediaPipe extraction")
        return

    # 5) extract + bundle. _process_split expects tasks as (filepath, class_name);
    #    it re-derives sample_name from the file path via _save_split, so to keep our
    #    user-tagged sample names we pass a per-split tag map through a shim below.
    totals = {}
    for s in args.splits:
        tasks = per_split_tasks.get(s, [])
        if args.limit and len(tasks) > args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print("[{}] no clips, skipping".format(s))
            continue
        totals[s] = _process_split_tagged(
            s, tasks, class_to_idx, args.output_dir, args.cache_dir,
            args.max_frames, args.jobs)
    print("done. processed per split:", totals)


# --- tagged variant: preserve <user>__<basename> sample names -------------------
def _process_split_tagged(name, tasks, class_to_idx, output_dir, cache_dir,
                          max_frames, jobs):
    """Like _process_split(_parallel) but sample_name = user-tagged tag, not basename.

    tasks: list of (filepath, class_name, tag). Extraction/caching is unchanged
    (cache is keyed by class_name + file basename), only the saved sample name
    differs so downstream metrics can recover the participant id.
    """
    import numpy as np
    import pickle
    from preprocessing.generate_signer_split_npy import _cached_or_process

    # tag lookup keyed by absolute filepath.
    tag_of = {fp: tag for fp, _cls, tag in tasks}
    # Sanitize the gloss used ONLY as the cache subdir name (a few ASL Citizen
    # glosses contain '/', e.g. HURDLE/TRIP1, THIS/IT, which would mis-nest the
    # <cache>/<gloss>/<base>.npz path). The integer label (2nd tuple element) is
    # unaffected, so this changes nothing about the class mapping.
    def _safe(g):
        return g.replace("/", "_").replace("\\", "_")
    mp_tasks = [(fp, class_to_idx[cls], _safe(cls)) for fp, cls, _tag in tasks]

    results = []
    if jobs and jobs > 1:
        import multiprocessing as _mp
        from functools import partial
        from tqdm import tqdm
        worker = partial(_cached_or_process, cache_dir=cache_dir)
        with _mp.Pool(jobs) as pool:
            for res in tqdm(pool.imap_unordered(worker, mp_tasks),
                            total=len(mp_tasks), desc="[{}] extract x{}".format(name, jobs)):
                if res is not None:
                    results.append(res)
    else:
        from tqdm import tqdm
        for task in tqdm(mp_tasks, desc="[{}] extract".format(name)):
            res = _cached_or_process(task, cache_dir)
            if res is not None:
                results.append(res)

    if not results:
        print("[{}] no clips extracted, skipping".format(name))
        return 0

    n = len(results)
    large = np.zeros((n, 3, max_frames, 27, 1), dtype=np.float32)
    sample_names, labels = [], []
    for i, (data, label, _label_name, file_path) in enumerate(results):
        t = data.shape[1]
        if t > max_frames:
            large[i, :, :max_frames, :, :] = data[:, :max_frames, :, :]
        else:
            large[i, :, :t, :, :] = data
        labels.append(label)
        sample_names.append(tag_of.get(file_path, os.path.basename(file_path)))

    np.save(os.path.join(output_dir, "{}_data.npy".format(name)), large)
    with open(os.path.join(output_dir, "{}_label.pkl".format(name)), "wb") as f:
        pickle.dump((sample_names, labels), f)
    print("[{}] wrote {}_data.npy shape={}".format(name, name, large.shape))
    return n


if __name__ == "__main__":
    main()
