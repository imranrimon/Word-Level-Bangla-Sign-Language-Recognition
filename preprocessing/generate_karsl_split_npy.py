"""KArSL (Arabic SL) -> 27-keypoint pose, signer-INDEPENDENT split.

Gap-4 fix: a 4th, distant target language to test the distance-aware-lambda law
(Arabic is far from the ASL+BdSL pretraining pool -> predicts small optimal lambda).

KArSL ships as per-signer .7z archives of .mp4 clips organised by class range
(data/karsl_download/<SS>/{train,test}/<range>.7z, each -> <class>/<sample>.mp4).
This script un-7z's a class range for all 3 signers, then bundles a signer-disjoint
3-way split (train=signer01, val=signer02, test=signer03) with the SAME dataset-agnostic
MediaPipe->27kpt extractor used for BdSLW60/LSA64/AUTSL, so the format matches exactly.

  python preprocessing/generate_karsl_split_npy.py \
      --archive-root data/karsl_download --range 0001-0070 \
      --work-dir data/karsl_extracted --output-dir data/karsl_si \
      --cache-dir data/karsl_cache --jobs 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.generate_signer_split_npy import _process_split          # noqa: E402
try:
    from preprocessing.generate_lsa64_split_npy import _process_split_parallel  # noqa: E402
except Exception:
    _process_split_parallel = None

# 2 training signers so low-data fine-tuning isn't starved (1-signer @10% was ~5
# clips/class -> chance). val = stratified 12% slice of train (selection only); test
# signer U03 is the signer-independent metric.
TRAIN_SIGNERS = [1, 2]
TEST_SIGNERS = [3]
VAL_FRAC = 0.12


def _extract(archive_root, rng, work_dir):
    import py7zr
    for ss in ("01", "02", "03"):
        dest = os.path.join(work_dir, ss)
        os.makedirs(dest, exist_ok=True)
        for sub in ("train", "test"):
            arc = os.path.join(archive_root, ss, sub, f"{rng}.7z")
            if not os.path.exists(arc):
                print(f"  [skip] {arc} missing"); continue
            print(f"  extracting {arc} -> {dest}")
            with py7zr.SevenZipFile(arc, "r") as z:
                z.extractall(path=dest)


def _collect_tasks(work_dir):
    """Return {signer_int: [(mp4_path, class_name), ...]} from work_dir/<SS>/<class>/*.mp4."""
    per_signer = {}
    classes = set()
    for ss in ("01", "02", "03"):
        sdir = os.path.join(work_dir, ss)
        if not os.path.isdir(sdir):
            continue
        tasks = []
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    tasks.append((os.path.join(cdir, f), cls))
                    classes.add(cls)
        per_signer[int(ss)] = tasks
    return per_signer, sorted(classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", default="data/karsl_download")
    ap.add_argument("--range", default="0001-0070", help="class-range archive to use")
    ap.add_argument("--work-dir", default="data/karsl_extracted")
    ap.add_argument("--output-dir", default="data/karsl_si")
    ap.add_argument("--cache-dir", default="data/karsl_cache")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if not args.skip_extract:
        _extract(args.archive_root, args.range, args.work_dir)

    per_signer, classes = _collect_tasks(args.work_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"classes={len(classes)}  per-signer clip counts: "
          + ", ".join(f"U{s}={len(t)}" for s, t in sorted(per_signer.items())))

    import json
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)

    import random as _random
    from collections import defaultdict as _dd
    train_all = [t for s in TRAIN_SIGNERS for t in per_signer.get(s, [])]
    test_tasks = [t for s in TEST_SIGNERS for t in per_signer.get(s, [])]
    # stratified val carve from train (by class)
    by_cls = _dd(list)
    for t in train_all:
        by_cls[t[1]].append(t)
    rng = _random.Random(42)
    val_tasks, train_tasks = [], []
    for c, ts in by_cls.items():
        rng.shuffle(ts)
        k = max(1, int(round(len(ts) * VAL_FRAC)))
        val_tasks += ts[:k]; train_tasks += ts[k:]
    splits = {"train": train_tasks, "val": val_tasks, "test": test_tasks}
    for split, tasks in splits.items():
        if not tasks:
            print(f"[{split}] no clips; skip"); continue
        print(f"[{split}]: {len(tasks)} clips (~{len(tasks)/max(1,len(classes)):.1f}/class)")
        if args.jobs > 1 and _process_split_parallel is not None:
            _process_split_parallel(split, tasks, class_to_idx, args.output_dir,
                                    args.cache_dir, args.max_frames, args.jobs)
        else:
            _process_split(split, tasks, class_to_idx, args.output_dir,
                           args.cache_dir, args.max_frames)
    print("done: KArSL SI bundle (train U01+U02, test U03)")


if __name__ == "__main__":
    main()
