#!/usr/bin/env python
"""
Build a top-60-class subset of the ASL-Citizen SI bundle so it matches the
BdSLW60 (60-class) scale for the P1 cross-lingual transfer predictor.

Source: data/aslcitizen_si/{train,val,test}_{data.npy,label.pkl}  (2731 classes)
Output: data/aslcitizen60_si/{train,val,test}_{data.npy,label.pkl}  (60 classes)

Class selection: the 60 most frequent class IDs in the TRAIN split. Ties are
broken by smallest class id for determinism. Kept classes are remapped to
0..59 sorted by their original class id (stable, reproducible).

Labels keep the repo Feeder format: each *_label.pkl is a pickle of
[sample_names_list, labels_list]. classes.json (if present) is filtered to the
60 kept classes, with class_to_idx rebuilt for the new 0..59 indexing.

CPU-only. Uses mmap_mode='r' when loading data so the 3.9 GB train array is
never fully materialised in RAM (only the selected rows are copied out).
"""
import collections
import json
import os
import pickle
import subprocess

import numpy as np


def _prestripe(path, count=4):
    """Pre-create the target file striped across `count` Lustre OSTs, so a
    single slow/congested OST cannot stall the whole write (observed on
    a shared Lustre file system: a single-OST 109 MB write hung in osc_extent_wait at the
    64 MiB dirty-page limit, while a 4-OST striped write of the same array
    completed in 0.2 s). No-op on non-Lustre filesystems (lfs absent -> caught).
    """
    try:
        if os.path.exists(path):
            os.remove(path)
        subprocess.run(
            ["lfs", "setstripe", "-c", str(count), path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # non-Lustre or lfs unavailable -> plain write is fine

SRC_DIR = "data/aslcitizen_si"
DST_DIR = "data/aslcitizen60_si"
SPLITS = ["train", "val", "test"]
TOP_K = 60


def load_label(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    names, labels = obj[0], obj[1]
    return list(names), list(labels)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # --- pick top-60 classes from the TRAIN split frequencies ---
    train_names, train_labels = load_label(os.path.join(SRC_DIR, "train_label.pkl"))
    freq = collections.Counter(train_labels)
    # sort by (-count, class_id): most frequent first, ties -> smallest class id
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:TOP_K]
    kept_old = sorted(c for c, _ in top)  # remap sorted by old class id
    assert len(kept_old) == TOP_K, f"expected {TOP_K} classes, got {len(kept_old)}"
    remap = {old: new for new, old in enumerate(kept_old)}
    kept_set = set(kept_old)

    print(f"[top-{TOP_K}] selected class ids (old -> new):")
    for old in kept_old:
        print(f"  {old:5d} -> {remap[old]:2d}   (train count={freq[old]})")

    # --- filter each split ---
    for split in SPLITS:
        names, labels = load_label(os.path.join(SRC_DIR, f"{split}_label.pkl"))
        data = np.load(os.path.join(SRC_DIR, f"{split}_data.npy"), mmap_mode="r")
        assert data.shape[0] == len(labels), (
            f"{split}: data rows {data.shape[0]} != labels {len(labels)}"
        )

        keep_idx = [i for i, lb in enumerate(labels) if lb in kept_set]
        keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)

        # copy only selected rows out of the mmap (materialises just the subset)
        sub_data = np.ascontiguousarray(data[keep_idx_arr])
        sub_names = [names[i] for i in keep_idx]
        sub_labels = [remap[labels[i]] for i in keep_idx]

        # write to a temp file then atomically rename, so a filesystem stall
        # never leaves a truncated/corrupt .npy in place.
        data_out = os.path.join(DST_DIR, f"{split}_data.npy")
        tmp_out = os.path.join(DST_DIR, f"{split}_data.tmp.npy")
        _prestripe(tmp_out, count=4)  # stripe across OSTs to avoid single-OST stall
        with open(tmp_out, "r+b" if os.path.exists(tmp_out) else "wb") as f:
            np.save(f, sub_data)  # writes exactly tmp_out (fd already open)
            f.flush()
            os.fsync(f.fileno())  # flush this file only (not a global sync)
        os.replace(tmp_out, data_out)

        lbl_out = os.path.join(DST_DIR, f"{split}_label.pkl")
        tmp_lbl = lbl_out + ".tmp"
        with open(tmp_lbl, "wb") as f:
            pickle.dump([sub_names, sub_labels], f)
        os.replace(tmp_lbl, lbl_out)

        n_cls = len(set(sub_labels))
        print(
            f"[{split}] kept {len(keep_idx)} / {len(labels)} samples "
            f"-> data {sub_data.shape}, {n_cls} classes present"
        )
        del data  # release mmap

    # --- filter classes.json if present ---
    src_cj = os.path.join(SRC_DIR, "classes.json")
    if os.path.exists(src_cj):
        with open(src_cj) as f:
            cj = json.load(f)
        old_classes = cj["classes"]
        new_classes = [old_classes[old] for old in kept_old]  # ordered by new idx
        new_cj = {
            "classes": new_classes,
            "class_to_idx": {name: i for i, name in enumerate(new_classes)},
            "note": (
                f"top-{TOP_K} most-frequent ASL-Citizen classes (by train freq, "
                f"ties -> smallest old id); remapped 0..{TOP_K - 1} sorted by old id. "
                f"Derived from {src_cj}."
            ),
        }
        with open(os.path.join(DST_DIR, "classes.json"), "w") as f:
            json.dump(new_cj, f, indent=2)
        print(f"[classes.json] wrote {len(new_classes)} kept class names")
    else:
        print("[classes.json] source not found -> skipped")

    print(f"\nDONE. final class count = {TOP_K}")


if __name__ == "__main__":
    main()
