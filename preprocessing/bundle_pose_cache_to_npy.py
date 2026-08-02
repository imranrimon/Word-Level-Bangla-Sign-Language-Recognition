"""Bundle per-clip MediaPipe pose caches into a single (N, 3, T_max, 27, 1) NPY.

Generic — any pose-cache directory + any label callback works. Used to produce:
  * `data/bdsl60_singletrial_eval/` for Stage D cross-recording eval
  * `data/bdslw401_class/{train,test}_*.npy` for the BdSLW401 401-way benchmark

Filename-to-label callbacks live inline because each dataset has a different
naming convention; add a new callback if bundling a new dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Per-dataset filename callbacks
# ---------------------------------------------------------------------------

_BDSL60_ST_RE = re.compile(r"^U(\d+)W(\d+)([A-Z]*)\.npz$", re.I)
_BDSLW401_RE = re.compile(r"^W(\d+)S(\d+)([A-Z]*)_(\d+)\.npz$", re.I)


def _bdsl60_singletrial_label(path, context):
    """BdSL60-SingleTrial: U<signer>W<word>F.npz. Uses canonical word-id map."""
    m = _BDSL60_ST_RE.match(os.path.basename(path))
    if m is None:
        return None
    word_id = int(m.group(2))
    class_name = context["word_id_to_class"].get(str(word_id))
    if class_name is None:
        return None
    label = context["class_to_idx"].get(class_name)
    if label is None:
        return None
    return label, os.path.basename(path)


def _bdslw401_label(path, context):
    """BdSLW401: W<word>S<signer>F_<trial>.npz. Word id is the 401-way class directly."""
    m = _BDSLW401_RE.match(os.path.basename(path))
    if m is None:
        return None
    # word_id is 1..401 -> 0..400 index
    label = int(m.group(1)) - 1
    return label, os.path.basename(path)


CALLBACKS = {
    "bdsl60_singletrial": _bdsl60_singletrial_label,
    "bdslw401_401class": _bdslw401_label,
}


# ---------------------------------------------------------------------------
# Bundler core
# ---------------------------------------------------------------------------

def _walk_npz(root):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".npz"):
                yield os.path.join(dirpath, fn)


def bundle(pose_cache_dir, output_dir, split_name, label_fn, context,
           max_frames=300, num_channels=3, num_joints=27):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(_walk_npz(pose_cache_dir))
    print(f"found {len(paths)} .npz files under {pose_cache_dir}")

    records = []
    skipped = 0
    for p in paths:
        label_info = label_fn(p, context)
        if label_info is None:
            skipped += 1
            continue
        label, name = label_info
        records.append((p, label, name))
    print(f"labelled: {len(records)}  skipped: {skipped}")

    if not records:
        raise RuntimeError("no clips had a valid label; check the callback + mapping")

    n = len(records)
    large = np.zeros((n, num_channels, max_frames, num_joints, 1), dtype=np.float32)
    labels = []
    sample_names = []
    for i, (path, label, name) in enumerate(tqdm(records, desc=f"[{split_name}] pack")):
        with np.load(path) as h:
            data = h["data"]  # (C, T, V, M)
        t = data.shape[1]
        if t > max_frames:
            large[i, :, :max_frames, :, :] = data[:, :max_frames, :, :]
        else:
            large[i, :, :t, :, :] = data
        labels.append(int(label))
        sample_names.append(name)

    data_path = os.path.join(output_dir, f"{split_name}_data.npy")
    label_path = os.path.join(output_dir, f"{split_name}_label.pkl")
    np.save(data_path, large)
    with open(label_path, "wb") as f:
        pickle.dump((sample_names, labels), f)
    print(f"wrote {data_path} shape={large.shape}")
    print(f"wrote {label_path}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-name", default="eval")
    ap.add_argument("--callback", required=True, choices=list(CALLBACKS.keys()))
    ap.add_argument("--word-id-mapping",
                    default="splits/word_id_mapping.json",
                    help="path to word-id mapping JSON (for bdsl60_singletrial)")
    ap.add_argument("--classes-json",
                    default="data/bdsl_si/classes.json",
                    help="path to classes.json for label index alignment")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    context = {}
    if os.path.exists(args.classes_json):
        with open(args.classes_json) as f:
            cj = json.load(f)
        context["class_to_idx"] = cj.get("class_to_idx", {})
    if os.path.exists(args.word_id_mapping):
        with open(args.word_id_mapping) as f:
            wm = json.load(f)
        context["word_id_to_class"] = wm.get("word_id_to_class", {})

    fn = CALLBACKS[args.callback]
    bundle(args.cache_dir, args.output_dir, args.split_name, fn, context,
           max_frames=args.max_frames)


if __name__ == "__main__":
    main()
