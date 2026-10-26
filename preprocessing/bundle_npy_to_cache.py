"""Explode an SI bundle (<dir>/<split>_data.npy + <split>_label.pkl) into a
per-clip pose cache directory of .npz files -- the format the SSL pool builder
(preprocessing/build_ssl_pool_manifest.py) and k-means target computer
(preprocessing/compute_pretrain_targets.py) consume.

A bundle stores every clip time-padded to a common T (e.g. LSFB is padded to
T=300 with trailing all-zero frames). The SSL pose_motion k-means targets are
sensitive to those zero frames -- a static-zero tail dominates the motion
channel with zeros and pollutes cluster geometry. So each clip is TRIMMED to its
last non-empty frame (a frame is "empty" when the pose is entirely 0 across the
C, V, M axes) before being written. Cache format matches the per-clip caches:
a single array under key 'data' of shape (C, T_eff, V, M) = (3, T_eff, 27, 1).

Each output file is named after the clip's sample name (sanitized to a safe
filename) plus its bundle index, guaranteeing uniqueness even if two samples
share a name. Writes are per-clip and fault-isolated: a failure on one clip is
logged and skipped rather than aborting the whole (13k-clip) run, and an
existing output file is left in place so the tool is resumable.

Usage:
    python preprocessing/bundle_npy_to_cache.py \\
        --bundle-dir data/lsfb_si \\
        --split train \\
        --output-dir data/lsfb_cache
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(name) -> str:
    """Make an arbitrary sample name safe as a single filename component."""
    s = str(name).strip().replace(os.sep, "_")
    s = _SAFE_RE.sub("_", s)
    s = s.strip("._") or "clip"
    # keep filenames well under any filesystem limit even before the _<idx> suffix
    return s[:180]


def _trim_trailing_zero_frames(clip: np.ndarray) -> np.ndarray:
    """clip: (C, T, V, M). Drop trailing frames that are all-zero across C,V,M.

    Returns (C, T_eff, V, M) with T_eff >= 1 (keeps a single frame if a clip is
    entirely zero, so downstream min-frames filtering can drop it explicitly)."""
    # valid[t] is True where any keypoint value in frame t is nonzero.
    valid = clip.sum(axis=(0, 2, 3)) != 0
    if not valid.any():
        return clip[:, :1]
    last = int(np.where(valid)[0].max()) + 1
    return clip[:, :last]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True,
                    help="dir holding <split>_data.npy + <split>_label.pkl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", required=True,
                    help="destination per-clip .npz cache directory")
    args = ap.parse_args()

    data_path = os.path.join(args.bundle_dir, f"{args.split}_data.npy")
    label_path = os.path.join(args.bundle_dir, f"{args.split}_label.pkl")
    for p in (data_path, label_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    os.makedirs(args.output_dir, exist_ok=True)

    # mmap the (N, C, T, V, M) bundle -- never materialize it all in RAM.
    data = np.load(data_path, mmap_mode="r")
    with open(label_path, "rb") as f:
        label = pickle.load(f)

    # label is typically (names, labels); tolerate a bare list of names too.
    if isinstance(label, (list, tuple)) and len(label) >= 1 and \
            isinstance(label[0], (list, tuple, np.ndarray)):
        names = list(label[0])
    else:
        names = list(label)

    n = data.shape[0]
    if len(names) != n:
        print(f"[WARN] {len(names)} names but {n} clips; naming missing ones by index")

    print(f"exploding {n} clips from {data_path} -> {args.output_dir}")
    written = 0
    skipped_exist = 0
    failed = 0
    for i in range(n):
        name = names[i] if i < len(names) else f"clip{i}"
        out_name = f"{_sanitize(name)}_{i:06d}.npz"
        out_path = os.path.join(args.output_dir, out_name)
        if os.path.exists(out_path):
            skipped_exist += 1
            continue
        try:
            clip = np.asarray(data[i])                 # (C, T, V, M)
            clip = _trim_trailing_zero_frames(clip)
            # atomic-ish: write to a temp file (ending .npz so np.savez_compressed
            # does not append another suffix) then rename, so a killed job never
            # leaves a half-written .npz that a resume would treat as done.
            tmp_path = out_path + f".tmp{os.getpid()}.npz"
            np.savez_compressed(tmp_path, data=clip)
            os.replace(tmp_path, out_path)
            written += 1
        except OSError:  # I/O error (e.g. Lustre OST offline) must not silently leave a hole in the cache
            raise
        except Exception as exc:  # noqa: BLE001 -- fault-isolate a single clip
            failed += 1
            print(f"[WARN] clip {i} ({name!r}) failed: {exc}")
            # clean up any stray temp file
            stray = out_path + f".tmp{os.getpid()}.npz"
            try:
                if os.path.exists(stray):
                    os.remove(stray)
            except OSError:
                pass
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} (written={written} skipped={skipped_exist} failed={failed})")

    print(f"done: written={written} skipped_existing={skipped_exist} "
          f"failed={failed} total={n}")
    if failed:
        print(f"[WARN] {failed} clips failed; cache is usable but incomplete")


if __name__ == "__main__":
    main()
