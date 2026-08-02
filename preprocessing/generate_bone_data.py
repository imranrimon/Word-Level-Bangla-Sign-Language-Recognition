"""Generate bone-modality NPY files from joint-modality NPYs.

For each frame, bone = child_joint - parent_joint, computed over the
edges defined in `graph.sign_27.Graph.inward`. The root joint's bone
vector is left as zero.

Output convention follows the rest of the pipeline: bone NPY lives
alongside the joint NPY in the same split directory, named with a
`_bone` suffix on the data file. Labels are NOT duplicated — bone
training configs reuse the same `<split>_label.pkl` as joint.

Usage:

    # Default — process every (train, val, test) found under --data-dir:
    python preprocessing/generate_bone_data.py --data-dir data/bdsl_si

    # Restrict to one or more splits:
    python preprocessing/generate_bone_data.py --data-dir data/bdsl_si \\
        --splits train val test

    # Image modality:
    python preprocessing/generate_bone_data.py --data-dir data/bdsl_img

    # Custom output (rarely needed):
    python preprocessing/generate_bone_data.py \\
        --data-dir data/bdsl --output-dir data/bdsl_bone_alt \\
        --splits train val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.sign_27 import Graph    # noqa: E402


def _joint_to_bone(data, inward_edges):
    """data: (N, C, T, V, M) -> bone (N, C, T, V, M).

    For every edge (parent v1, child v2) in `inward_edges`:
        bone[..., v2, :] = joint[..., v2, :] - joint[..., v1, :]
    Roots (joints with no parent edge) keep their bone vectors at zero.
    """
    bone = np.zeros_like(data)
    for v1, v2 in inward_edges:
        # v1 is closer-to-root, v2 is distal; assign vector to the child slot.
        bone[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]
    return bone


def generate_bone_data(data_path, out_path, graph=None):
    if not os.path.exists(data_path):
        print(f"[skip] {data_path} not found")
        return False

    print(f"generating bone from: {data_path}")
    data = np.load(data_path)
    if data.ndim != 5:
        print(f"  [skip] expected (N,C,T,V,M); got shape {data.shape}")
        return False
    N, C, T, V, M = data.shape

    if graph is None:
        graph = Graph(labeling_mode="spatial")
    bone = _joint_to_bone(data, graph.inward)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.save(out_path, bone)
    print(f"  shape={bone.shape} wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True,
                    help="directory containing <split>_data.npy joint files")
    ap.add_argument("--output-dir", default=None,
                    help="where to write <split>_data_bone.npy; default = --data-dir")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="splits to process; default = all <split>_data.npy "
                         "files present in --data-dir")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir not found: {data_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else data_dir

    if args.splits:
        splits = args.splits
    else:
        # Auto-discover from <split>_data.npy filenames.
        splits = sorted({
            p.name.replace("_data.npy", "")
            for p in data_dir.glob("*_data.npy")
            if not p.name.endswith("_data_bone.npy")
        })

    if not splits:
        raise SystemExit(f"no <split>_data.npy files found in {data_dir}")

    print(f"data-dir: {data_dir}")
    print(f"output-dir: {out_dir}")
    print(f"splits: {splits}")

    graph = Graph(labeling_mode="spatial")
    n_ok = n_skip = 0
    for split in splits:
        src = data_dir / f"{split}_data.npy"
        dst = out_dir / f"{split}_data_bone.npy"
        if generate_bone_data(str(src), str(dst), graph=graph):
            n_ok += 1
        else:
            n_skip += 1
    print(f"\ndone. wrote {n_ok} bone files; skipped {n_skip}")


if __name__ == "__main__":
    main()
