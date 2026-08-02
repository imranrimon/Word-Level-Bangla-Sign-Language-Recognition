"""Compute k-means cluster IDs for SHuBERT-style masked pretraining.

For each pose-cache clip in the SSL pool, produce a (T,) int64 array of cluster
IDs. The masked-prediction objective is cross-entropy of predicted cluster vs
ground-truth cluster at masked time steps.

Algorithm:
  1. Load a random `--fit-sample` fraction of frames from the pool.
  2. Fit MiniBatchKMeans with `--num-clusters` on the per-frame feature vector.
  3. For every clip, assign each of its T frames to a cluster, save as a single
     row in the output `.npz` keyed by the clip's path from the manifest.

Feature modes (audit fix #1 — was per-frame pose only, which made the SSL
task trivially solvable from neighbouring frames):

  * `frame`        — bare per-frame pose, dim = C*V = 81. Legacy / ablation only.
  * `pose_motion`  — pose concat first-derivative motion (pose_t - pose_{t-1}),
                     dim = 2*C*V = 162. **Default.** Neighbouring frames now
                     differ via the motion channel even when the pose is static.
  * `window<K>`    — K-frame zero-padded window centred on t,
                     dim = K*C*V (e.g. window4 -> 324). Closer to SHuBERT-for-
                     audio's MFCC-window paradigm. More expensive but captures
                     longer local structure.

Usage:
    python preprocessing/compute_pretrain_targets.py \\
        --manifest data/ssl_pool_manifest.json \\
        --output data/pretrain_kmeans_targets.npz \\
        --num-clusters 64 --fit-sample 0.1 --seed 0 \\
        --feature-mode pose_motion
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from tqdm import tqdm

# Imported at module top so missing deps fail fast (0s) rather than after the
# minutes-long fit-sampling walk. Seen once when the env was missing sklearn.
from sklearn.cluster import MiniBatchKMeans


DEFAULT_NUM_CHANNELS = 3
DEFAULT_NUM_JOINTS = 27


def _feature_dim(feature_mode, num_channels=DEFAULT_NUM_CHANNELS,
                 num_joints=DEFAULT_NUM_JOINTS):
    base = num_channels * num_joints
    if feature_mode == "frame":
        return base
    if feature_mode == "pose_motion":
        return 2 * base
    m = re.fullmatch(r"window(\d+)", feature_mode)
    if m:
        return int(m.group(1)) * base
    raise ValueError(f"unknown feature_mode: {feature_mode!r}")


def _clip_to_per_frame_features(data, feature_mode):
    """Convert (C, T_eff, V, M) -> (T_eff, D) feature vectors per frame."""
    C, T, V, M = data.shape
    pose = data[:, :, :, 0].transpose(1, 0, 2).reshape(T, C * V).astype(np.float32)

    if feature_mode == "frame":
        return pose

    if feature_mode == "pose_motion":
        motion = np.zeros_like(pose)
        if T > 1:
            motion[1:] = pose[1:] - pose[:-1]
        return np.concatenate([pose, motion], axis=1)

    m = re.fullmatch(r"window(\d+)", feature_mode)
    if m:
        K = int(m.group(1))
        if K < 1:
            raise ValueError(f"window size must be >= 1, got {K}")
        half_l = K // 2  # frames before t in the window
        out = np.zeros((T, K * C * V), dtype=np.float32)
        # Slot k in the window represents pose at position t + (k - half_l).
        # For each k, copy pose[src_lo:src_hi] into out[dst_lo:dst_hi, slot_k].
        for k in range(K):
            offset = k - half_l
            src_lo = max(offset, 0)
            src_hi = T + min(offset, 0)        # T if offset>=0, else T-|offset|
            dst_lo = max(-offset, 0)
            length = src_hi - src_lo
            if length > 0:
                out[dst_lo:dst_lo + length,
                    k * C * V:(k + 1) * C * V] = pose[src_lo:src_hi]
        return out

    raise ValueError(f"unknown feature_mode: {feature_mode!r}")


def _load_clip_frames(path, feature_mode):
    with np.load(path) as h:
        data = h["data"]            # (C, T, V, M)
    valid = data.sum(axis=(0, 2, 3)) != 0  # True where any keypoint nonzero
    if not valid.any():
        return np.zeros((0, _feature_dim(feature_mode)), dtype=np.float32), 0
    last = int(np.where(valid)[0].max()) + 1
    data = data[:, :last]
    flat = _clip_to_per_frame_features(data, feature_mode)
    return flat, flat.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-clusters", type=int, default=64)
    ap.add_argument("--fit-sample", type=float, default=0.10,
                    help="fraction of frames to use for KMeans fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8192,
                    help="MiniBatchKMeans batch size")
    ap.add_argument(
        "--feature-mode", default="pose_motion",
        help="per-frame feature for k-means clustering. "
             "'frame' (legacy, 81-dim, trivial-from-neighbours), "
             "'pose_motion' (default, 162-dim, pose+first-derivative), "
             "or 'window<K>' (e.g. window4 = 324-dim, K consecutive frames stacked).",
    )
    ap.add_argument(
        "--per-source-cap", type=int, default=None,
        help="audit fix #12 / Path A: cap fit-sample to N frames per source "
             "(based on manifest 'source' field). With multi-source pools "
             "(e.g. BdSL + WLASL where BdSLW401 alone has 4.2M frames vs "
             "WLASL's 1.3M), proportional --fit-sample lets the largest source "
             "dominate cluster geometry. Capping per-source forces balance. "
             "Recommended: 200000 for K=64. Default (None) keeps the legacy "
             "proportional behaviour.",
    )
    args = ap.parse_args()

    feature_mode = args.feature_mode
    feature_dim = _feature_dim(feature_mode)

    with open(args.manifest) as f:
        manifest = json.load(f)
    clips = manifest["clips"]
    print(f"loaded {len(clips)} clips from manifest")
    print(f"feature_mode = {feature_mode} (dim = {feature_dim})")
    if args.per_source_cap is not None:
        print(f"per-source fit-sample cap = {args.per_source_cap} frames "
              f"(balanced k-means, audit fix #12)")

    rng = np.random.default_rng(args.seed)

    # Fit phase — sample frames per-clip, optionally cap per source.
    print(f"sampling {args.fit_sample:.0%} of frames for KMeans fit...")
    per_source_samples = {}    # source -> list of (T, D) per-clip slices
    per_source_total = {}
    for entry in tqdm(clips, desc="fit-sample"):
        flat, T = _load_clip_frames(entry["path"], feature_mode)
        if T == 0:
            continue
        k = max(1, int(T * args.fit_sample))
        idx = rng.choice(T, size=k, replace=False)
        source = entry.get("source", "_default_")
        per_source_samples.setdefault(source, []).append(flat[idx])
        per_source_total[source] = per_source_total.get(source, 0) + int(k)

    # Apply per-source cap if requested.
    if args.per_source_cap is not None:
        cap = int(args.per_source_cap)
        for source, chunks in per_source_samples.items():
            if not chunks:
                continue
            arr = np.concatenate(chunks, axis=0)
            if arr.shape[0] > cap:
                sel = rng.choice(arr.shape[0], size=cap, replace=False)
                per_source_samples[source] = [arr[sel]]
            else:
                per_source_samples[source] = [arr]

    fit_chunks = [c for chunks in per_source_samples.values() for c in chunks]
    fit_arr = (np.concatenate(fit_chunks, axis=0)
               if fit_chunks
               else np.zeros((0, feature_dim), dtype=np.float32))
    print(f"fit corpus shape: {fit_arr.shape}")
    print(f"  per-source contribution to fit corpus:")
    for source in sorted(per_source_samples.keys()):
        n_kept = sum(c.shape[0] for c in per_source_samples[source])
        n_proposed = per_source_total.get(source, 0)
        if args.per_source_cap is not None and n_proposed > args.per_source_cap:
            print(f"    {source}: {n_kept} (capped from {n_proposed})")
        else:
            print(f"    {source}: {n_kept}")
    if fit_arr.shape[0] < args.num_clusters:
        raise RuntimeError(f"not enough frames ({fit_arr.shape[0]}) to fit "
                           f"{args.num_clusters} clusters")

    print(f"fitting MiniBatchKMeans K={args.num_clusters}...")
    km = MiniBatchKMeans(
        n_clusters=args.num_clusters,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init=3,
    )
    km.fit(fit_arr)

    # Assignment phase
    print("assigning cluster IDs to every clip...")
    targets = {}
    for entry in tqdm(clips, desc="assign"):
        flat, T = _load_clip_frames(entry["path"], feature_mode)
        if T == 0:
            continue
        ids = km.predict(flat).astype(np.int32)
        targets[entry["path"]] = ids

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        cluster_centers=km.cluster_centers_.astype(np.float32),
        feature_mode=np.array(feature_mode),
        feature_dim=np.array(feature_dim, dtype=np.int32),
        keys=np.array(list(targets.keys()), dtype=object),
        **{f"t_{i}": v for i, v in enumerate(targets.values())},
    )
    print(f"wrote {args.output}: {len(targets)} clips")


if __name__ == "__main__":
    main()
