"""Audit fix #12: report per-cluster source distribution for the SSL k-means.

Loads `data/pretrain_kmeans_targets.npz` (cluster IDs per clip) and joins
against `data/ssl_pool_manifest.json` (path -> source). Reports:

  * total frames per source,
  * per-cluster source occupancy (which source owns each centroid),
  * dominance fractions (max-source share per cluster),
  * imbalance metrics (cluster-size entropy + largest/smallest ratio).

Why we care: with multi-source SSL pools (BdSLW60 + BdSLW401 + BdSLW102_A
+ WLASL), the 64 k-means centroids may collapse onto whichever source
dominates frame count. If that happens, BdSL-frame cluster IDs become
near-random labels — SHuBERT pretraining still runs but the masked-
prediction signal degrades for the target domain.

Usage:
    python tools/diagnose_cluster_occupancy.py \\
        --manifest data/ssl_pool_manifest.json \\
        --targets data/pretrain_kmeans_targets.npz \\
        --output results/cluster_occupancy.md
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def _source_label(source_path):
    """Compact label for a cache directory (basename only)."""
    return os.path.basename(source_path.rstrip("/").rstrip("\\"))


def _entropy_nats(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/ssl_pool_manifest.json")
    ap.add_argument("--targets", default="data/pretrain_kmeans_targets.npz")
    ap.add_argument("--output", default="results/cluster_occupancy.md")
    ap.add_argument("--num-clusters", type=int, default=None,
                    help="override; default reads from cluster_centers shape")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    path_to_source = {c["path"]: _source_label(c["source"]) for c in manifest["clips"]}
    print(f"loaded manifest: {len(path_to_source)} clips, "
          f"{len(set(path_to_source.values()))} sources")

    bundle = np.load(args.targets, allow_pickle=True)
    keys = [str(k) for k in bundle["keys"]]
    cluster_centers = bundle["cluster_centers"]
    K = int(args.num_clusters) if args.num_clusters else int(cluster_centers.shape[0])
    feature_mode = (str(bundle["feature_mode"])
                    if "feature_mode" in bundle.files else "unknown")
    print(f"loaded targets: {len(keys)} clips with cluster IDs, K={K}, "
          f"feature_mode={feature_mode}")

    # per (cluster, source) frame count
    matrix = defaultdict(lambda: np.zeros(K, dtype=np.int64))
    source_total = defaultdict(int)
    cluster_total = np.zeros(K, dtype=np.int64)
    missing_source = 0

    for i, path in enumerate(keys):
        source = path_to_source.get(path)
        if source is None:
            missing_source += 1
            continue
        codes = bundle[f"t_{i}"]
        # Bincount per cluster
        counts = np.bincount(codes, minlength=K)
        matrix[source] += counts
        source_total[source] += int(counts.sum())
        cluster_total += counts

    if missing_source:
        print(f"[warn] {missing_source} target keys were not in the manifest "
              f"(stale targets file?)")

    sources = sorted(source_total.keys())
    total_frames = int(cluster_total.sum())
    print(f"total frames assigned: {total_frames}")

    # Per-source totals + expected vs actual cluster share if uniform.
    src_share = {s: source_total[s] / total_frames for s in sources}

    # Per-cluster dominance (which source owns the cluster + fraction)
    dominance = {}
    entropy = {}
    for c in range(K):
        per_src = np.array([matrix[s][c] for s in sources], dtype=np.float64)
        if per_src.sum() == 0:
            dominance[c] = (None, 0.0, 0)
            entropy[c] = 0.0
            continue
        p = per_src / per_src.sum()
        i_max = int(p.argmax())
        dominance[c] = (sources[i_max], float(p[i_max]), int(per_src.sum()))
        entropy[c] = _entropy_nats(p)

    # Cluster-size imbalance
    cluster_share = cluster_total / total_frames if total_frames > 0 else cluster_total.astype(float)
    largest = float(cluster_total.max()) / total_frames if total_frames > 0 else 0.0
    smallest = float(cluster_total[cluster_total > 0].min()) / total_frames if (cluster_total > 0).any() else 0.0
    cluster_entropy = _entropy_nats(cluster_share)

    # Per-source "cluster ownership": how many clusters have >X% of their
    # frames from this source.
    own_thresh = 0.50
    owned_by = defaultdict(int)
    for c, (src, frac, _n) in dominance.items():
        if src is not None and frac >= own_thresh:
            owned_by[src] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# K-means cluster occupancy by source\n\n")
        f.write(f"Targets file: `{args.targets}`  \n")
        f.write(f"Manifest: `{args.manifest}`  \n")
        f.write(f"Feature mode: `{feature_mode}`, K = {K}, total frames = {total_frames}\n\n")

        f.write("## Per-source totals and cluster ownership\n\n")
        f.write("| source | frames | frame-share | clusters >50% owned | expected-uniform | over/under |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for s in sources:
            owned = owned_by.get(s, 0)
            expected = K * src_share[s]
            f.write(f"| {s} | {source_total[s]} | {src_share[s]*100:.1f}% | "
                    f"{owned}/{K} | {expected:.1f} | "
                    f"{(owned - expected):+.1f} |\n")

        f.write("\n## Cluster-size imbalance\n\n")
        f.write(f"- Cluster-size entropy: **{cluster_entropy:.3f} nats** "
                f"(max possible at K={K}: {np.log(K):.3f})\n")
        f.write(f"- Largest cluster: **{largest*100:.2f}%** of frames\n")
        f.write(f"- Smallest non-empty cluster: **{smallest*100:.3f}%** of frames\n")
        ratio = largest / smallest if smallest > 0 else float("inf")
        f.write(f"- Largest / smallest ratio: **{ratio:.1f}x**\n\n")

        f.write("## Per-cluster dominance (sorted by frame count)\n\n")
        f.write("| cluster | frames | dominant source | dominance | entropy (nats) |\n")
        f.write("|---:|---:|---|---:|---:|\n")
        order = sorted(range(K), key=lambda c: -dominance[c][2])
        for c in order:
            src, frac, n = dominance[c]
            f.write(f"| {c} | {n} | {src or '-'} | {frac*100:.1f}% | {entropy[c]:.3f} |\n")

    # Console summary
    print("\n=== Summary ===")
    print(f"K-means K={K}, total frames={total_frames}, feature_mode={feature_mode}")
    print(f"Cluster-size: largest={largest*100:.2f}%, smallest={smallest*100:.3f}%, "
          f"ratio={ratio:.1f}x, entropy={cluster_entropy:.3f} (max {np.log(K):.3f})")
    print()
    print(f"{'source':<35} {'frames':>10} {'share':>7}  {'owned':>10}  "
          f"{'expected':>9}")
    for s in sources:
        owned = owned_by.get(s, 0)
        expected = K * src_share[s]
        print(f"{s:<35} {source_total[s]:>10} {src_share[s]*100:>6.1f}%  "
              f"{owned:>3}/{K:>3}    {expected:>9.1f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
