#!/bin/bash
# Step 5a (P1 fix): k-means targets on the v2 pool. Dropout diagnostic (job J36) -> pose_motion_masked.
set -eo pipefail
OUT=data/pretrain_kmeans_targets_bdsl_asl_v2.npz
[ -f $OUT ] && { echo "exists: $OUT"; exit 0; }
python -u preprocessing/compute_pretrain_targets_v2.py --manifest data/ssl_pool_manifest_bdsl_asl_v2.json \
  --output $OUT.tmp.npz --num-clusters 64 --fit-sample 0.1 --seed 0 --per-source-cap 200000 \
  --feature-mode pose_motion_masked
mv $OUT.tmp.npz $OUT
python - <<'PY'
import numpy as np, json
t = np.load("data/pretrain_kmeans_targets_bdsl_asl_v2.npz", allow_pickle=True)
keys = t["keys"]; n = len(keys)
occ = np.bincount(np.concatenate([t[f"t_{i}"] for i in range(n)]), minlength=64)
print("clips", n, "feature_mode", t["feature_mode"], "occupancy min/max share", occ.min()/occ.sum(), occ.max()/occ.sum())
PY
