"""Prove the v1 WLASL cache scramble mechanism: v1 cache slots == which V3 indices?
Checks, on a random sample of clips, that the v1 cache's 27 slots are bit-identical to
the V3 indices the v1 converter used, and states what those indices really are under the
documented V3 order [RH 0-20 | LH 21-41 | Pose 42-74 | Face 75-552]."""
import os, sys, json
import numpy as np
V3 = sys.argv[1] if len(sys.argv) > 1 else "data/raw_wlasl_v3/landmarks_V3.npz"
OLD = "data/wlasl_pose_cache"
V1_IDX = [0, 12, 11, 14, 13, 16, 15] + [522 + i for i in [0,4,5,8,9,12,13,16,17,20]] + [501 + i for i in [0,4,5,8,9,12,13,16,17,20]]
def name(i):
    if i < 21: return f"RH{i}"
    if i < 42: return f"LH{i-21}"
    if i < 75: return f"POSE{i-42}"
    return f"FACE{i-75}"
b = np.load(V3)
keys = list(b.keys())
old = sorted(f[:-4] for f in os.listdir(OLD) if f.endswith(".npz"))
print("V3 keys", len(keys), "v1 cache files", len(old), "same set:", set(keys) == set(old))
rng = np.random.default_rng(0); n_eq = n = 0; tmis = 0
for k in rng.choice(keys, 300, replace=False):
    with np.load(os.path.join(OLD, k + ".npz")) as h: v1 = h["data"]
    v3 = b[k]
    if v1.shape[1] != v3.shape[0]: tmis += 1; continue
    n += 1
    n_eq += int(np.allclose(v1[0, :, :, 0], v3[:, V1_IDX, 0], atol=1e-6) and np.allclose(v1[1, :, :, 0], v3[:, V1_IDX, 1], atol=1e-6))
print(f"v1 slots == V3[V1_IDX] on {n_eq}/{n} clips (T mismatch {tmis})")
print("v1 slot -> true landmark:", json.dumps({s: name(i) for s, i in enumerate(V1_IDX)}))
