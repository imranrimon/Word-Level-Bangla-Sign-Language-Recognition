#!/usr/bin/env python3
"""Pairwise pose-motion MMD between NTU domains (cross-domain generality demo).
Same descriptor as tools/compute_lang_distance.py (per-joint frame-to-frame speed
mean+std), RBF-MMD with the median-heuristic bandwidth. Prints the pairwise matrix
so we can pick source->target pairs spanning a distance range."""
import numpy as np, os

DOMAINS = ["ntu_hrnet", "hmdb51", "ucf101", "gym"]
N = 400   # clips sampled per domain

def clip_desc(c):
    xy = np.asarray(c[:2, :, :, 0], dtype=np.float32)   # (2,T,V)
    e = np.abs(c).sum(axis=(0, 2, 3)); nz = np.nonzero(e)[0]
    T = int(nz[-1] + 1) if len(nz) else xy.shape[1]
    xy = xy[:, :max(T, 2)]
    # SCALE-NORMALIZE by per-clip pose bounding-box diagonal (pixel coords vary 5-6x
    # across in-the-wild datasets; raw velocity-MMD would track resolution, not motion).
    valid = xy != 0
    if valid.any():
        xs = xy[0][valid[0]]; ys = xy[1][valid[1]]
        diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) if xs.size and ys.size else 1.0
    else:
        diag = 1.0
    diag = diag if diag > 1e-6 else 1.0
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0) / diag   # (T-1,V), scale-invariant
    return np.concatenate([vel.mean(0), vel.std(0)])     # (2V,)

def sample(domain, rng):
    arr = np.load(f"data/hrnet_{domain}/train_data.npy", mmap_mode="r")
    idx = rng.permutation(arr.shape[0])[:N]
    return np.stack([clip_desc(np.asarray(arr[i])) for i in idx])

def mmd_rbf(X, Y):
    Z = np.vstack([X, Y]); d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sig = np.median(d2[d2 > 0]) + 1e-9; K = np.exp(-d2 / sig)
    n, m = len(X), len(Y); Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    return float((Kxx.sum() - np.trace(Kxx)) / (n*(n-1)) + (Kyy.sum() - np.trace(Kyy)) / (m*(m-1)) - 2*Kxy.mean())

def main():
    rng = np.random.RandomState(0)
    desc = {d: sample(d, rng) for d in DOMAINS if os.path.exists(f"data/hrnet_{d}/train_data.npy")}
    doms = list(desc)
    print("domains:", doms)
    print(f"{'':8}" + "".join(f"{d:>9}" for d in doms))
    M = {}
    for a in doms:
        row = ""
        for b in doms:
            v = 0.0 if a == b else mmd_rbf(desc[a], desc[b]); M[(a, b)] = v
            row += f"{v:9.4f}"
        print(f"{a:8}{row}")
    print("\nsource->target pairs sorted by distance:")
    for (a, b), v in sorted({k: val for k, val in M.items() if k[0] != k[1]}.items(), key=lambda kv: kv[1]):
        print(f"  {a:7} -> {b:7}  MMD={v:.4f}")

if __name__ == "__main__":
    main()
