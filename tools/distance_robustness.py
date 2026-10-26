"""P1 distance-robustness: is the pose-motion distance an FPS/acquisition artifact,
and is MMD special?  (Answers reviewer points on the source->target distance.)

Recomputes the source->target distance for the 5 P1 targets under many
DESCRIPTOR normalizations and DISTANCE metrics, and refits the gating law
    gain = a + b * distance
for each, reporting R^2, d* = -a/b, LOOCV MAE, and sign-agreement (k/5).

Two robustness axes:
  A. descriptor  (MMD fixed): position / velocity(current) / acceleration /
     velocity-bbox-normalized / velocity-FPS-normalized / velocity-zscored.
  B. metric      (velocity descriptor fixed): MMD / energy / Frechet / proxy-A /
     kNN two-sample.
Also prints a real-length (non-padded frame count) diagnostic per language --
if all corpora were resampled to a common T the raw FPS confound is already
largely removed; if real-T varies widely the FPS/duration axis is live.

CPU-only, no training. MUST run on a COMPUTE NODE (login-node scratch is down).

    python tools/distance_robustness.py            # full run
    python tools/distance_robustness.py --n 400     # sample size per pool
"""
import argparse
import glob
import os

import numpy as np

TARGETS = ["GSL", "BdSL", "LSA64", "KArSL", "AUTSL"]
PATHS = {
    "GSL": "data/gsl_si/train_data.npy",
    "BdSL": "data/bdsl_si/train_data.npy",
    "LSA64": "data/lsa64_si/train_data.npy",
    "KArSL": "data/karsl_si/train_data.npy",
    "AUTSL": "data/autsl_si/train_data.npy",
}
# xling - pool @10% val Top-1 gains, SAME order as TARGETS (from tools/gating_law_fit.py)
GAINS = np.array([2.06, 1.70, 1.00, -4.40, -6.50])
# ASSUMED native capture rates (fps) -- flagged, used only by the *_fps variant.
FPS = {"GSL": 30.0, "BdSL": 30.0, "LSA64": 60.0, "KArSL": 30.0, "AUTSL": 30.0}


# ----------------------------- descriptors ---------------------------------
def _exists(p):
    """os.path.exists, but an I/O error (e.g. Lustre OST offline: stat -> ESHUTDOWN) raises instead of reading as 'missing'."""
    try:
        os.stat(p)
        return True
    except FileNotFoundError:
        return False


def _xy(a):
    a = np.asarray(a, np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                       # (2, T, V)
    nz = np.abs(xy).sum((0, 2)) > 0           # drop zero-padded frames
    xy = xy[:, nz, :] if nz.any() else xy
    return xy                                 # (2, Tr, V)


def real_len(a):
    return _xy(a).shape[1]


def descriptor(a, mode, fps):
    xy = _xy(a)                               # (2, Tr, V)
    if xy.shape[1] < 3:
        return None
    if mode == "position":
        return np.concatenate([xy.mean(1).ravel(), xy.std(1).ravel()])         # 4V
    # signed frame-to-frame delta then speed magnitude
    d = np.diff(xy, axis=1)                    # (2, Tr-1, V)
    spd = np.linalg.norm(d, axis=0)           # (Tr-1, V)
    if mode == "velocity":
        return np.concatenate([spd.mean(0), spd.std(0)])                        # 2V
    if mode == "velocity_fps":
        spd = spd * fps                        # per-second speed
        return np.concatenate([spd.mean(0), spd.std(0)])                        # 2V
    if mode == "velocity_bbox":
        xmin, xmax = xy[0].min(), xy[0].max()
        ymin, ymax = xy[1].min(), xy[1].max()
        diag = np.hypot(xmax - xmin, ymax - ymin) + 1e-6
        spd = spd / diag                       # scale-invariant
        return np.concatenate([spd.mean(0), spd.std(0)])                        # 2V
    if mode == "acceleration":
        acc = np.abs(np.diff(spd, axis=0))     # (Tr-2, V)
        return np.concatenate([acc.mean(0), acc.std(0)])                        # 2V
    raise ValueError(mode)


# ------------------------------- samplers ----------------------------------
def sample_npy(path, mode, fps, n, rng):
    a = np.load(path, mmap_mode="r")
    idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
    out = [descriptor(a[i], mode, fps) for i in idx]
    return [d for d in out if d is not None]


def sample_npz_dir(root, mode, fps, n, rng):
    fs = glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
    rng.shuffle(fs)
    out = []
    for f in fs[: n * 3]:
        try:
            with np.load(f) as h:
                d = descriptor(h["data"], mode, fps)
            if d is not None:
                out.append(d)
        except OSError:  # unreadable storage (e.g. Lustre OST offline on the login node) must not silently swap the clip sample
            raise
        except Exception:
            pass
        if len(out) >= n:
            break
    return out


# ------------------------------- metrics -----------------------------------
def mmd_rbf(X, Y):
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    s2 = np.median(d2[d2 > 0]) + 1e-9
    K = np.exp(-d2 / s2)
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
            - 2 * Kxy.mean())


def energy_dist(X, Y):
    def pd(A, B):
        return np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0)).mean()
    return 2 * pd(X, Y) - pd(X, X) - pd(Y, Y)


def frechet(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Cx, Cy = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    diff = ((mx - my) ** 2).sum()
    # sqrt(Cx) via eigh, then eig of sqrt(Cx) Cy sqrt(Cx)
    wx, Vx = np.linalg.eigh((Cx + Cx.T) / 2)
    wx = np.clip(wx, 0, None)
    Cxs = (Vx * np.sqrt(wx)) @ Vx.T
    M = Cxs @ Cy @ Cxs
    wm = np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)
    return float(diff + np.trace(Cx) + np.trace(Cy) - 2 * np.sqrt(wm).sum())


def proxy_a(X, Y, rng):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        return None
    Z = np.vstack([X, Y])
    y = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    clf = LogisticRegression(max_iter=500)
    err = 1 - cross_val_score(clf, Zs, y, cv=5).mean()
    return float(2 * (1 - 2 * err))           # Proxy-A distance


def knn_stat(X, Y, k=5):
    Z = np.vstack([X, Y])
    lab = np.concatenate([np.zeros(len(X)), np.ones(len(Y))]).astype(int)
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    same = (lab[nn] == lab[:, None]).mean()
    return float(same)                        # 0.5 = indistinguishable, 1 = separable


# ------------------------------- law refit ---------------------------------
def fit_law(dist, gains):
    b, a = np.polyfit(dist, gains, 1)
    pred = a + b * dist
    ss = ((gains - pred) ** 2).sum()
    tot = ((gains - gains.mean()) ** 2).sum()
    r2 = 1 - ss / tot if tot > 0 else float("nan")
    dstar = -a / b if b != 0 else float("nan")
    # LOOCV
    errs, sgn = [], 0
    for i in range(len(dist)):
        m = np.ones(len(dist), bool)
        m[i] = False
        bb, aa = np.polyfit(dist[m], gains[m], 1)
        p = aa + bb * dist[i]
        errs.append(abs(p - gains[i]))
        sgn += int(np.sign(p) == np.sign(gains[i]))
    return r2, dstar, float(np.mean(errs)), sgn


def build_src(mode, fps_default, n, rng):
    src = sample_npz_dir("data/wlasl_pose_cache", mode, fps_default, n, rng)
    if _exists("data/bdsl_si/pretrain_data.npy"):
        src += sample_npy("data/bdsl_si/pretrain_data.npy", mode, FPS["BdSL"], n, rng)
    return np.array(src)


def distances_for(mode, metric, n, rng, zscore=False):
    src = build_src(mode, 30.0, n, rng)
    ds = []
    for t in TARGETS:
        if not os.path.exists(PATHS[t]):
            return None, f"MISSING {t}"
        tgt = np.array(sample_npy(PATHS[t], mode, FPS[t], n, rng))
        X, Y = src, tgt
        if zscore:
            Z = np.vstack([X, Y])
            mu, sd = Z.mean(0), Z.std(0) + 1e-8
            X, Y = (X - mu) / sd, (Y - mu) / sd
        if metric == "mmd":
            d = mmd_rbf(X, Y)
        elif metric == "energy":
            d = energy_dist(X, Y)
        elif metric == "frechet":
            d = frechet(X, Y)
        elif metric == "proxy_a":
            d = proxy_a(X, Y, rng)
        elif metric == "knn":
            d = knn_stat(X, Y)
        else:
            raise ValueError(metric)
        if d is None:
            return None, f"metric {metric} unavailable"
        ds.append(d)
    return np.array(ds, float), None


def report(tag, dist, err=None):
    if err:
        print(f"  {tag:26s} -> {err}")
        return
    r2, dstar, mae, sgn = fit_law(dist, GAINS)
    ds = " ".join(f"{d:7.4f}" for d in dist)
    print(f"  {tag:26s} [{ds}]  R2={r2:6.3f}  d*={dstar:7.4f}  LOOCV_MAE={mae:5.2f}  sign={sgn}/5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 96)
    print("P1 DISTANCE ROBUSTNESS  (targets order: %s)" % ", ".join(TARGETS))
    print("gains (xling-pool @10%%):", list(GAINS))
    print("ASSUMED native FPS:", FPS, " (only used by *_fps variant; flagged)")
    print("=" * 96)

    # real-length diagnostic (is the FPS confound already resampled away?)
    print("\n[real-length diagnostic] non-padded frame count per clip:")
    for t in TARGETS:
        if not os.path.exists(PATHS[t]):
            print(f"  {t:6s} MISSING {PATHS[t]}"); continue
        a = np.load(PATHS[t], mmap_mode="r")
        idx = rng.choice(len(a), size=min(120, len(a)), replace=False)
        L = np.array([real_len(a[i]) for i in idx])
        print(f"  {t:6s} N={len(a):6d}  realT: min={L.min():3d} med={int(np.median(L)):3d} "
              f"max={L.max():3d}  (fixed-T if min==max)")

    print("\n[AXIS A] descriptor robustness  (metric = MMD):")
    for mode in ["position", "velocity", "acceleration", "velocity_bbox", "velocity_fps"]:
        d, e = distances_for(mode, "mmd", args.n, np.random.default_rng(0))
        report(f"MMD / {mode}", d, e)
    d, e = distances_for("velocity", "mmd", args.n, np.random.default_rng(0), zscore=True)
    report("MMD / velocity_zscore", d, e)

    print("\n[AXIS B] metric robustness  (descriptor = velocity, current):")
    for metric in ["mmd", "energy", "frechet", "proxy_a", "knn"]:
        d, e = distances_for("velocity", metric, args.n, np.random.default_rng(0))
        report(f"{metric} / velocity", d, e)

    print("\n[interpretation]")
    print("  - law ROBUST if R2 stays high and sign=5/5 across AXIS-A normalizations")
    print("    (esp. velocity_bbox and velocity_zscore, which remove global scale);")
    print("  - MMD NOT uniquely required if AXIS-B alt-metrics also give high R2 / 5-5;")
    print("  - FPS confound already muted if real-T is ~constant across corpora (resampled).")
    print("=" * 96)


if __name__ == "__main__":
    main()
