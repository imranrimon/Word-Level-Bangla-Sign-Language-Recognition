"""Source-target pose distance for the distance-aware-lambda law (P1).

Turns the oracle observation ("distant targets want smaller lambda") into a predictive
signal: computes a pose-motion MMD distance from the pretraining SOURCE pool (ASL/WLASL
+ BdSL) to each TARGET language, then correlates distance with the empirically-optimal
lambda. If distance predicts lambda, we can propose lambda = f(distance).

Per-clip descriptor: per-joint mean & std of frame-to-frame speed (channels x,y),
i.e. a motion signature independent of absolute position. MMD with an RBF kernel
(median-heuristic bandwidth) between a sample of source clips and each target.

NOTE (sampling / draw dependence). The n=400 estimate carries a per-language
sampling SD of ~0.016 (measured over 8 independent draws; 0.005-0.029, larger for
the more dispersed distant corpora). Because `sample_npy`/`sample_npz_dir` advance a
SINGLE shared rng, a target's MMD also depends on the draw ORDER, so tools that
sample in a different order (e.g. articulator_distance.py) get values ~0.01 apart,
and the same estimator can read R^2=0.997 vs 0.989 on different draws. This is
< the 0.16 close/far cluster gap, so the align-vs-pool DECISION is draw-robust even
though the exact slope/d* wobble at the ~0.01 level. The canonical paper registry is
ONE fixed one-pass draw (values hardcoded in tools/gating_law_fit.py); to reduce the
wobble, average the MMD over several draws or raise n.
"""
import glob, os
import numpy as np


def _exists(p):
    """os.path.exists, but an I/O error (e.g. Lustre OST offline: stat -> ESHUTDOWN) raises instead of reading as 'missing'."""
    try:
        os.stat(p)
        return True
    except FileNotFoundError:
        return False


def clip_descriptor(arr):
    # arr: (C,T,V,M) -> use x,y ; speed_t = |p_{t+1}-p_t|
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                      # (2,T,V)
    # drop all-zero (padding) frames
    nz = np.abs(xy).sum((0, 2)) > 0
    xy = xy[:, nz, :] if nz.any() else xy
    if xy.shape[1] < 2:
        return None
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0)   # (T-1, V)
    return np.concatenate([vel.mean(0), vel.std(0)])     # (2V,)


def sample_npy(path, n, rng):
    a = np.load(path, mmap_mode='r')
    idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
    out = [clip_descriptor(a[i]) for i in idx]
    return [d for d in out if d is not None]


def sample_npz_dir(root, n, rng):
    fs = glob.glob(os.path.join(root, '**', '*.npz'), recursive=True)
    rng.shuffle(fs)
    out = []
    for f in fs[:n * 2]:
        try:
            with np.load(f) as h:
                d = clip_descriptor(h['data'])
            if d is not None:
                out.append(d)
        except OSError:  # unreadable storage (e.g. Lustre OST offline on the login node) must not silently swap the clip sample
            raise
        except Exception:
            pass
        if len(out) >= n:
            break
    return out


def mmd_rbf(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sigma2 = np.median(d2[d2 > 0]) + 1e-9
    K = np.exp(-d2 / sigma2)
    n, m = len(X), len(Y)
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    return (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
         - 2 * Kxy.mean()


def main():
    rng = np.random.default_rng(0)
    N = 400
    # SOURCE pool = ASL(WLASL) + BdSL pretrain-pool signers
    src = sample_npz_dir('data/wlasl_pose_cache', N, rng)
    bdsl_pool = 'data/bdsl_si/pretrain_data.npy'
    if _exists(bdsl_pool):
        src += sample_npy(bdsl_pool, N, rng)
    src = np.array(src)
    targets = {'BdSL': 'data/bdsl_si/train_data.npy',
               'LSA64': 'data/lsa64_si/train_data.npy',
               'AUTSL': 'data/autsl_si/train_data.npy',
               'KArSL': 'data/karsl_si/train_data.npy',   # Arabic, distant (gap-4)
               'GSL': 'data/gsl_si/train_data.npy',  # Greek SL isol., 5th target (P1 #4; INCLUDE dropped - no signer labels)
               'BISINDO': 'data/bisindo_si/train_data.npy',  # WL-BISINDO (Indonesian SL, Banten), 6th target (P1 #4)
               'LSFB': 'data/lsfb_si/train_data.npy',  # LSFB-ISOL (Belgian-French SL), 6th language (top-60 signs, 99 signers)
               'LIBRAS': 'data/libras_si/train_data.npy'}  # MINDS-Libras (Brazilian SL), gap-filling target for empty [0.18,0.34] (ZIP64 fixed)
    # empirical optimal lambda (Table 4); KArSL filled once its fine-tunes land
    opt_lambda = {'BdSL': 0.1, 'LSA64': 0.1, 'AUTSL': 0.01, 'KArSL': float('nan'),
                  'GSL': float('nan'), 'BISINDO': float('nan'), 'LSFB': float('nan'),
                  'LIBRAS': float('nan')}
    print(f"source pool sample: {len(src)} clips  (desc dim {src.shape[1]})")
    print(f"{'target':8s} {'MMD(pool,target)':>18s} {'optimal_lambda':>15s}")
    rows = []
    for name, path in targets.items():
        if not os.path.exists(path):
            print(f"  {name}: MISSING"); continue
        tgt = np.array(sample_npy(path, N, rng))
        d = mmd_rbf(src, tgt)
        rows.append((name, d, opt_lambda[name]))
        print(f"  {name:8s} {d:18.4f} {opt_lambda[name]:15.3f}")
    # monotonicity check: larger distance -> smaller optimal lambda?
    rows.sort(key=lambda r: r[1])
    print(f"\n  by increasing distance: {[r[0] for r in rows]}")
    known = [(r[0], r[1], r[2]) for r in rows if r[2] == r[2]]   # drop NaN lambdas
    lams = [r[2] for r in known]
    print(f"  (known-lambda) order: {[r[0] for r in known]}  lambdas: {lams}")
    print("  PREDICTIVE LAW holds if lambda is (weakly) DECREASING as distance grows.")
    ok = all(lams[i] >= lams[i+1] for i in range(len(lams)-1))
    print(f"  -> monotone decreasing? {ok}")
    nanrows = [r for r in rows if r[2] != r[2]]
    for r in nanrows:
        print(f"  {r[0]}: distance {r[1]:.4f} -> PREDICTED lambda "
              f"({'small/0.01 (distant)' if r[1] > (known[len(known)//2][1] if known else 0) else 'large/0.1 (close)'}); "
              f"confirm against its fine-tune result.")


if __name__ == "__main__":
    main()
