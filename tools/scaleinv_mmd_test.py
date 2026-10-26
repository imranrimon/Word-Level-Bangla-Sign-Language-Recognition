#!/usr/bin/env python
"""P1 reviewer #24 -- can a SCALE-INVARIANT pose-motion descriptor rescue the
distance's robustness to the pose estimator?

Diagnosis (an RTMPose-vs-MediaPipe cross-check of the descriptor): the raw descriptor
    d(clip) = [ mean_v speed_v , std_v speed_v ]           (speed_v = |p_{t+1}-p_t|_v)
inflates MMD(pool, BdSL) from 0.143 (MediaPipe) to 0.288 (RTMPose, +101%) because
RTMPose reports ~0.4x the per-joint velocity of MediaPipe (a near-global multiplier),
and the source pool is 100% MediaPipe. A descriptor invariant to a global speed
multiplier c (speed -> c*speed) should cancel that estimator factor.

Two scale-invariant variants (both computed per-clip, so invariance holds regardless
of bundle composition):
  V1 "unit-mean-speed":  d / (mean over v,t of speed)      -> mean_v & std_v both /s
  V2 "coeff-of-variation": [ std_v/mean_v ]                -> dimensionless burstiness

We report, for each descriptor:
  (A) ROBUSTNESS  -- MMD(pool, BdSL-MediaPipe) vs MMD(pool, BdSL-RTMPose); the gap
      must collapse for the estimator-robustness claim to hold.
  (B) VALIDITY    -- MMD(pool, target) for all 7 languages, and R^2 of the @10%
      mechanism gain (xling-pool) regressed on MMD. The predictor is only useful if
      the ordering and the gain-fit survive the scale normalization.

CPU-only. Run in bdsl_graph.   python tools/scaleinv_mmd_test.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from tools.compute_lang_distance import mmd_rbf, _exists   # reuse the exact RBF-MMD

EPS = 1e-8


def descriptor(arr, mode="raw"):
    """arr (C,T,V,M) -> motion descriptor under the chosen scale mode, or None."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                                  # (2,T,V)
    nz = np.abs(xy).sum((0, 2)) > 0
    xy = xy[:, nz, :] if nz.any() else xy
    if xy.shape[1] < 2:
        return None
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0)    # (T-1, V) per-joint speed
    mean_v = vel.mean(0); std_v = vel.std(0)             # (V,), (V,)
    if mode == "raw":
        return np.concatenate([mean_v, std_v])
    if mode == "v1":                                     # unit global-mean-speed
        s = vel.mean()
        if s < EPS:
            return None
        return np.concatenate([mean_v, std_v]) / s
    if mode == "v2":                                     # per-joint coeff of variation
        return std_v / (mean_v + EPS)
    raise ValueError(mode)


def sample_npy(path, n, rng, mode):
    a = np.load(path, mmap_mode='r')
    idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
    out = [descriptor(a[i], mode) for i in idx]
    return [d for d in out if d is not None]


def sample_npz_dir(root, n, rng, mode):
    fs = glob.glob(os.path.join(root, '**', '*.npz'), recursive=True)
    rng.shuffle(fs)
    out = []
    for f in fs[:n * 2]:
        try:
            with np.load(f) as h:
                d = descriptor(h['data'], mode)
            if d is not None:
                out.append(d)
        except OSError:  # unreadable storage (e.g. Lustre OST offline on the login node) must not silently swap the clip sample
            raise
        except Exception:
            pass
        if len(out) >= n:
            break
    return out


def build_pool(mode, N=400):
    rng = np.random.default_rng(0)
    src = sample_npz_dir('data/wlasl_pose_cache', N, rng, mode)
    if _exists('data/bdsl_si/pretrain_data.npy'):
        src += sample_npy('data/bdsl_si/pretrain_data.npy', N, rng, mode)
    return np.array(src)


def r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    p = A @ [a, b]
    ss = ((y - y.mean()) ** 2).sum()
    return (1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float('nan'), a, b


# @10% mechanism gain (xling - pool), from tab:law / articulator_distance.py
GAIN = {"BdSL": 1.70, "LSA64": 1.00, "KArSL": -4.40, "AUTSL": -6.50, "GSL": 2.06, "LSFB": -5.06}
TARGETS = {"BdSL": 'data/bdsl_si/train_data.npy', "LSA64": 'data/lsa64_si/train_data.npy',
           "AUTSL": 'data/autsl_si/train_data.npy', "KArSL": 'data/karsl_si/train_data.npy',
           "GSL": 'data/gsl_si/train_data.npy', "BISINDO": 'data/bisindo_si/train_data.npy',
           "LSFB": 'data/lsfb_si/train_data.npy'}


def run(mode):
    print(f"\n================= descriptor mode = {mode.upper()} =================")
    pool = build_pool(mode)
    print(f"source pool {len(pool)} clips, dim {pool.shape[1]}")

    # (A) ROBUSTNESS: MediaPipe vs RTMPose BdSL
    rng = np.random.default_rng(1)
    mp = np.array(sample_npy('data/bdsl_si/train_data.npy', 400, rng, mode))
    d_mp = mmd_rbf(pool, mp)
    if os.path.exists('data/bdsl_si_rtm/train_data.npy'):
        rng = np.random.default_rng(1)
        rtm = np.array(sample_npy('data/bdsl_si_rtm/train_data.npy', 400, rng, mode))
        d_rtm = mmd_rbf(pool, rtm)
        spread = 0.48 - 0.12
        print(f"(A) ROBUSTNESS  MMD(pool,BdSL-MediaPipe)={d_mp:.4f}  "
              f"MMD(pool,BdSL-RTMPose)={d_rtm:.4f}  |Δ|={abs(d_rtm-d_mp):.4f}  "
              f"({abs(d_rtm-d_mp)/max(d_mp,EPS)*100:+.1f}% ; {abs(d_rtm-d_mp)/spread*100:.1f}% of cross-lang spread)")
    else:
        print("(A) RTMPose bundle missing -> skip")

    # (B) VALIDITY: all-language MMD + gain fit
    rng = np.random.default_rng(0)
    rows = []
    for name, path in TARGETS.items():
        if not os.path.exists(path):
            print(f"    {name}: MISSING"); continue
        tgt = np.array(sample_npy(path, 400, rng, mode))
        rows.append((name, mmd_rbf(pool, tgt)))
    rows.sort(key=lambda r: r[1])
    print("(B) VALIDITY  MMD by increasing distance:")
    for n, d in rows:
        g = GAIN.get(n, float('nan'))
        print(f"      {n:8s} MMD={d:.4f}   gain@10%={g:+.2f}" if g == g else f"      {n:8s} MMD={d:.4f}   gain@10%= n/a")
    fit = [(n, d, GAIN[n]) for n, d in rows if n in GAIN]
    xs = [f[1] for f in fit]; gs = [f[2] for f in fit]
    R2, a, b = r2(xs, gs)
    print(f"    gain ~ MMD over {len(fit)} langs: R^2={R2:.3f}  slope={a:.2f}  intercept={b:.2f}")
    return rows


def main():
    for mode in ["raw", "v1", "v2"]:
        run(mode)
    print("\nVERDICT: the estimator-robustness claim is SAVED iff a scale-invariant mode "
          "(v1/v2) both (A) collapses the MediaPipe-vs-RTMPose |Δ| toward the sampling floor "
          "AND (B) keeps a comparable gain~MMD R^2. Compare against RAW above.")


if __name__ == "__main__":
    main()
