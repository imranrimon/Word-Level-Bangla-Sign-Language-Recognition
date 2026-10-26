#!/usr/bin/env python
"""P1 reviewer #23 -- decompose the pose-motion distance by ARTICULATOR (body vs
hands) and ask which body parts carry the transfer-predictive signal.

Our 27-node skeleton = 7 body (idx 0-6) + 20 hand (idx 7-26; 10/hand). It has NO
face, a stated limitation. The clip descriptor (tools/compute_lang_distance) is
per-joint velocity mean+std, laid out [mean_0..mean_{V-1}, std_0..std_{V-1}], so we
slice joint indices to get body-only / hands-only sub-descriptors and recompute the
RBF-MMD(source-pool, target) for each. We then correlate each with the @10% gain to
see which articulator's motion-distribution distance best predicts alignment gain.

CPU-only. Run in bdsl_graph.  python tools/articulator_distance.py
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from tools.compute_lang_distance import mmd_rbf, sample_npy, sample_npz_dir

V = 27
BODY = list(range(0, 7))          # 7 body keypoints
HANDS = list(range(7, 27))        # 20 hand keypoints (10/hand)

# @10% mechanism gain (xling - pool), from tab:law / gating_law_fit
GAIN = {"BdSL": 1.70, "LSA64": 1.00, "KArSL": -4.40, "AUTSL": -6.50, "GSL": 2.06, "LSFB": -5.06,
        "LIBRAS": -4.68}


def sub(desc, idx):
    """Slice a (2V,) per-joint [mean|std] descriptor to the given joint indices."""
    idx = np.array(idx)
    return np.concatenate([desc[:, idx], desc[:, V + idx]], axis=1)


def main():
    rng = np.random.default_rng(0); N = 400
    src = sample_npz_dir('data/wlasl_pose_cache', N, rng)
    if os.path.exists('data/bdsl_si/pretrain_data.npy'):
        src += sample_npy('data/bdsl_si/pretrain_data.npy', N, rng)
    src = np.array(src)
    targets = {"BdSL": 'data/bdsl_si/train_data.npy', "LSA64": 'data/lsa64_si/train_data.npy',
               "AUTSL": 'data/autsl_si/train_data.npy', "KArSL": 'data/karsl_si/train_data.npy',
               "GSL": 'data/gsl_si/train_data.npy', "LSFB": 'data/lsfb_si/train_data.npy',
               "LIBRAS": 'data/libras_si/train_data.npy'}
    print(f"source pool {len(src)} clips, desc dim {src.shape[1]} (=2V, V={V})\n")
    print(f"{'target':7s} {'MMD_full':>9s} {'MMD_body':>9s} {'MMD_hands':>9s} {'gain@10%':>9s}")
    rows = []
    for name, path in targets.items():
        if not os.path.exists(path):
            print(f"  {name}: MISSING"); continue
        tgt = np.array(sample_npy(path, N, rng))
        df = mmd_rbf(src, tgt)
        db = mmd_rbf(sub(src, BODY), sub(tgt, BODY))
        dh = mmd_rbf(sub(src, HANDS), sub(tgt, HANDS))
        rows.append((name, df, db, dh, GAIN[name]))
        print(f"  {name:7s} {df:9.4f} {db:9.4f} {dh:9.4f} {GAIN[name]:9.2f}")

    def r2(x, y):
        x = np.array(x); y = np.array(y)
        A = np.vstack([x, np.ones_like(x)]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        p = A @ [a, b]
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    g = [r[4] for r in rows]
    from scipy.stats import spearmanr
    print("\n-- which articulator's distance predicts the @10% gain? --")
    for lbl, col in [("full 27", 1), ("body-only (7)", 2), ("hands-only (20)", 3)]:
        x = [r[col] for r in rows]
        print(f"  {lbl:16s}: R^2(gain~MMD)={r2(x, g):.3f}   Spearman={spearmanr(x, g).correlation:+.3f}")
    print("\n(If hands-only >= full, the manual channel carries the transfer signal;"
          "\n a weighted body-part distance D=a_h*D_h+a_b*D_b is the natural extension."
          "\n Face is absent from the 27-node skeleton -- a stated scope limit, #23.)")


if __name__ == "__main__":
    main()
