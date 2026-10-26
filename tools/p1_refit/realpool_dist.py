#!/usr/bin/env python
"""Sensitivity x-axis for the P1 law refit (NOT pre-registered; added 2026-10-22 at the
request of the distance audit). P1's canonical d values (tools/compute_lang_distance.py) measure the
source as 400 WLASL + 400 BdSLW60-pretrain clips, but the backbones were pretrained on
data/ssl_pool_manifest_bdsl_asl.json = BdSLW401-front 51,098 + WLASL 21,083 + BdSLW102a 3,408 (no
BdSLW60). Here the source is a uniform (=> proportional) 800-clip draw from that manifest, same
descriptor (clip_descriptor: per-joint speed mean/std), same median-bandwidth RBF-MMD, targets = 400
train clips each. Three pool draws (rng 1, 11, 12) are averaged; per-draw values are kept.
A second variant masks MediaPipe dropout (zero) keypoints before differencing.

  python tools/p1_refit/realpool_dist.py --out docs/p1_refit_2026-10/realpool_distances.json
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
from compute_lang_distance import clip_descriptor, mmd_rbf  # noqa: E402

TGT = {"BdSL": "data/bdsl_si/train_data.npy", "LSA64": "data/lsa64_si/train_data.npy",
       "KArSL": "data/karsl_si/train_data.npy", "AUTSL": "data/autsl_si/train_data.npy",
       "GSL": "data/gsl_si/train_data.npy", "LSFB": "data/lsfb_si/train_data.npy",
       "LIBRAS": "data/libras_si/train_data.npy"}


def desc_masked(arr):
    a = np.asarray(arr, np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]; nz = np.abs(xy).sum((0, 2)) > 0; xy = xy[:, nz, :]
    if xy.shape[1] < 2:
        return None
    valid = np.abs(xy).sum(0) > 0
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0); ok = valid[1:] & valid[:-1]
    m = np.zeros(vel.shape[1]); s = np.zeros(vel.shape[1])
    for v in range(vel.shape[1]):
        x = vel[ok[:, v], v]
        if len(x):
            m[v], s[v] = x.mean(), x.std()
    return np.concatenate([m, s])


def pool_clips(paths, n, rng):
    idx = rng.permutation(len(paths)); out = []
    for i in idx:
        with np.load(paths[i]) as h:
            d = np.array(h["data"])
        if clip_descriptor(d) is not None:
            out.append(d)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/p1_refit_2026-10/realpool_distances.json")
    ap.add_argument("--n-pool", type=int, default=800)
    ap.add_argument("--n-tgt", type=int, default=400)
    a = ap.parse_args()
    man = json.load(open("data/ssl_pool_manifest_bdsl_asl.json"))
    paths = [e["path"] if isinstance(e, dict) else e for e in man["clips"]] if "clips" in man else None
    if paths is None:
        k = [k for k, v in man.items() if isinstance(v, list)][0]
        paths = [e["path"] if isinstance(e, dict) else e for e in man[k]]
    tg = {}
    for t, p in TGT.items():
        arr = np.load(p, mmap_mode="r"); r = np.random.default_rng(0)
        idx = r.choice(len(arr), size=min(a.n_tgt, len(arr)), replace=False)
        tg[t] = [np.array(arr[i]) for i in idx]
    res = {"per_draw": {}, "source_counts": {}}
    for fn_name, fn in (("descriptor", clip_descriptor), ("masked", desc_masked)):
        TD = {t: np.array([v for v in (fn(x) for x in c) if v is not None]) for t, c in tg.items()}
        draws = []
        for seed in (1, 11, 12):
            clips = pool_clips(paths, a.n_pool, np.random.default_rng(seed))
            S = np.array([v for v in (fn(x) for x in clips) if v is not None])
            d = {t: float(mmd_rbf(S, TD[t])) for t in TGT}
            res["per_draw"][f"{fn_name}_rng{seed}"] = d; draws.append(d)
            print(fn_name, seed, {t: round(v, 3) for t, v in d.items()}, flush=True)
        res[f"realpool_{fn_name}"] = {t: float(np.mean([d[t] for d in draws])) for t in TGT}
    res["variants"] = {"realpool_descriptor": res["realpool_descriptor"], "realpool_masked": res["realpool_masked"]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["variants"], indent=1))


if __name__ == "__main__":
    main()
