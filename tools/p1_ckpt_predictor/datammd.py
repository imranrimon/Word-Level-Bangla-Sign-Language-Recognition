#!/usr/bin/env python
"""Data-only pose-motion MMD (pool -> target), recomputed against the REAL pretraining pool.

The paper's canonical one-pass values (tools/gating_law_fit.py DIST) were computed against 400 WLASL +
400 BdSLW60-pretrain clips, not the pool the backbones were trained on. Here the source sample is drawn
uniformly (hence proportionally: BdSLW401 51,098 / WLASL 21,083 / BdSLW102a 3,408) from
data/ssl_pool_manifest_bdsl_asl.json, and targets from their TRAIN split only. Two descriptors:

  * "paper":  tools/compute_lang_distance.clip_descriptor (per-joint mean/std of frame-to-frame speed,
              all-zero frames dropped) -- MediaPipe hand dropout (joint -> 0) shows up as huge jumps;
  * "masked": same, but a speed sample is used only where the joint is detected (non-zero) in BOTH
              frames (dropout-masked); joints never detected contribute 0.

RBF-MMD with median-heuristic bandwidth (compute_lang_distance.mmd_rbf), n=600 per side, averaged over
5 independent draws (draw SD reported). Writes <out>/data_mmd.json.
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.compute_lang_distance import clip_descriptor, mmd_rbf  # noqa: E402

TGT = {"BdSL": "data/bdsl_si/train_data.npy", "LSA64": "data/lsa64_si/train_data.npy",
       "KArSL": "data/karsl_si/train_data.npy", "AUTSL": "data/autsl_si/train_data.npy",
       "GSL": "data/gsl_si/train_data.npy", "LSFB": "data/lsfb_si/train_data.npy",
       "LIBRAS": "data/libras_si/train_data.npy"}


def masked_descriptor(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                                   # (2,T,V)
    nzf = np.abs(xy).sum((0, 2)) > 0
    xy = xy[:, nzf, :] if nzf.any() else xy
    if xy.shape[1] < 2:
        return None
    det = np.abs(xy).sum(0) > 0                           # (T,V) joint detected
    ok = det[1:] & det[:-1]                               # (T-1,V)
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0)     # (T-1,V)
    n = ok.sum(0)
    mean = np.where(n > 0, (vel * ok).sum(0) / np.maximum(n, 1), 0.0)
    var = np.where(n > 0, ((vel - mean) ** 2 * ok).sum(0) / np.maximum(n, 1), 0.0)
    return np.concatenate([mean, np.sqrt(var)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "work_dir", "p1_ckpt_predictor"))
    a = ap.parse_args()
    fn = os.path.join(a.out, "data_mmd.json")
    if os.path.exists(fn):
        print("[datammd] exists", fn); return
    os.makedirs(a.out, exist_ok=True)
    clips = json.load(open(os.path.join(ROOT, "data/ssl_pool_manifest_bdsl_asl.json")))["clips"]
    tg = {k: np.load(os.path.join(ROOT, v), mmap_mode="r") for k, v in TGT.items()}
    res = {k: {"paper": [], "masked": []} for k in TGT}
    for d in range(a.draws):
        rng = np.random.default_rng(100 + d)
        src = {"paper": [], "masked": []}
        for j in rng.permutation(len(clips)):
            if len(src["paper"]) >= a.n:
                break
            with np.load(os.path.join(ROOT, clips[j]["path"])) as h:
                c = h["data"]
            p, m = clip_descriptor(c), masked_descriptor(c)
            if p is None or m is None:
                continue
            src["paper"].append(p); src["masked"].append(m)
        for k, arr in tg.items():
            idx = rng.choice(len(arr), size=min(a.n, len(arr)), replace=False)
            P, M = [], []
            for i in idx:
                c = np.asarray(arr[i]); p, m = clip_descriptor(c), masked_descriptor(c)
                if p is None or m is None:
                    continue
                P.append(p); M.append(m)
            res[k]["paper"].append(float(mmd_rbf(np.array(src["paper"]), np.array(P))))
            res[k]["masked"].append(float(mmd_rbf(np.array(src["masked"]), np.array(M))))
        print(f"[datammd] draw {d}: " + ", ".join(f"{k} {res[k]['paper'][-1]:.3f}/{res[k]['masked'][-1]:.3f}"
                                                    for k in TGT), flush=True)
    out = {k: {"realpool": float(np.mean(v["paper"])), "realpool_sd": float(np.std(v["paper"])),
               "realpool_masked": float(np.mean(v["masked"])), "realpool_masked_sd": float(np.std(v["masked"]))}
           for k, v in res.items()}
    json.dump(out, open(fn, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
