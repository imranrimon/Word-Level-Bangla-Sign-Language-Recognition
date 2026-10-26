#!/usr/bin/env python
"""Pool -> target pose-motion distance, v2 (P1 fix program 2026-10).

Fixes the four defects of tools/compute_lang_distance.py (v1):
  1. SOURCE. v1 compared targets with 400 WLASL + 400 BdSLW60-pretrain clips, but the backbones
     were pretrained on data/ssl_pool_manifest_bdsl_asl.json (BdSLW401-front 51,098 + WLASL 21,083
     + BdSLW102a 3,408; no BdSLW60). v2 draws a source sample PROPORTIONAL to the per-source counts
     of the actual pretraining manifest(s) (`--source name=manifest.json`, one per arm / pool).
  2. DROPOUT. MediaPipe stores an undetected joint as (0,0); v1 differenced through those zeros, so
     every detect/lose event injected a ~0.5-1.0 jump (normal per-frame speed ~0.01) and dropout
     dominated the descriptor for most corpora. v2 only differences between samples where the
     joint is detected at both ends (or, `interp` variant, after bridging interior gaps <= 0.2 s).
  3. FRAME RATE. v1 speeds are per FRAME, and native rates differ (25-60 fps). v2 resamples each
     joint trajectory onto a common 25 Hz grid (linear interpolation inside detected runs only) and
     reports speed per SECOND. Native rates come from a documented registry
     (docs/p1_fix_2026-10/step2/fps_registry.json) verified in step 2.
  4. SCALE. v2 divides coordinates by the clip's median shoulder width over frames where both
     shoulders are detected (missing joints never enter the scale).
Statistic: the UNBIASED squared MMD (U-statistic, diagonal terms excluded)
    MMD^2_u = 1/(n(n-1)) sum_{i!=j} k(x_i,x_j) + 1/(m(m-1)) sum_{i!=j} k(y_i,y_j) - 2/(nm) sum k(x_i,y_j)
with an RBF kernel k = exp(-|a-b|^2 / sigma^2). (v1 computed the same U-statistic; the paper calls it
"MMD" -- it is MMD^2_u and can be slightly negative.) Bandwidth: `pooled` (v2 default) = median
pairwise squared distance of the union of the source sample and ALL target samples in that draw, so
every target is measured with the same kernel; `pair` = v1's per-pair median heuristic.
Uncertainty: R independent draws (fresh source and target subsamples), mean and SD reported.

Descriptors (per clip, 27 joints x {mean, std} of speed = 54-d):
  old           v1 clip_descriptor (raw per-frame speed through dropouts; all-zero frames dropped)
  masked_frame  dropout fix only (per-frame speed, detected-both-ends pairs, raw coordinates)
  masked_sec    + per-second (x native fps), + shoulder-width scale
  v2            masked + 25 Hz resampling + per-second + shoulder scale          <- PRIMARY (prereg)
  v2_interp     v2 after bridging interior detection gaps <= 0.2 s by linear interpolation
Joints never validly tracked in a clip get speed 0 (not visible -> no motion evidence).

  python tools/compute_lang_distance_v2.py --out docs/p1_fix_2026-10/step2/distances_v2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.compute_lang_distance import clip_descriptor as old_descriptor  # noqa: E402

TARGETS = {"BdSL": "data/bdsl_si/train_data.npy", "LSA64": "data/lsa64_si/train_data.npy",
           "AUTSL": "data/autsl_si/train_data.npy", "KArSL": "data/karsl_si/train_data.npy",
           "GSL": "data/gsl_si/train_data.npy", "BISINDO": "data/bisindo_si/train_data.npy",
           "LSFB": "data/lsfb_si/train_data.npy", "LIBRAS": "data/libras_si/train_data.npy",
           "ASL-Citizen-60": "data/aslcitizen60_si/train_data.npy"}
SOURCES = {"pool_v2": "data/ssl_pool_manifest_bdsl_asl_v2.json",
           "pool_v1": "data/ssl_pool_manifest_bdsl_asl.json",
           # v1 tool's source: 400 WLASL + 400 BdSLW60 pretrain split (with the old and the fixed WLASL)
           "oldsrc_wlasl_v1": {"data/wlasl_pose_cache": 0.5, "data/bdsl_si/pretrain_data.npy": 0.5},
           "oldsrc_wlasl_v2": {"data/wlasl_pose_cache_v2": 0.5, "data/bdsl_si/pretrain_data.npy": 0.5}}
DESCS = ["old", "masked_frame", "masked_sec", "v2", "v2_interp"]
RATE = 25.0
BRIDGE_S = 0.2


# ---------------------------------------------------------------- descriptors
def _prep(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]                                   # (2,T,V)
    det = np.abs(xy).sum(0) > 0                           # (T,V)
    anyd = det.any(1)
    if not anyd.any():
        return None, None
    lo, hi = np.where(anyd)[0][[0, -1]]                   # trim leading/trailing empty frames (padding)
    return xy[:, lo:hi + 1].copy(), det[lo:hi + 1].copy()


def _shoulder_scale(xy, det):
    m = det[:, 1] & det[:, 2]
    if m.sum() == 0:
        return None
    s = float(np.median(np.linalg.norm(xy[:, m, 1] - xy[:, m, 2], axis=0)))
    return s if s > 0 else None


def _bridge(xy, det, max_gap):
    """Linearly interpolate interior undetected runs of length <= max_gap frames (per joint)."""
    T, V = det.shape
    for v in range(V):
        d = det[:, v]
        idx = np.where(d)[0]
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        for j in np.where((gaps > 1) & (gaps - 1 <= max_gap))[0]:
            a, b = idx[j], idx[j + 1]
            w = (np.arange(a + 1, b) - a) / (b - a)
            xy[:, a + 1:b, v] = xy[:, a:a + 1, v] * (1 - w) + xy[:, b:b + 1, v] * w
            det[a + 1:b, v] = True
    return xy, det


def _stats(sp, ok):
    V = sp.shape[1]
    m = np.zeros(V, np.float32); s = np.zeros(V, np.float32)
    for v in range(V):
        x = sp[ok[:, v], v]
        if x.size:
            m[v] = x.mean(); s[v] = x.std()
    return np.concatenate([m, s])


def _resampled_speeds(xy, det, fps, rate=RATE):
    """Speeds (per second) on a `rate` Hz grid; grid samples interpolated only inside detected runs."""
    T, V = det.shape
    if T < 2:
        return None, None
    dur = (T - 1) / fps
    g = np.arange(0.0, dur + 1e-9, 1.0 / rate) * fps        # grid in native-frame units
    if g.size < 2:
        return None, None
    k = np.minimum(np.floor(g).astype(int), T - 2)
    f = (g - k)[:, None]
    pos = xy[:, k, :] * (1 - f)[None] + xy[:, k + 1, :] * f[None]  # (2,G,V)
    valid = det[k] & det[k + 1]                              # (G,V)
    run = np.cumsum(~det, axis=0)                            # same run <=> no undetected frame between
    same = run[k[:-1]] == run[k[1:] + 1]
    ok = valid[:-1] & valid[1:] & same
    sp = np.linalg.norm(np.diff(pos, axis=1), axis=0) * rate
    return sp, ok


def descriptor(arr, mode, fps):
    if mode == "old":
        return old_descriptor(arr)
    xy, det = _prep(arr)
    if xy is None or xy.shape[1] < 2:
        return None
    if mode == "masked_frame":
        sp = np.linalg.norm(np.diff(xy, axis=1), axis=0)
        return _stats(sp, det[1:] & det[:-1])
    s = _shoulder_scale(xy, det)
    if s is None:
        return None
    xy = xy / s
    if mode == "masked_sec":
        sp = np.linalg.norm(np.diff(xy, axis=1), axis=0) * fps
        return _stats(sp, det[1:] & det[:-1])
    if mode == "v2_interp":
        xy, det = _bridge(xy, det, int(round(BRIDGE_S * fps)))
    elif mode != "v2":
        raise ValueError(mode)
    sp, ok = _resampled_speeds(xy, det, fps)
    if sp is None:
        return None
    return _stats(sp, ok)


# ---------------------------------------------------------------- data access
def _npz_dir_files(root):
    fs = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    if not fs:
        raise FileNotFoundError(root)
    return fs


def load_corpus_clips(path, n, rng):
    """Up to n raw clips (random, without replacement) from an .npy bundle or an .npz cache dir."""
    if path.endswith(".npy"):
        a = np.load(path, mmap_mode="r")
        idx = np.sort(rng.choice(len(a), size=min(n, len(a)), replace=False))
        return [np.array(a[i]) for i in idx]
    fs = _npz_dir_files(path)
    idx = rng.choice(len(fs), size=min(n, len(fs)), replace=False)
    out = []
    for i in idx:
        with np.load(fs[i]) as h:                          # I/O errors propagate
            out.append(np.array(h["data"]))
    return out


def load_manifest_clips(manifest, n, rng):
    """Proportional (stratified by 'source') sample of n clips from a pretraining manifest."""
    clips = json.load(open(manifest))["clips"]
    by = {}
    for c in clips:
        by.setdefault(c["source"], []).append(c["path"])
    tot = sum(len(v) for v in by.values())
    out = {}
    for src, paths in sorted(by.items()):
        k = int(round(n * len(paths) / tot))
        idx = rng.choice(len(paths), size=min(k, len(paths)), replace=False)
        L = []
        for i in idx:
            with np.load(paths[i]) as h:
                L.append(np.array(h["data"]))
        out[src] = L
    return out


def fps_of(source_or_path, reg):
    for key, val in reg["map"].items():
        if key in source_or_path:
            return float(val)
    raise KeyError(f"no fps registered for {source_or_path}")


# ---------------------------------------------------------------- MMD
def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0.0)


def mmd2_u(X, Y, sigma2):
    Kxx = np.exp(-sqdist(X, X) / sigma2); Kyy = np.exp(-sqdist(Y, Y) / sigma2); Kxy = np.exp(-sqdist(X, Y) / sigma2)
    n, m = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def median_sq(Z):
    d2 = sqdist(Z, Z)
    return float(np.median(d2[np.triu_indices(len(Z), 1)]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps-registry", default="docs/p1_fix_2026-10/step2/fps_registry.json")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--descs", nargs="+", default=DESCS)
    ap.add_argument("--pool-src", type=int, default=6000, help="clips loaded per source (descriptor pool)")
    ap.add_argument("--pool-tgt", type=int, default=2400, help="clips loaded per target (descriptor pool)")
    ap.add_argument("--n-src", type=int, default=800); ap.add_argument("--n-tgt", type=int, default=400)
    ap.add_argument("--draws", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    reg = json.load(open(a.fps_registry))
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    # descriptor pools: {desc: {corpus: array}}
    D = {d: {} for d in a.descs}
    meta = {"sources": {}, "targets": {}}
    for name in a.sources:
        spec = SOURCES[name]
        if isinstance(spec, str):
            groups = load_manifest_clips(spec, a.pool_src, rng)
            cnt = {}
            for c in json.load(open(spec))["clips"]:
                cnt[c["source"]] = cnt.get(c["source"], 0) + 1
            gw = {g: cnt[g] / sum(cnt.values()) for g in groups}      # manifest share
        else:
            groups = {p: load_corpus_clips(p, int(round(a.pool_src * w)), rng) for p, w in spec.items()}
            gw = dict(spec)                                            # e.g. v1 tool: 400 + 400
        meta["sources"][name] = {g: len(v) for g, v in groups.items()}
        for d in a.descs:
            rows, grp = [], []
            for g, clips in groups.items():
                f = fps_of(g, reg)
                for c in clips:
                    v = descriptor(c, d, f)
                    if v is not None:
                        rows.append(v); grp.append(g)
            D[d]["SRC:" + name] = (np.array(rows), np.array(grp), gw)
        print(f"[{time.time()-t0:.0f}s] source {name}: {meta['sources'][name]}", flush=True)
    for t in a.targets:
        p = TARGETS[t]
        if not os.path.exists(p):
            print(f"target {t} missing {p}"); continue
        clips = load_corpus_clips(p, a.pool_tgt, rng)
        f = fps_of(t, reg)
        meta["targets"][t] = {"path": p, "loaded": len(clips), "fps": f}
        for d in a.descs:
            rows = [descriptor(c, d, f) for c in clips]
            D[d]["TGT:" + t] = np.array([r for r in rows if r is not None])
        print(f"[{time.time()-t0:.0f}s] target {t}: {len(clips)} clips @ {f} fps", flush=True)

    # draws: for each draw, subsample source (stratified proportions preserved) and targets
    res = {d: {s: {t: {"pooled": [], "pair": []} for t in meta["targets"]} for s in a.sources} for d in a.descs}
    for r in range(a.draws):
        rr = np.random.default_rng(10_000 + r)
        for d in a.descs:
            T = {t: D[d]["TGT:" + t][rr.choice(len(D[d]["TGT:" + t]), size=min(a.n_tgt, len(D[d]["TGT:" + t])), replace=False)]
                 for t in meta["targets"]}
            for s in a.sources:
                X, grp, gw = D[d]["SRC:" + s]
                # stratified subsample with the source's mixing weights (manifest shares, or the
                # v1 tool's 50/50 WLASL/BdSLW60 split) -- NOT the loaded descriptor-pool shares
                sel = []
                for g in np.unique(grp):
                    ii = np.where(grp == g)[0]
                    k = int(round(a.n_src * gw[g]))
                    sel.append(rr.choice(ii, size=min(k, len(ii)), replace=False))
                Xs = X[np.concatenate(sel)]
                Z = np.vstack([Xs] + list(T.values()))
                s2_pool = median_sq(Z[rr.choice(len(Z), size=min(3000, len(Z)), replace=False)])
                for t, Y in T.items():
                    res[d][s][t]["pooled"].append(mmd2_u(Xs, Y, s2_pool))
                    res[d][s][t]["pair"].append(mmd2_u(Xs, Y, median_sq(np.vstack([Xs, Y]))))
        print(f"[{time.time()-t0:.0f}s] draw {r+1}/{a.draws}", flush=True)

    summ = {d: {s: {t: {bw: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "draws": v}
                         for bw, v in res[d][s][t].items()} for t in meta["targets"]} for s in a.sources} for d in a.descs}
    out = {"meta": meta, "args": vars(a), "fps_registry": reg, "statistic": "MMD^2_u (unbiased U-statistic), RBF",
           "results": summ}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    # compact table
    for d in a.descs:
        for s in a.sources:
            print(f"\n{d:12s} {s:16s} " + " ".join(f"{t}={summ[d][s][t]['pooled']['mean']:.4f}±{summ[d][s][t]['pooled']['sd']:.4f}" for t in meta["targets"]))


if __name__ == "__main__":
    main()
