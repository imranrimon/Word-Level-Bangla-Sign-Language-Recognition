#!/usr/bin/env python
"""P1 checkpoint-aware predictor of the alignment gain G = acc(xling) - acc(pool): feature extraction.

For every available SSL backbone (arm in {pool, xling} x pretraining seed p) and every P1 law target
language, extracts features of TARGET TRAIN-split clips only (never val/test) plus a fixed sample of
SOURCE-pool clips (data/ssl_pool_manifest_bdsl_asl.json, the pool both arms were pretrained on):

  view "ft"  (what fine-tuning sees): fine-tune preprocessing of feeders.feeder.Feeder in eval mode
             (tools.center_choose to window 120 + joint-0 centering), BlockGCN with the fine-tune
             model_args (stride_between_stages=True), pooled 256-d clip feature. BatchNorm statistics are
             re-estimated on the target train clips (AdaBN), approximating train-mode BN at the start
             of fine-tuning; source clips go through the same target-adapted function.
  view "ftj" (same as "ft" but BatchNorm re-estimated on the source+target union, so source and
             target go through a shared normalisation; used for source-vs-target feature distances).
  view "ssl" (what pretraining saw): raw (uncentered) pose, centre crop / zero-pad to 120,
             stride_between_stages=False, pretrained BN running stats; per-frame features mean-pooled
             over valid frames. When train_state.pt (with the SSL head + mask token) exists, also the
             masked code-prediction loss / accuracy on the clips, using k-means codes assigned with the
             pool's own cluster centres (pose_motion, data/pretrain_kmeans_targets_bdsl_asl.npz) and 4
             fixed masks (ratio 0.3, shared by all backbones).

Clip sets per language (deterministic, shared by all backbones): the 10% labelled subset exactly as the
fine-tunes draw it (Feeder._subsample: stratified, frac 0.1, seed 42) UNION a random 3000-clip unlabelled
sample of the train split (seed 0). Output: <out>/<lang>/<arm>_ps<p>.npz (idempotent: skipped if present).

  python tools/p1_ckpt_predictor/extract.py [--pseeds 0-6] [--langs ...] [--out work_dir/p1_ckpt_predictor]
"""
from __future__ import annotations

import argparse, copy, json, os, pickle, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from feeders import tools as ftools  # noqa: E402
from model.block_gcn import Model as BlockGCN  # noqa: E402

LANG_CFG = {"BdSL": "config/bdsl_block_gcn_si_lr01_f10.yaml", "LSA64": "config/lsa64_si_lr01_f10.yaml",
            "KArSL": "config/karsl_si_lr01_f10.yaml", "AUTSL": "config/autsl_si_lr01_f10.yaml",
            "GSL": "config/gsl_si_lr01_f10.yaml", "LSFB": "config/lsfb_si_lr01_f10.yaml",
            "LIBRAS": "config/libras_si_lr01_f10.yaml"}
PEXP = {"pool": "bdsl_shubert_pretrain_bdsl_asl", "xling": "bdsl_shubert_pretrain_xlingual"}
POOL_MANIFEST = "data/ssl_pool_manifest_bdsl_asl.json"
KMEANS = "data/pretrain_kmeans_targets_bdsl_asl.npz"
W = 120
N_UNL = 3000
N_SRC = 2000
N_MASKS = 4
MASK_RATIO = 0.3
BB_ARGS = dict(num_class=60, num_point=27, num_person=1, graph="graph.sign_27.Graph",
               graph_args={"labeling_mode": "spatial"}, in_channels=3, num_blocks=[2, 2, 2, 2],
               stage_channels=[64, 128, 256, 256], max_distance=8)


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def backbone_paths(arm, p):
    d = os.path.join(ROOT, "work_dir", PEXP[arm] + ("" if p == 0 else f"_pseed{p}"))
    return os.path.join(d, "pretrained_epoch30.pt"), os.path.join(d, "train_state.pt")


def backbone_ready(arm, p):
    bb, st = backbone_paths(arm, p)
    if not os.path.exists(bb):
        return False
    if p == 0:
        return True
    # pseed>=1 runs write train_state alongside; require it at the final epoch (no half-written run)
    if not os.path.exists(st):
        return False
    try:
        return int(torch.load(st, map_location="cpu")["epoch"]) == 29
    except Exception:
        return False


# ------------------------------------------------------------------ preprocessing
def ft_view(clip):
    """Feeder eval-mode transform (random_choose=False, normalization=True, no shift/mirror)."""
    x = np.array(clip, dtype=np.float32)
    x = ftools.center_choose(x, W).astype(np.float32)
    x[0] = x[0] - x[0, :, 0, 0].mean(axis=0)
    x[1] = x[1] - x[1, :, 0, 0].mean(axis=0)
    return x


def per_frame_pose_motion(x):
    """(C,T,V,M) -> (T,2*C*V), as preprocessing/compute_pretrain_targets.py (pose_motion)."""
    C, T, V, M = x.shape
    pose = x[:, :, :, 0].transpose(1, 0, 2).reshape(T, C * V).astype(np.float32)
    motion = np.zeros_like(pose)
    if T > 1:
        motion[1:] = pose[1:] - pose[:-1]
    return np.concatenate([pose, motion], 1)


def ssl_view(clip, centers):
    """Raw pose (no centering) trimmed to last non-zero frame, centre-cropped / zero-padded to W.
    Returns x (C,W,V,M), codes (W,), valid (W,) bool."""
    x = np.asarray(clip, dtype=np.float32)
    nz = x.sum(axis=(0, 2, 3)) != 0
    T = int(np.where(nz)[0].max()) + 1 if nz.any() else 1
    x = x[:, :T]
    f = per_frame_pose_motion(x)
    d2 = (f ** 2).sum(1, keepdims=True) - 2 * f @ centers.T + (centers ** 2).sum(1)[None]
    codes = d2.argmin(1).astype(np.int64)
    C, _, V, M = x.shape
    out = np.zeros((C, W, V, M), np.float32); cod = np.zeros(W, np.int64); val = np.zeros(W, bool)
    if T >= W:
        s = (T - W) // 2
        out[:] = x[:, s:s + W]; cod[:] = codes[s:s + W]; val[:] = True
    else:
        out[:, :T] = x; cod[:T] = codes; val[:T] = True
    return out, cod, val


def subsample_idx(labels, frac=0.1, seed=42):
    """Exact replica of feeders.feeder.Feeder._subsample index selection."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * frac)))
        idx.extend(rng.choice(ci, k, replace=False).tolist())
    return np.array(sorted(idx))


def load_lang(lang, centers):
    cfg = yaml.safe_load(open(os.path.join(ROOT, LANG_CFG[lang])))
    fa = cfg["train_feeder_args"]
    assert "train" in os.path.basename(fa["data_path"]), fa["data_path"]   # TRAIN split only
    data = np.load(os.path.join(ROOT, fa["data_path"]), mmap_mode="r")
    with open(os.path.join(ROOT, fa["label_path"]), "rb") as fh:
        _, labels = pickle.load(fh, encoding="latin1")
    labels = np.asarray(labels, int)
    assert float(cfg.get("train_fraction", 1.0)) == 0.1 and int(cfg.get("subset_seed", 42)) == 42
    lab = subsample_idx(labels, 0.1, 42)
    rng = np.random.default_rng(0)
    unl = np.sort(rng.choice(len(labels), size=min(N_UNL, len(labels)), replace=False))
    U = np.union1d(lab, unl)
    Xft, Xs, Cs, Vs = [], [], [], []
    for i in U:
        clip = np.asarray(data[i])
        Xft.append(ft_view(clip))
        a, b, c = ssl_view(clip, centers)
        Xs.append(a); Cs.append(b); Vs.append(c)
    return dict(idx=U, labels=labels[U], is_lab=np.isin(U, lab), is_unl=np.isin(U, unl),
                num_class=int(cfg["model_args"]["num_class"]),
                Xft=np.stack(Xft), Xs=np.stack(Xs), Cs=np.stack(Cs), Vs=np.stack(Vs))


def load_src(centers):
    clips = json.load(open(os.path.join(ROOT, POOL_MANIFEST)))["clips"]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(clips))
    Xft, Xs, Cs, Vs, srcs, paths = [], [], [], [], [], []
    for j in order:
        if len(Xft) >= N_SRC:
            break
        p = clips[j]["path"]
        with np.load(os.path.join(ROOT, p)) as h:     # I/O errors must raise, not swap the sample
            clip = h["data"]
        if not (clip.sum(axis=(0, 2, 3)) != 0).any():
            continue
        Xft.append(ft_view(clip))
        a, b, c = ssl_view(clip, centers)
        Xs.append(a); Cs.append(b); Vs.append(c); srcs.append(clips[j]["source"]); paths.append(p)
    return dict(Xft=np.stack(Xft), Xs=np.stack(Xs), Cs=np.stack(Cs), Vs=np.stack(Vs),
                source=np.array(srcs), path=np.array(paths))


# ------------------------------------------------------------------ models
def build(arm, p, dev):
    bbp, stp = backbone_paths(arm, p)
    sd = torch.load(bbp, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    ft = BlockGCN(**BB_ARGS, return_features=True, stride_between_stages=True)
    ft.load_state_dict(sd, strict=False)
    ssl = BlockGCN(**BB_ARGS, return_features=True, stride_between_stages=False)
    ssl.load_state_dict(sd, strict=False)
    head, tok = None, None
    if os.path.exists(stp):
        st = torch.load(stp, map_location="cpu")["pretr"]
        for k in ("layers.0.gcn.bn.weight", "data_bn.running_mean"):
            assert torch.equal(st["backbone.backbone." + k], sd[k]), f"train_state != backbone ({k})"
        head = nn.Linear(256, st["head.weight"].shape[0])
        head.load_state_dict({"weight": st["head.weight"], "bias": st["head.bias"]})
        tok = st["mask_token"].clone()
        head, tok = head.to(dev).eval(), tok.to(dev)
    return ft.to(dev).eval(), ssl.to(dev).eval(), head, tok


def ssl_per_frame(bb, x):
    """Generic per-time replay, as main_pretrain._TimeFeatureBackbone: (N,C,T,V,M) -> (N,T,C)."""
    N, C, T, V, M = x.shape
    y = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
    y = bb.data_bn(y)
    y = y.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
    for layer in bb.layers:
        y = layer(y)
    y = y.mean(dim=-1)
    y = y.view(N, M, -1, y.size(-1)).mean(dim=1)
    return y.transpose(1, 2).contiguous()


@torch.no_grad()
def adabn(model, X, dev, bs=256):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None   # cumulative average
    m.train()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g).numpy()
    for i in range(0, len(X), bs):
        sel = np.sort(perm[i:i + bs])
        if len(sel) < 2:
            continue
        m(torch.from_numpy(X[sel]).to(dev))
    return m.eval()


@torch.no_grad()
def feats_ft(m, X, dev, bs=256):
    return np.concatenate([m(torch.from_numpy(X[i:i + bs]).to(dev)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


@torch.no_grad()
def feats_ssl(m, head, tok, X, Cd, Vd, masks, dev, bs=128):
    """Mean-pooled (valid frames) per-frame features; masked-prediction CE/acc per clip (mean over masks)."""
    pooled, loss, acc = [], [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        v = torch.from_numpy(Vd[i:i + bs]).to(dev).float()
        f = ssl_per_frame(m, x)                                           # (n,T,256)
        pooled.append(((f * v[..., None]).sum(1) / v.sum(1, keepdim=True).clamp(min=1)).cpu().numpy())
        if head is None:
            continue
        c = torch.from_numpy(Cd[i:i + bs]).to(dev)
        L, A = torch.zeros(len(x), device=dev), torch.zeros(len(x), device=dev)
        for mk in masks:
            mk = mk.to(dev)
            mm = mk.view(1, 1, -1, 1, 1).expand_as(x)
            xm = torch.where(mm, tok.expand_as(x), x)
            lg = head(ssl_per_frame(m, xm))                               # (n,T,K)
            ce = F.cross_entropy(lg.transpose(1, 2), c, reduction="none")  # (n,T)
            w = v * mk.float()[None]
            L += (ce * w).sum(1) / w.sum(1).clamp(min=1)
            A += ((lg.argmax(-1) == c).float() * w).sum(1) / w.sum(1).clamp(min=1)
        loss.append((L / len(masks)).cpu().numpy()); acc.append((A / len(masks)).cpu().numpy())
    out = dict(pooled=np.concatenate(pooled))
    if loss:
        out["loss"] = np.concatenate(loss); out["acc"] = np.concatenate(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseeds", default="0-6")
    ap.add_argument("--arms", nargs="+", default=["pool", "xling"])
    ap.add_argument("--langs", nargs="+", default=list(LANG_CFG))
    ap.add_argument("--out", default=os.path.join(ROOT, "work_dir", "p1_ckpt_predictor"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    os.makedirs(a.out, exist_ok=True)

    bbs = [(arm, p) for p in parse_range(a.pseeds) for arm in a.arms if backbone_ready(arm, p)]
    print(f"[extract] backbones ready: {bbs}", flush=True)
    todo = {l: [b for b in bbs if not os.path.exists(os.path.join(a.out, l, f"{b[0]}_ps{b[1]}.npz"))]
            for l in a.langs}
    if not any(todo.values()):
        print("[extract] nothing to do"); return

    centers = np.load(os.path.join(ROOT, KMEANS))["cluster_centers"].astype(np.float32)
    g = torch.Generator().manual_seed(1234)
    masks = []
    for _ in range(N_MASKS):
        mk = torch.zeros(W, dtype=torch.bool)
        mk[torch.randperm(W, generator=g)[: int(round(W * MASK_RATIO))]] = True
        masks.append(mk)

    t0 = time.time()
    src_cache = os.path.join(a.out, "_src_inputs.npz")
    if os.path.exists(src_cache):
        src = dict(np.load(src_cache, allow_pickle=False))
    else:
        src = load_src(centers)
        np.savez(src_cache + ".tmp.npz", **src); os.replace(src_cache + ".tmp.npz", src_cache)
    print(f"[extract] source sample {len(src['Xft'])} clips ({time.time() - t0:.0f}s)", flush=True)

    for lang in a.langs:
        if not todo[lang]:
            continue
        t1 = time.time()
        D = load_lang(lang, centers)
        os.makedirs(os.path.join(a.out, lang), exist_ok=True)
        print(f"[extract] {lang}: {len(D['idx'])} clips ({D['is_lab'].sum()} labelled) "
              f"loaded in {time.time() - t1:.0f}s", flush=True)
        for arm, p in todo[lang]:
            t2 = time.time()
            ft, ssl, head, tok = build(arm, p, dev)
            ftA = adabn(ft, D["Xft"], dev)
            out = dict(idx=D["idx"], labels=D["labels"], is_lab=D["is_lab"], is_unl=D["is_unl"],
                       num_class=D["num_class"],
                       ft_tgt=feats_ft(ftA, D["Xft"], dev), ft_src=feats_ft(ftA, src["Xft"], dev),
                       ft_tgt_noada=feats_ft(ft, D["Xft"], dev))
            # shared normalisation for source-vs-target comparisons: AdaBN on source+target union
            ftJ = adabn(ft, np.concatenate([D["Xft"], src["Xft"]]), dev)
            out["ftj_tgt"], out["ftj_src"] = feats_ft(ftJ, D["Xft"], dev), feats_ft(ftJ, src["Xft"], dev)
            st = feats_ssl(ssl, head, tok, D["Xs"], D["Cs"], D["Vs"], masks, dev)
            ss = feats_ssl(ssl, head, tok, src["Xs"], src["Cs"], src["Vs"], masks, dev)
            out["ssl_tgt"], out["ssl_src"] = st["pooled"], ss["pooled"]
            if "loss" in st:
                out.update(ssl_loss_tgt=st["loss"], ssl_acc_tgt=st["acc"],
                           ssl_loss_src=ss["loss"], ssl_acc_src=ss["acc"])
            fn = os.path.join(a.out, lang, f"{arm}_ps{p}.npz")
            np.savez(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn)
            msg = f" ssl_loss tgt {st['loss'].mean():.3f} src {ss['loss'].mean():.3f}" if "loss" in st else ""
            print(f"[extract] {lang} {arm} ps{p} done in {time.time() - t2:.0f}s{msg}", flush=True)
            del ft, ssl, ftA, ftJ
            torch.cuda.empty_cache()
    print(f"[extract] all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
