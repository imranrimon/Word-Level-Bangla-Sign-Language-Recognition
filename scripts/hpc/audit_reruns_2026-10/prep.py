#!/usr/bin/env python
"""2026-10 P1 audit reruns -- leak-free SSL pools, distances and configs.

Subcommands (each idempotent; run on a compute node, inside env bdsl_graph):
  r1noautsl     multisrc pool with ALL AUTSL clips removed (AUTSL target is
                genuinely foreign) + recomputed MMD(source lang -> AUTSL train)
                + pretrain configs *_noautsl.
  r1autsltrain  multisrc pool with AUTSL restricted to its TRAIN signers (no
                val/test signer clips) -- needed by CORAL, whose loss requires
                target-language clips -- + configs *_autsltrain.
  r2mono        TRAIN-SIGNERS-ONLY monolingual pools for LSA64 / KArSL / AUTSL /
                GSL + configs bdsl_shubert_pretrain_monotr_<t>.

Every manifest written here is checked: no clip from a target's val/test signers
(and, for KArSL, none of its seen-signer val clips) may enter that target's pool.
"""
import argparse, collections, json, os, pickle, re, sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from feeders.pretrain_feeder import language_of  # noqa: E402

AUTSL_VAL = {1, 11, 16, 18, 25, 35}
AUTSL_TEST = {6, 14, 27, 30, 34, 39}
MULTISRC = "data/ssl_pool_manifest_multisrc.json"


def load(p):
    with open(p) as f:
        return json.load(f)


def dump_manifest(src, clips, out, note):
    m = {k: v for k, v in src.items() if k != "clips"}
    m["clips"] = clips
    m["per_source_counts"] = dict(collections.Counter(c["source"] for c in clips))
    m["note"] = note
    with open(out, "w") as f:
        json.dump(m, f)
    print(f"wrote {out}: {len(clips)} clips {m['per_source_counts']}")


def autsl_signer(path):
    return int(re.search(r"signer(\d+)", path).group(1))


def is_autsl(c):
    return "autsl" in (c.get("source", "") + c.get("path", "")).lower()


# ---------------------------------------------------------------- distances
def _clip_descriptor(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        a = a[0]
    xy = a[:2, :, :, 0]
    nz = np.abs(xy).sum((0, 2)) > 0
    xy = xy[:, nz, :] if nz.any() else xy
    if xy.shape[1] < 2:
        return None
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=0)
    return np.concatenate([vel.mean(0), vel.std(0)])


def _mmd_rbf(X, Y):  # identical estimator to tools/compute_lang_distance.py
    X, Y = np.asarray(X), np.asarray(Y)
    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sigma2 = np.median(d2[d2 > 0]) + 1e-9
    K = np.exp(-d2 / sigma2)
    n, m = len(X), len(Y)
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    return (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean()


def recompute_autsl_distances(clips, n=400, seed=0):
    """MMD(source language -> AUTSL TRAIN split), same descriptor/estimator/n as the
    paper registry. Source clips per language come from the AUTSL-free pool."""
    rng = np.random.default_rng(seed)
    by_lang = collections.defaultdict(list)
    for c in clips:
        by_lang[language_of(c["source"])].append(c["path"])
    a = np.load("data/autsl_si/train_data.npy", mmap_mode="r")
    idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
    tgt = [d for d in (_clip_descriptor(a[i]) for i in idx) if d is not None]
    dist = {}
    for lang in sorted(by_lang):
        paths = list(by_lang[lang]); rng.shuffle(paths)
        src = []
        for p in paths[: n * 2]:
            try:
                with np.load(p) as h:
                    d = _clip_descriptor(h["data"])
                if d is not None:
                    src.append(d)
            except OSError:  # unreadable storage (e.g. Lustre OST offline) must not silently swap the clip sample
                raise
            except Exception:
                pass
            if len(src) >= n:
                break
        dist[lang] = float(_mmd_rbf(src, tgt))
        print(f"  MMD(lang {lang} -> AUTSL train) = {dist[lang]:.4f}  (n_src={len(src)}, n_tgt={len(tgt)})")
    return dist


# ---------------------------------------------------------------- configs
def write_cfg(orig, new_name, manifest, targets, overrides=None, drop=()):
    with open(f"config/{orig}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["Experiment_name"] = new_name
    cfg["pretrain"]["manifest"] = manifest
    cfg["pretrain"]["targets"] = targets
    for k in drop:
        cfg["pretrain"].pop(k, None)
    for k, v in (overrides or {}).items():
        cfg["pretrain"][k] = v
    cfg["num_worker"] = 4
    out = f"config/{new_name}.yaml"
    with open(out, "w") as f:
        f.write(f"# 2026-10 P1 audit rerun: leak-free copy of config/{orig}.yaml\n"
                f"# (only manifest/targets/Experiment_name/num_worker"
                f"{' + ' + ', '.join(list((overrides or {}).keys()) + list(drop)) if (overrides or drop) else ''} changed)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {out}")


# ---------------------------------------------------------------- subcommands
def r1noautsl():
    src = load(MULTISRC)
    out = "data/ssl_pool_manifest_multisrc_noautsl.json"
    kept = [c for c in src["clips"] if not is_autsl(c)]
    assert not any(is_autsl(c) for c in kept)
    dump_manifest(src, kept, out, "multisrc minus ALL AUTSL clips (2026-10 audit R1): AUTSL target genuinely foreign")
    tgt = "data/pretrain_kmeans_targets_multisrc_noautsl.npz"
    dist_path = "docs/audit_reruns_2026-10/r1_autsl_distances.json"
    if os.path.exists(dist_path):
        dist = {int(k): v for k, v in load(dist_path)["recomputed"].items()}
    else:
        dist = recompute_autsl_distances(kept)
        os.makedirs(os.path.dirname(dist_path), exist_ok=True)
        with open(dist_path, "w") as f:
            json.dump({"recomputed": dist,
                       "original_autsl_anchored": [0.1207, 0.8357, 0.4458, 0.0, 0.7242],
                       "lang_ids": "0=BdSL 1=ASL 2=LSA64 3=AUTSL 4=KArSL",
                       "estimator": "RBF-MMD^2 biased diag-removed, median heuristic, n=400, seed 0, "
                                    "per-joint speed mean+std; target = data/autsl_si/train_data.npy"}, f, indent=1)
    # AUTSL's own entry stays 0.0 as in the original vector; it has no clips in this
    # pool, and the hard gate is a relative re-weight so present-source ratios are unchanged.
    vec = [round(dist.get(0), 4), round(dist.get(1), 4), round(dist.get(2), 4), 0.0, round(dist.get(4), 4)]
    print("recomputed AUTSL-anchored source_distances:", vec)
    write_cfg("bdsl_shubert_pretrain_pool6_uniform", "bdsl_shubert_pretrain_pool6_uniform_noautsl", out, tgt)
    write_cfg("bdsl_shubert_pretrain_dgxl", "bdsl_shubert_pretrain_dgxl_noautsl", out, tgt)  # BdSL-keyed vector unchanged
    write_cfg("bdsl_shubert_pretrain_dgxl_autsl", "bdsl_shubert_pretrain_dgxl_autsl_noautsl", out, tgt,
              {"source_distances": vec})
    write_cfg("bdsl_shubert_pretrain_lambdad_autsl", "bdsl_shubert_pretrain_lambdad_autsl_noautsl", out, tgt,
              {"source_distances": vec})
    write_cfg("bdsl_shubert_pretrain_pool6_uniform", "bdsl_shubert_pretrain_poolnoalign_noautsl", out, tgt,
              {"lambda_adv": 0.0, "lambda_contrast": 0.0})


def r1autsltrain():
    src = load(MULTISRC)
    out = "data/ssl_pool_manifest_multisrc_autsltrain.json"
    kept = [c for c in src["clips"]
            if not (is_autsl(c) and autsl_signer(c["path"]) in AUTSL_VAL | AUTSL_TEST)]
    bad = [c for c in kept if is_autsl(c) and autsl_signer(c["path"]) in AUTSL_VAL | AUTSL_TEST]
    assert not bad
    n_autsl = sum(is_autsl(c) for c in kept)
    print(f"AUTSL train-signer clips kept: {n_autsl}")
    dump_manifest(src, kept, out, "multisrc with AUTSL restricted to TRAIN signers (2026-10 audit R1, CORAL needs target clips)")
    tgt = "data/pretrain_kmeans_targets_multisrc_autsltrain.npz"
    write_cfg("bdsl_shubert_pretrain_coral_autsl", "bdsl_shubert_pretrain_coral_autsl_autsltrain", out, tgt)
    write_cfg("bdsl_shubert_pretrain_pool6_uniform", "bdsl_shubert_pretrain_poolnoalign_autsltrain", out, tgt,
              {"lambda_adv": 0.0, "lambda_contrast": 0.0})


def _stems(pkl):
    names, _ = pickle.load(open(pkl, "rb"))
    return {os.path.splitext(os.path.basename(n))[0] for n in names}


def r2mono():
    rules = {
        # target: (src manifest, keep(path) -> bool)
        "lsa64": ("data/ssl_pool_manifest_lsa64.json",
                  lambda p: int(os.path.basename(p).split("_")[1]) in range(1, 8)),
        "karsl": ("data/ssl_pool_manifest_karsl.json",
                  lambda p: os.path.basename(p).split("_")[1] in ("01", "02")),
        "autsl": ("data/ssl_pool_manifest_autsl.json",
                  lambda p: autsl_signer(p) not in AUTSL_VAL | AUTSL_TEST),
        "gsl": ("data/ssl_pool_manifest_gsl.json",
                lambda p: int(re.search(r"signer(\d+)", p).group(1)) in range(1, 6)),
    }
    for t, (mf, keep) in rules.items():
        src = load(mf)
        kept = [c for c in src["clips"] if keep(c["path"])]
        if t == "karsl":  # val is a seen-signer carve from signers 1/2: drop those exact clips too
            vstem = _stems("data/karsl_si/val_label.pkl")
            kept = [c for c in kept if os.path.splitext(os.path.basename(c["path"]))[0] not in vstem]
        # leak guard: no val/test stem may survive (GSL stems are not unique -> signer rule above is the guard)
        if t != "gsl":
            held = _stems(f"data/{t}_si/val_label.pkl") | _stems(f"data/{t}_si/test_label.pkl")
            leak = [c for c in kept if os.path.splitext(os.path.basename(c["path"]))[0] in held]
            assert not leak, (t, len(leak))
        out = f"data/ssl_pool_manifest_{t}_trainonly.json"
        dump_manifest(src, kept, out, f"{t} TRAIN-signers-only monolingual pool (2026-10 audit R2); "
                                      f"was {len(src['clips'])} clips incl. val/test signers")
        write_cfg("bdsl_shubert_pretrain_src_gsl", f"bdsl_shubert_pretrain_monotr_{t}", out,
                  f"data/pretrain_kmeans_targets_{t}_trainonly.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["r1noautsl", "r1autsltrain", "r2mono"])
    {"r1noautsl": r1noautsl, "r1autsltrain": r1autsltrain, "r2mono": r2mono}[ap.parse_args().cmd]()
