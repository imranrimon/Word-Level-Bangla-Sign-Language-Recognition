#!/usr/bin/env python
"""P1 item6: refit tab:transfer (transferability estimators) and tab:robust (descriptor/metric
robustness sweep) against PRETRAINING-SEED-POOLED law gains instead of the single seed-0 gains.

Estimator values per calibration target are recomputed with exactly the code/sampling of
tools/distance_robustness.py (rng seed 0, n=400 per pool; cached to --cache so the CPU pass runs
once), naive controls (#classes, train size) from tools/transferability_baselines.py, and LogME
from results/logme_transfer.json (written by tools/logme_transfer.py; log-mean over sources).
Gains (5 calibration languages, GSL/BdSL/LSA64/KArSL/AUTSL) come from
tools/p1_law_pretrain_seed_analysis.py --json (per-pseed G matrix).

  python tools/p1_law_baseline_refit.py --gains-json <analysis.json> [--cache <dist.json>]
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
import distance_robustness as DR  # noqa: E402
from transferability_baselines import NCLS, SIZE  # noqa: E402  (order GSL,BdSL,LSA64,KArSL,AUTSL)

CAL = DR.TARGETS                    # ["GSL","BdSL","LSA64","KArSL","AUTSL"]
PAPER_GAIN = np.array([2.06, 1.70, 1.00, -4.40, -6.50])
VARIANTS = ([("descriptor", f"MMD / {m}", m, "mmd", False) for m in
             ["position", "velocity", "acceleration", "velocity_bbox", "velocity_fps"]] +
            [("descriptor", "MMD / velocity_zscore", "velocity", "mmd", True)] +
            [("metric", f"{m} / velocity", "velocity", m, False) for m in ["mmd", "energy", "frechet", "proxy_a", "knn"]])


def compute_dists(n):
    out = {}
    for axis, name, mode, metric, z in VARIANTS:
        d, e = DR.distances_for(mode, metric, n, np.random.default_rng(0), zscore=z)
        out[name] = None if d is None else d.tolist()
        print(f"  [{axis}] {name:24s} {e or np.round(d, 4).tolist()}", flush=True)
    return out


def logme_vec():
    p = os.path.join(ROOT, "results", "logme_transfer.json")
    d = json.load(open(p))
    key = {"GSL": "gsl", "BdSL": "bdsl", "LSA64": "lsa64", "KArSL": "karsl", "AUTSL": "autsl"}
    return np.array([np.log(np.mean([v for k, v in d.items() if k.endswith("2" + key[t])])) for t in CAL])


def fit(x, g):
    b, a = np.polyfit(x, g, 1)
    pred = a + b * x
    r2 = 1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    errs, sgn = [], 0
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        bb, aa = np.polyfit(x[m], g[m], 1); p = aa + bb * x[i]
        errs.append(abs(p - g[i])); sgn += int(np.sign(p) == np.sign(g[i]))
    from scipy.stats import spearmanr
    return dict(r2=r2, slope=b, dstar=-a / b if b else np.nan, mae=float(np.mean(errs)), sign=sgn,
                rho=spearmanr(x, g).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains-json", required=True, help="output of p1_law_pretrain_seed_analysis.py --json")
    ap.add_argument("--cache", default=os.path.join(ROOT, "docs/improvement_runs_2026-10/raw/item6_estimator_distances.json"))
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    if os.path.exists(a.cache):
        D = json.load(open(a.cache))
    else:
        print("computing estimator distances (distance_robustness.py sampling, rng 0) ...")
        D = compute_dists(a.n); json.dump(D, open(a.cache, "w"), indent=1)
    gj = json.load(open(a.gains_json))
    order = [gj["targets"].index(t) for t in CAL]
    G = np.array(gj["G"])[:, order]
    pseeds = gj["pseeds"]
    sources = {"paper (seed 0, rounded)": PAPER_GAIN, f"pooled ({len(pseeds)} pseeds)": G.mean(0)}
    if 0 in pseeds and len(pseeds) > 2:
        sources[f"pooled excl. ps0 ({len(pseeds) - 1})"] = G[[i for i, p in enumerate(pseeds) if p != 0]].mean(0)
    print("calibration targets:", CAL)
    for k, g in sources.items():
        print(f"  gains [{k}]: {np.round(g, 2).tolist()}")

    # per-pseed R^2 for the headline MMD, to show the spread
    mmd = np.array(D["MMD / velocity"])
    print("\nheadline-MMD (velocity) R^2 per pretraining seed: " +
          ", ".join(f"ps{p} {fit(mmd, G[i])['r2']:.3f}" for i, p in enumerate(pseeds)))

    est = [("pose-motion MMD (ours; one-pass law distance)", np.array([0.1206, 0.1429, 0.1763, 0.3445, 0.4115]))]
    est += [(n, np.array(D[n])) for n in ["energy / velocity", "frechet / velocity", "proxy_a / velocity", "knn / velocity"] if D.get(n)]
    est += [("LogME (log-mean over sources)", logme_vec()), ("target #classes", NCLS), ("target train size", SIZE)]
    print("\n## tab:transfer refit (5 calibration targets; R² of gain ~ estimator)\n")
    print("| estimator | " + " | ".join(f"R² [{k}]" for k in sources) + " | " + " | ".join(f"LOO sign [{k}]" for k in sources) + " |")
    print("|---|" + "---|" * (2 * len(sources)))
    for name, x in est:
        fs = [fit(x, g) for g in sources.values()]
        print(f"| {name} | " + " | ".join(f"{f['r2']:.3f}" for f in fs) + " | " + " | ".join(f"{f['sign']}/5" for f in fs) + " |")

    print("\n## tab:robust refit (descriptor / metric variants)\n")
    print("| axis | variant | " + " | ".join(f"R² / d* / LOO MAE / sign [{k}]" for k in sources) + " |")
    print("|---|---|" + "---|" * len(sources))
    for axis, name, *_ in VARIANTS:
        if not D.get(name):
            print(f"| {axis} | {name} | unavailable |"); continue
        x = np.array(D[name])
        fs = [fit(x, g) for g in sources.values()]
        print(f"| {axis} | {name} | " + " | ".join(f"{f['r2']:.3f} / {f['dstar']:.3f} / {f['mae']:.2f} / {f['sign']}/5" for f in fs) + " |")


if __name__ == "__main__":
    main()
