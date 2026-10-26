#!/usr/bin/env python
"""P1 item5: pretraining-seed replication of the headline BdSL @10% xling-vs-mono gain.

Groups (experiment base = bdsl_block_gcn_si_lr01_f10_<TAG>, fine-tune seeds 0,1,2):
  pretrain seed 0 (original headline)   : mono / xling
  pretrain seed 1 / 2 (new backbones)   : mono_ps1 / xling_ps1, mono_ps2 / xling_ps2
  controls (not pooled)                 : mono_rerun / xling_rerun  (seed-0 backbones, current code)
                                          mono_w1bb  / xling_w1bb   (width-study 1x backbones, headline
                                                                     unclipped recipe)
Metrics: VAL = best-epoch val Top-1 (the headline's metric, seed-mean; P1 Table tab:sig +4.2),
         TEST = best-val checkpoint scored on held-out TEST (*_seed<S>_test rows).
Hierarchical bootstrap: per arm, resample pretrain seeds with replacement, then fine-tune seeds
within each drawn pretrain seed (2-level); a 3-level variant also resamples clips (per-clip
correctness from eval_results/best_acc.pkl).
"""
import argparse, csv, os, pickle
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEM = "bdsl_block_gcn_si_lr01_f10_"
PSEEDS = {0: "", 1: "_ps1", 2: "_ps2"}
CONTROLS = ["_rerun", "_w1bb"]
FT = [0, 1, 2]


def load_csv(path):
    best, test = defaultdict(lambda: -1.0), {}
    for r in csv.DictReader(open(path)):
        e = r["Experiment"]
        if not e.startswith(STEM):
            continue
        v = float(r["Top1_Acc"])
        if e.endswith("_test"):
            test[e] = v                       # last row wins (single eval)
        else:
            best[e] = max(best[e], v)
    return best, test


def run_name(arm, suf, s):
    return f"{STEM}{arm}{suf}_seed{s}"


def table(best, test, suf):
    out = {}
    for arm in ("mono", "xling"):
        out[arm] = {
            "val": [best.get(run_name(arm, suf, s), np.nan) for s in FT],
            "test": [test.get(run_name(arm, suf, s) + "_test", np.nan) for s in FT],
        }
    return out


def clip_correct(exp, split):
    """Per-clip 0/1 correctness vector for a run (val: train dir; test: *_test dir)."""
    d = os.path.join(ROOT, "work_dir", exp + ("_test" if split == "test" else ""), "eval_results", "best_acc.pkl")
    lab = os.path.join(ROOT, "data", "bdsl_si", f"{split}_label.pkl")
    if not os.path.exists(d):
        return None
    sc = pickle.load(open(d, "rb"))
    names, labels = pickle.load(open(lab, "rb"))
    lmap = dict(zip(names, labels))
    keys = sorted(sc)
    return np.array([int(np.argmax(sc[k]) == lmap[k]) for k in keys], dtype=np.float64)


def hier_boot(arr_x, arr_m, n=20000, rng=None):
    """arr_*: (P, F) per-run accuracies. Unpaired 2-level bootstrap of mean(x)-mean(m)."""
    rng = rng or np.random.default_rng(0)
    def draw(a):
        P, F = a.shape
        pi = rng.integers(0, P, size=(n, P))
        fi = rng.integers(0, F, size=(n, P, F))
        return a[pi[..., None], fi].mean(axis=(1, 2))
    d = draw(arr_x) - draw(arr_m)
    return d


def hier_boot3(cx, cm, n=4000, rng=None):
    """cx/cm: list(P) of list(F) of per-clip 0/1 arrays. 3-level unpaired bootstrap."""
    rng = rng or np.random.default_rng(1)
    def draw(c):
        P = len(c); F = len(c[0]); out = np.empty(n)
        for b in range(n):
            acc = []
            for p in rng.integers(0, P, P):
                for f in rng.integers(0, F, F):
                    v = c[p][f]
                    acc.append(v[rng.integers(0, v.size, v.size)].mean())
            out[b] = np.mean(acc)
        return out
    return draw(cx) - draw(cm)


def fmt(a):
    a = np.asarray(a, float) * 100
    return f"{np.nanmean(a):.2f} ± {np.nanstd(a, ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "results_final.csv"))
    ap.add_argument("--clip-boot", action="store_true", help="also run the 3-level clip bootstrap")
    a = ap.parse_args()
    best, test = load_csv(a.csv)

    rows, per = [], {}
    for ps, suf in list(PSEEDS.items()) + [(c, c) for c in CONTROLS]:
        t = table(best, test, suf)
        per[ps] = t
        line = [str(ps)]
        for m in ("val", "test"):
            g = np.nanmean(t["xling"][m]) - np.nanmean(t["mono"][m])
            line += [fmt(t["mono"][m]), fmt(t["xling"][m]), f"{100*g:+.2f}"]
        line.append(" / ".join(f"{100*v:.2f}" if not np.isnan(v) else "NA" for v in t["mono"]["val"] + t["xling"]["val"]))
        line.append(" / ".join(f"{100*v:.2f}" if not np.isnan(v) else "NA" for v in t["mono"]["test"] + t["xling"]["test"]))
        rows.append(line)
    print("| pretrain seed | VAL mono | VAL xling | VAL gain | TEST mono | TEST xling | TEST gain | val runs (m0 m1 m2 / x0 x1 x2) | test runs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")

    pooled = [0, 1, 2]
    for m in ("val", "test"):
        X = np.array([per[p]["xling"][m] for p in pooled]); M = np.array([per[p]["mono"][m] for p in pooled])
        if np.isnan(X).any() or np.isnan(M).any():
            print(f"\n[{m}] incomplete -> skip pooled bootstrap"); continue
        obs = X.mean() - M.mean()
        d = hier_boot(X, M)
        lo, hi = np.percentile(d, [2.5, 97.5])
        gains = (X.mean(1) - M.mean(1)) * 100
        print(f"\n[{m}] pooled xling-mono = {100*obs:+.2f}pp  2-level hier-boot 95% CI [{100*lo:+.2f}, {100*hi:+.2f}]"
              f"  P(gain<=0)={np.mean(d <= 0):.4f}  per-pseed gains {np.round(gains,2).tolist()}"
              f"  (SD over pseeds {gains.std(ddof=1):.2f})")
        # pretrain-seed-only bootstrap (ft seeds averaged): degenerate n=3, show for reference
        if a.clip_boot:
            cx = [[clip_correct(run_name("xling", PSEEDS[p], s), m) for s in FT] for p in pooled]
            cm = [[clip_correct(run_name("mono", PSEEDS[p], s), m) for s in FT] for p in pooled]
            if any(v is None for r in cx + cm for v in r):
                print(f"[{m}] clip scores missing -> skip 3-level"); continue
            d3 = hier_boot3(cx, cm)
            lo3, hi3 = np.percentile(d3, [2.5, 97.5])
            print(f"[{m}] 3-level (pseed x ftseed x clip) 95% CI [{100*lo3:+.2f}, {100*hi3:+.2f}]  P(gain<=0)={np.mean(d3 <= 0):.4f}")


if __name__ == "__main__":
    main()
