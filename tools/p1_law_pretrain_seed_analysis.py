#!/usr/bin/env python
"""P1 item6: does the distance-gating law survive pretraining-seed noise?

Law quantity (tools/gating_law_fit.py, P1 Tables tab:law / tab:loocv):
    G_align(target) = xling - pool   (best-epoch VAL Top-1 @10% labels, lr 0.01, unclipped,
                                     seed-mean over fine-tune seeds; 3 seeds, 5 for GSL/LSFB/LIBRAS)
regressed on the fixed, label-free pool->target pose-motion MMD. Every target fine-tunes the SAME
two backbones (pool = bdsl_shubert_pretrain_bdsl_asl, xling = bdsl_shubert_pretrain_xlingual), so
pretraining seed p in {0 (original), 1, 2} is shared across all seven law points.

Experiment names:  <stem>_lr01_f10_<arm>[_ps<p>]_seed<S>[_test]   (p=0 -> no _ps suffix)

Reported per language: G at each pretraining seed, pooled G (= mean over pretraining seeds of the
per-seed G), and a 2-level hierarchical bootstrap (resample pretraining seeds, then fine-tune seeds
within each drawn pretraining seed; unpaired across arms, as in item5; a paired-pretraining-seed
variant is also given). Law refits on each pretraining seed separately and on the pooled values:
7-language OLS, calibration-5 OLS, calibration-5 LOO MAE / sign, prospective LSFB/LIBRAS sign from
the calibration-5 fit (=> the paper's x/7), all-7 LOO MAE / sign, plus a bootstrap distribution of
the pooled-law statistics.

  python tools/p1_law_pretrain_seed_analysis.py [--pseeds 0-6] [--seeds all|012] [--n-boot 20000] [--json out.json]
"""
import argparse, csv, os, re
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# same order / values as tools/gating_law_fit.py
TARGETS = ["BdSL", "LSA64", "KArSL", "AUTSL", "GSL", "LSFB", "LIBRAS"]
STEM = {"BdSL": "bdsl_block_gcn_si", "LSA64": "lsa64_si", "KArSL": "karsl_si", "AUTSL": "autsl_si",
        "GSL": "gsl_si", "LSFB": "lsfb_si", "LIBRAS": "libras_si"}
DIST = np.array([0.1429, 0.1763, 0.3445, 0.4115, 0.1206, 0.4798, 0.3151])
PROSPECTIVE = ["LSFB", "LIBRAS"]
CALIB = [i for i, t in enumerate(TARGETS) if t not in PROSPECTIVE]
PROSP = [TARGETS.index(t) for t in PROSPECTIVE]
TEST_LANGS = ["BdSL", "LSA64", "AUTSL"]          # P1 tab:test
PSEEDS = [0, 1, 2]          # default; --pseeds overrides (e.g. 0-6 for the 7-seed scale-up)
PAPER = dict(a7=4.78, b7=-24.78, r2_7=0.887, d7=0.193, a5=5.99, b5=-30.18, r2_5=0.997, d5=0.198,
             loo5=0.235, loo7=1.42, sign="7/7")

_SEED = re.compile(r"_seed(\d+)(_test)?$")


def load(csv_path):
    """{(base, seed, is_test): top1_pct}; val = best over rows, test = last row."""
    val, test = defaultdict(lambda: -1.0), {}
    for r in csv.reader(open(csv_path)):
        if len(r) < 4 or r[1] in ("", "Experiment"):
            continue
        m = _SEED.search(r[1])
        if not m:
            continue
        try:
            v = float(r[3])
        except ValueError:
            continue
        v = v * 100 if v <= 1.0 else v
        key = (r[1][:m.start()], int(m.group(1)))
        if m.group(2):
            test[key] = v
        else:
            val[key] = max(val[key], v)
    return dict(val), test


def base(t, arm, p):
    return f"{STEM[t]}_lr01_f10_{arm}" + ("" if p == 0 else f"_ps{p}")


def runs(store, t, arm, p, seeds):
    out = [store[(base(t, arm, p), s)] for s in seeds if (base(t, arm, p), s) in store]
    return np.array(out, float)


def linfit(d, g):
    b, a = np.polyfit(d, g, 1)
    pred = a + b * d
    r2 = 1 - np.sum((g - pred) ** 2) / np.sum((g - g.mean()) ** 2)
    return a, b, r2


def law_stats(G):
    """All the paper's law statistics for one vector of 7 gains (TARGETS order)."""
    G = np.asarray(G, float)
    a7, b7, r7 = linfit(DIST, G)
    a5, b5, r5 = linfit(DIST[CALIB], G[CALIB])
    from scipy.stats import spearmanr
    rho5 = spearmanr(DIST[CALIB], G[CALIB]).correlation
    # calibration-5 LOO
    e5, s5 = [], 0
    for i in CALIB:
        m = [j for j in CALIB if j != i]
        ai, bi, _ = linfit(DIST[m], G[m])
        p = ai + bi * DIST[i]
        e5.append(abs(p - G[i])); s5 += int(np.sign(p) == np.sign(G[i]))
    sp = sum(int(np.sign(a5 + b5 * DIST[i]) == np.sign(G[i])) for i in PROSP)
    # all-7 LOO
    e7, s7 = [], 0
    for i in range(len(TARGETS)):
        m = np.arange(len(TARGETS)) != i
        ai, bi, _ = linfit(DIST[m], G[m])
        p = ai + bi * DIST[i]
        e7.append(abs(p - G[i])); s7 += int(np.sign(p) == np.sign(G[i]))
    return dict(a7=a7, b7=b7, r2_7=r7, d7=-a7 / b7, a5=a5, b5=b5, r2_5=r5, d5=-a5 / b5, rho5=rho5,
                loo5=np.mean(e5), sign5=s5, signP=sp, sign_paper=s5 + sp, loo7=np.mean(e7), sign7=s7,
                prosp_pred={TARGETS[i]: a5 + b5 * DIST[i] for i in PROSP})


def fmt_stats(name, s):
    return (f"| {name} | {s['a7']:+.2f} {s['b7']:+.2f}·d | {s['r2_7']:.3f} | {s['d7']:.3f} | "
            f"{s['a5']:+.2f} {s['b5']:+.2f}·d | {s['r2_5']:.3f} | {s['d5']:.3f} | {s['rho5']:+.2f} | "
            f"{s['loo5']:.2f} | {s['sign5']}/5 + {s['signP']}/2 = **{s['sign_paper']}/7** | "
            f"{s['loo7']:.2f} | {s['sign7']}/7 |")


def boot_lang(X, P, n, rng, paired=False):
    """X, P: list over pretraining seeds of per-ft-seed arrays. Returns n draws of mean(X)-mean(P)."""
    k = len(X)
    def draw(arrs, pidx):
        out = np.zeros(n)
        for j in range(k):
            sel = pidx[:, j]
            acc = np.zeros(n)
            for p in range(k):
                hit = sel == p
                if not hit.any():
                    continue
                a = arrs[p]
                fi = rng.integers(0, a.size, size=(hit.sum(), a.size))
                acc[hit] = a[fi].mean(1)
            out += acc
        return out / k
    px = rng.integers(0, k, size=(n, k))
    pp = px if paired else rng.integers(0, k, size=(n, k))
    return draw(X, px) - draw(P, pp)


def parse_pseeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def collect(val, pseeds, seeds):
    """X[t], P[t]: list over pseeds of per-ft-seed arrays; G (n_pseed, 7) (nan if missing)."""
    G = np.full((len(pseeds), len(TARGETS)), np.nan)
    X, P = {}, {}
    for j, t in enumerate(TARGETS):
        X[t] = [runs(val, t, "xling", p, seeds) for p in pseeds]
        P[t] = [runs(val, t, "pool", p, seeds) for p in pseeds]
        for i in range(len(pseeds)):
            if X[t][i].size and P[t][i].size:
                G[i, j] = X[t][i].mean() - P[t][i].mean()
    return G, X, P


def pooled_gains(csv_path=None, pseeds=None, seeds=(0, 1, 2, 3, 4)):
    """Pooled (mean over pretraining seeds) G per TARGET; used by p1_law_baseline_refit.py."""
    val, _ = load(csv_path or os.path.join(ROOT, "results", "results_final.csv"))
    G, _, _ = collect(val, pseeds or PSEEDS, list(seeds))
    return G


def joint_law_boot(X, P, nb, rng):
    """Pretraining seeds drawn ONCE per draw and shared across the 7 languages (shared backbones;
    arms unpaired), then fine-tune seeds resampled within language x drawn pretraining seed."""
    k = len(X[TARGETS[0]]); out = []
    for _ in range(nb):
        kx, kp = rng.integers(0, k, k), rng.integers(0, k, k)
        g = [np.mean([rng.choice(X[t][i], X[t][i].size).mean() for i in kx]) -
             np.mean([rng.choice(P[t][i], P[t][i].size).mean() for i in kp]) for t in TARGETS]
        s = law_stats(np.array(g)); s["g"] = np.array(g); out.append(s)
    return out


def summarize_boot(S, title):
    def q(key):
        return np.percentile(np.array([s[key] for s in S], float), [2.5, 50, 97.5])
    print(f"\n### {title} ({len(S)} draws)")
    res = {}
    for key, lab in [("r2_7", "R²(7)"), ("b7", "slope(7)"), ("a7", "intercept(7)"), ("d7", "d*(7)"),
                     ("r2_5", "R²(5)"), ("b5", "slope(5)"), ("d5", "d*(5)"),
                     ("loo5", "LOO MAE(5)"), ("loo7", "LOO MAE(7)")]:
        lo, md, hi = q(key); res[key] = (lo, md, hi)
        print(f"  {lab:12s} median {md:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")
    sp = np.array([s["sign_paper"] for s in S]); s7 = np.array([s["sign7"] for s in S])
    print("  paper-protocol sign count: " + ", ".join(f"{c}/7: {np.mean(sp == c):.3f}" for c in range(7, -1, -1)))
    print("  all-7 LOO sign count:      " + ", ".join(f"{c}/7: {np.mean(s7 == c):.3f}" for c in range(7, -1, -1)))
    pneg = np.mean(np.array([s["b7"] for s in S]) < 0); pneg5 = np.mean(np.array([s["b5"] for s in S]) < 0)
    print(f"  P(slope(7) < 0) = {pneg:.4f}   P(slope(5) < 0) = {pneg5:.4f}   P(7/7 paper protocol) = {np.mean(sp == 7):.4f}")
    res.update(p_neg7=pneg, p_neg5=pneg5, p77=float(np.mean(sp == 7)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "results_final.csv"))
    ap.add_argument("--pseeds", default="0-2", help="pretraining seeds, e.g. 0-2 or 0-6 or 1-6")
    ap.add_argument("--seeds", choices=["all", "012"], default="all",
                    help="fine-tune seeds: all available (paper: 5 for GSL/LSFB/LIBRAS) or 0-2 only")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--n-law-boot", type=int, default=5000)
    ap.add_argument("--json", help="write pooled gains + stats here")
    a = ap.parse_args()
    pseeds = parse_pseeds(a.pseeds)
    val, test = load(a.csv)
    seeds = [0, 1, 2, 3, 4] if a.seeds == "all" else [0, 1, 2]
    rng = np.random.default_rng(0)
    G, X, P = collect(val, pseeds, seeds)
    nP = len(pseeds)
    print(f"# P1 law x pretraining seed  (pretraining seeds {pseeds}; fine-tune seeds: {a.seeds})\n")
    missing = [(pseeds[i], TARGETS[j]) for i in range(nP) for j in range(len(TARGETS)) if np.isnan(G[i, j])]
    if missing:
        print(f"PENDING cells (pseed, target): {missing}\n")

    # ---------------- per-language G ----------------
    print("## Per-language G = xling - pool (VAL, pp)\n")
    print("| target | MMD | " + " | ".join(f"ps{p}" for p in pseeds) +
          " | **pooled** | 95% hier CI | P(G>0) | paired-pseed CI | signs | n ft (pool/xling) |")
    print("|---|---|" + "---|" * nP + "---|---|---|---|---|---|")
    boots = {}
    for j, t in enumerate(TARGETS):
        ok = not np.isnan(G[:, j]).any()
        cells = " | ".join(f"{g:+.2f}" if not np.isnan(g) else "PEND" for g in G[:, j])
        signs = "".join("+" if g > 0 else ("-" if g < 0 else "?") for g in G[:, j])
        ns = " ".join(f"{P[t][i].size}/{X[t][i].size}" for i in range(nP))
        if ok:
            d = boot_lang(X[t], P[t], a.n_boot, rng); boots[t] = d
            dp = boot_lang(X[t], P[t], a.n_boot, rng, paired=True)
            lo, hi = np.percentile(d, [2.5, 97.5]); plo, phi = np.percentile(dp, [2.5, 97.5])
            tail = f"**{G[:, j].mean():+.2f}** | [{lo:+.2f}, {hi:+.2f}] | {np.mean(d > 0):.2f} | [{plo:+.2f}, {phi:+.2f}]"
        else:
            tail = "PEND | | | "
        print(f"| {t} | {DIST[j]:.3f} | {cells} | {tail} | {signs} | {ns} |")

    print("\n## Arm means per pretraining seed (VAL Top-1)\n")
    print("| target | arm | " + " | ".join(f"ps{p}" for p in pseeds) + " | mean | SD over pseeds |")
    print("|---|---|" + "---|" * nP + "---|---|")
    for t in TARGETS:
        for arm, D in (("pool", P), ("xling", X)):
            m = np.array([D[t][i].mean() if D[t][i].size else np.nan for i in range(nP)])
            print(f"| {t} | {arm} | " + " | ".join(f"{v:.2f}" for v in m) +
                  f" | {np.nanmean(m):.2f} | {np.nanstd(m, ddof=1):.2f} |")

    complete = not np.isnan(G).any()
    # ---------------- law refits ----------------
    print("\n## Law refits\n")
    print("| G source | 7-lang OLS | R²(7) | d*(7) | calib-5 OLS | R²(5) | d*(5) | ρ(5) | LOO MAE(5) | sign (paper protocol) | LOO MAE(7) | LOO sign(7) | frozen gate d*=0.193 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(f"| paper (P1) | {PAPER['a7']:+.2f} {PAPER['b7']:+.2f}·d | {PAPER['r2_7']:.3f} | {PAPER['d7']:.3f} | "
          f"{PAPER['a5']:+.2f} {PAPER['b5']:+.2f}·d | {PAPER['r2_5']:.3f} | {PAPER['d5']:.3f} | -1.00 | "
          f"{PAPER['loo5']:.2f} | 5/5 + 2/2 = **7/7** | {PAPER['loo7']:.2f} | 7/7 | 7/7 |")
    gate = lambda g: int(np.sum((DIST < PAPER["d7"]) == (np.asarray(g) > 0)))
    stats = {}
    for i, p in enumerate(pseeds):
        if not np.isnan(G[i]).any():
            stats[f"ps{p}"] = s = law_stats(G[i]); print(fmt_stats(f"pseed {p}", s) + f" {gate(G[i])}/7 |")
        else:
            print(f"| pseed {p} | PENDING |")
    out = {"pseeds": pseeds, "targets": TARGETS, "dist": DIST.tolist(), "G": G.tolist()}
    if complete:
        Gp = G.mean(0); stats["pooled"] = s = law_stats(Gp)
        print(fmt_stats(f"**pooled ({nP} pseeds)**", s) + f" {gate(Gp)}/7 |")
        if 0 in pseeds and nP > 2:
            rep = [i for i, p in enumerate(pseeds) if p != 0]
            Gr = G[rep].mean(0); stats["pooled_excl0"] = s = law_stats(Gr)
            print(fmt_stats(f"pooled, replication seeds only (excl. ps0, n={len(rep)})", s) + f" {gate(Gr)}/7 |")
        from scipy.stats import spearmanr, pearsonr
        print("\nSpearman / Pearson over all 7 (G vs MMD):")
        for k_, g in [(f"ps{p}", G[i]) for i, p in enumerate(pseeds)] + [("pooled", Gp)] + \
                     ([("pooled_excl0", G[[i for i, p in enumerate(pseeds) if p != 0]].mean(0))] if 0 in pseeds and nP > 2 else []):
            rs, ps_ = spearmanr(DIST, g); rp, pp_ = pearsonr(DIST, g)
            print(f"  {k_:13s} rho={rs:+.2f} (p={ps_:.2f})  r={rp:+.2f} (p={pp_:.2f})")
        for k_, s in stats.items():
            print(f"  [{k_}] prospective calib-5 predictions: " +
                  ", ".join(f"{t} pred {v:+.2f}" for t, v in s["prosp_pred"].items()))
        out["G_pooled"] = Gp.tolist()

        # ---------------- bootstrap of the law ----------------
        if len(boots) == len(TARGETS):
            nb = min(a.n_boot, a.n_law_boot)
            SJ = joint_law_boot(X, P, nb, rng)
            out["joint_boot"] = summarize_boot(SJ, "Pooled-law bootstrap, JOINT pretraining-seed draws (primary)")
            from scipy.stats import spearmanr as _sr
            gb = np.stack([s["g"] for s in SJ])
            nr = [TARGETS.index(t) for t in ("GSL", "BdSL", "LSA64")]; fr = [TARGETS.index(t) for t in ("LIBRAS", "KArSL", "AUTSL", "LSFB")]
            cb = gb[:, nr].mean(1) - gb[:, fr].mean(1)
            rb = np.array([_sr(DIST, g).correlation for g in gb])
            lo, hi = np.percentile(cb, [2.5, 97.5]); rlo, rhi = np.percentile(rb, [2.5, 97.5])
            print(f"  near(3)-far(4) contrast: median {np.median(cb):+.2f}  95% [{lo:+.2f}, {hi:+.2f}]  P(>0)={np.mean(cb > 0):.3f}")
            print(f"  Spearman rho(7):         median {np.median(rb):+.2f}  95% [{rlo:+.2f}, {rhi:+.2f}]  P(<0)={np.mean(rb < 0):.3f}")
            out["joint_boot"].update(contrast_ci=(lo, hi), p_contrast_pos=float(np.mean(cb > 0)), rho_ci=(rlo, rhi), p_rho_neg=float(np.mean(rb < 0)))
            B = np.stack([boots[t] for t in TARGETS], 1)
            SI = [law_stats(B[k]) for k in range(nb)]
            out["indep_boot"] = summarize_boot(SI, "Pooled-law bootstrap, languages resampled independently")
            print("  per-language P(G>0): " + ", ".join(f"{t} {np.mean(boots[t] > 0):.2f}" for t in TARGETS))

        # ---------------- variance components ----------------
        print("\n## Variance components of G per language (pp)\n")
        print("| target | SD of G over pseeds | mean within-pseed fine-tune SE | between-pseed SD component | ratio SD/SE |")
        print("|---|---|---|---|---|")
        sds, ses, comps = [], [], []
        for j, t in enumerate(TARGETS):
            sd = np.std(G[:, j], ddof=1)
            se2 = np.array([np.var(X[t][i], ddof=1) / X[t][i].size + np.var(P[t][i], ddof=1) / P[t][i].size
                            for i in range(nP)])
            comp = np.sqrt(max(sd ** 2 - se2.mean(), 0.0))
            sds.append(sd); ses.append(np.sqrt(se2).mean()); comps.append(comp)
            print(f"| {t} | {sd:.2f} | {np.sqrt(se2).mean():.2f} | {comp:.2f} | {sd / np.sqrt(se2).mean():.1f}x |")
        print(f"| mean | {np.mean(sds):.2f} | {np.mean(ses):.2f} | {np.mean(comps):.2f} | {np.mean(sds) / np.mean(ses):.1f}x |")
        print(f"  cross-language SD of pooled G = {np.std(Gp, ddof=1):.2f}pp (seed-0 alone: {np.std(G[pseeds.index(0)], ddof=1) if 0 in pseeds else float('nan'):.2f}pp)")
        print(f"  SE of a pooled G from pseed noise alone ~ {np.mean(comps) / np.sqrt(nP):.2f}pp")
        # shared backbone factor: correlation of per-pseed G across languages
        C = np.corrcoef(G.T)
        off = C[np.triu_indices(len(TARGETS), 1)]
        print(f"  mean pairwise cross-language correlation of per-pseed G = {off.mean():+.2f} (range {off.min():+.2f}..{off.max():+.2f})")
        near = [TARGETS.index(t) for t in ("GSL", "BdSL", "LSA64")]; far = [TARGETS.index(t) for t in ("LIBRAS", "KArSL", "AUTSL", "LSFB")]
        contrast = G[:, near].mean(1) - G[:, far].mean(1)
        print(f"  near(3) - far(4) mean-G contrast per pseed: {np.round(contrast, 2).tolist()}  mean {contrast.mean():+.2f}"
              f"  (positive = law direction; {int(np.sum(contrast > 0))}/{nP} pseeds positive)")
        out["variance"] = {"sd_G": sds, "ft_se": ses, "between_comp": comps}
        out["near_far_contrast"] = contrast.tolist()
        if nP > 2:
            from scipy.stats import ttest_1samp
            tt = ttest_1samp(contrast, 0.0)
            print(f"  one-sample t over pseeds (contrast > 0?): t={tt.statistic:+.2f}, two-sided p={tt.pvalue:.3f}"
                  f"  mean {contrast.mean():+.2f} +/- SE {contrast.std(ddof=1) / np.sqrt(nP):.2f}")
            rep = [i for i, p in enumerate(pseeds) if p != 0]
            if len(rep) > 1 and len(rep) < nP:
                tr = ttest_1samp(contrast[rep], 0.0)
                print(f"  replication seeds only (excl. ps0): mean {contrast[rep].mean():+.2f}, t={tr.statistic:+.2f}, p={tr.pvalue:.3f}")
        out["stats"] = {k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in stats.items()}

    # ---------------- TEST ----------------
    print("\n## Held-out TEST G_test = xling - pool (P1 tab:test languages)\n")
    print("| target | " + " | ".join(f"ps{p}" for p in pseeds) + " | pooled | 95% hier CI | P(G>0) |")
    print("|---|" + "---|" * nP + "---|---|---|")
    for t in TEST_LANGS:
        Xt = [np.array([test[(base(t, "xling", p), s)] for s in [0, 1, 2] if (base(t, "xling", p), s) in test]) for p in pseeds]
        Pt = [np.array([test[(base(t, "pool", p), s)] for s in [0, 1, 2] if (base(t, "pool", p), s) in test]) for p in pseeds]
        cells = " | ".join(f"{x.mean() - q.mean():+.2f}" if x.size and q.size else "PEND" for x, q in zip(Xt, Pt))
        if all(v.size for v in Xt + Pt):
            gp = np.mean([x.mean() - q.mean() for x, q in zip(Xt, Pt)])
            d = boot_lang(Xt, Pt, a.n_boot, rng); lo, hi = np.percentile(d, [2.5, 97.5])
            tail = f"**{gp:+.2f}** | [{lo:+.2f}, {hi:+.2f}] | {np.mean(d > 0):.2f}"
        else:
            tail = "PEND | |"
        print(f"| {t} | {cells} | {tail} |")

    if a.json:
        import json
        def conv(o):
            if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [conv(v) for v in o]
            if isinstance(o, (np.floating, np.integer)): return o.item()
            return o
        json.dump(conv(out), open(a.json, "w"), indent=1)
        print(f"\n[json] {a.json}")


if __name__ == "__main__":
    main()
