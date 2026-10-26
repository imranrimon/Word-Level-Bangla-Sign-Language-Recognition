#!/usr/bin/env python
"""Paper figures (Figs. 1-4) from the archived analysis outputs.

Reads ONLY existing analysis outputs (no training, no results_final.csv parsing):
  docs/p1_refit_2026-10/refit_results.json     pre-registered 7-pretraining-seed refit (v1 backbones)
  docs/p1_fix_2026-10/step2/distances_v2.json  v1 -> v2 distance decomposition (Addendum A source)
  docs/p1_fix_2026-10/results/refit_v2.json    pre-registered v2 analysis (tools/p1_fix/refit_v2.py)
  docs/p1_refit_2026-10/pooled_gains.json      v1 per-pseed G (for the v1-vs-v2 comparison)

Writes (into results/figures/, or --out-dir):
  fig_seedlaws.pdf      (a) the published single-run law (pseed 0); (b) all 7 pretraining seeds
  fig_variance.pdf      (a) G heat map language x pseed; (b) method-of-moments variance components
  fig_distance_steps.pdf  slope chart: v1 paper distance -> frozen v2 distance per language
  fig_v2law.pdf         (a) v2 G vs d^v2: cells per pretraining seed, per-seed OLS, pooled G + 95% CI,
                        posterior-mean line; (b) slope b with 95% interval, primary + sensitivities

Also prints every number the captions quote, and asserts they match the docs.
  python tools/paper/make_figs.py            # all figures
  python tools/paper/make_figs.py --only v2  # only fig_v2law.pdf + the derived v2 numbers
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.environ.get("P1_FIG_DIR", os.path.join(ROOT, "results", "figures"))   # output directory
REFIT = os.path.join(ROOT, "docs/p1_refit_2026-10/refit_results.json")
DIST = os.path.join(ROOT, "docs/p1_fix_2026-10/step2/distances_v2.json")
REFIT_V2 = os.path.join(ROOT, "docs/p1_fix_2026-10/results/refit_v2.json")
V1_GAINS = os.path.join(ROOT, "docs/p1_refit_2026-10/pooled_gains.json")

# palette (validated reference instance; blue = slot 1, red = diverging negative pole)
BLUE, RED, MID, INK, MUTED = "#2a78d6", "#e34948", "#f0efec", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.labelcolor": INK, "pdf.fonttype": 42})


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * np.asarray(x)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return a, b, r2


def fig_seedlaws(R):
    T, d = R["targets"], np.array(R["dist"])
    G = np.array(R["descriptive"]["G"])          # [pseed, target]
    PL = R["descriptive"]["per_lang"]
    order = np.argsort(d)
    xs = np.linspace(0.10, 0.50, 50)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    # (a) published single run
    ax = axs[0]
    a, b, r2 = ols(d, G[0])
    assert abs(r2 - R["descriptive"]["per_pseed"]["0"]["r2_7"]) < 1e-6
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(xs, a + b * xs, color=BLUE, lw=1.5)
    ax.scatter(d, G[0], s=22, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
    for i in order:
        off = (-30, -9) if T[i] == "LIBRAS" else (3, 3)
        ax.annotate(T[i], (d[i], G[0][i]), xytext=off, textcoords="offset points", fontsize=6.5, color="#52514e")
    ax.set_title(f"(a) pretraining seed 0 (as published)\nOLS slope {b:.1f}, $R^2$={r2:.2f}", fontsize=8)
    ax.set_xlabel("v1 pose-motion MMD $d_\\ell$")
    ax.set_ylabel("G = xling $-$ pool (pp, VAL @10%)")
    # (b) all seeds
    ax = axs[1]
    ax.axhline(0, color=MUTED, lw=0.6)
    for p in range(1, G.shape[0]):
        a_, b_, _ = ols(d, G[p])
        ax.plot(xs, a_ + b_ * xs, color="#b5b4ae", lw=0.9)
        ax.scatter(d, G[p], s=7, color="#b5b4ae", zorder=2)
    ax.plot(xs, a + b * xs, color=BLUE, lw=1.5, label="seed 0 (published)")
    pooled = np.array([PL[t]["pooled"] for t in T])
    lo = np.array([PL[t]["ci"][0] for t in T])
    hi = np.array([PL[t]["ci"][1] for t in T])
    ax.errorbar(d, pooled, yerr=[pooled - lo, hi - pooled], fmt="D", ms=4, color=INK, ecolor=INK, elinewidth=0.8,
                capsize=2, zorder=4, label="pooled over 7 seeds (95% CI)")
    ax.plot([], [], color="#b5b4ae", lw=0.9, label="seeds 1–6 (per-seed OLS)")
    ax.set_title("(b) all 7 pretraining seeds", fontsize=8)
    ax.set_xlabel("v1 pose-motion MMD $d_\\ell$")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_seedlaws.pdf"))
    plt.close(fig)
    print("[seedlaws] seed0 a,b,R2 =", round(a, 2), round(b, 2), round(r2, 3))
    for p in range(G.shape[0]):
        a_, b_, r_ = ols(d, G[p])
        print(f"  pseed {p}: slope {b_:+.2f} R2 {r_:.3f}")
    print("  pooled G:", {t: round(PL[t]['pooled'], 2) for t in T})


def fig_variance(R):
    T, d = R["targets"], np.array(R["dist"])
    G = np.array(R["descriptive"]["G"])  # [P, L]
    P, L = G.shape
    # two-way ANOVA, one observation per cell, method of moments
    gm = G.mean()
    ms_lang = P * np.sum((G.mean(0) - gm) ** 2) / (L - 1)
    ms_ps = L * np.sum((G.mean(1) - gm) ** 2) / (P - 1)
    resid = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + gm
    ms_res = np.sum(resid ** 2) / ((L - 1) * (P - 1))
    s2bar = float(np.mean([c["s2"] for c in R["cells"]]))
    v_lang = max(0.0, (ms_lang - ms_res) / P)
    v_ps = max(0.0, (ms_ps - ms_res) / L)
    v_int = max(0.0, ms_res - s2bar)
    comps = [("language\n(main effect)", v_lang), ("pretraining seed\n(shared)", v_ps),
             ("language ×\npretraining seed", v_int), ("fine-tune\nnoise", s2bar)]
    tot = sum(v for _, v in comps)
    print("[variance] MoM pp^2:", {k.replace(chr(10), ' '): round(v, 2) for k, v in comps},
          "MS_res", round(ms_res, 2), "shares", [round(v / tot, 3) for _, v in comps])
    # documented values: RESULTS.md (lang 0.65, pseed 1.91, residual MS 10.87), ckpt README (ft 6.6, int 4.3)
    assert abs(v_lang - 0.65) < 0.01 and abs(v_ps - 1.91) < 0.01 and abs(ms_res - 10.87) < 0.01, (v_lang, v_ps, ms_res)
    assert abs(s2bar - 6.616) < 0.01

    order = np.argsort(d)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    lim = np.ceil(np.abs(G).max())
    im = ax.imshow(G[:, order].T, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"{T[i]} ({d[i]:.2f})" for i in order], fontsize=6.5)
    ax.set_xticks(range(P))
    ax.set_xticklabels([str(p) for p in R["pseeds"]], fontsize=6.5)
    ax.set_xlabel("pretraining seed")
    ax.set_title("(a) G (pp) per language × pretraining seed", fontsize=8)
    for i_, i in enumerate(order):
        for p in range(P):
            ax.text(p, i_, f"{G[p, i]:+.1f}", ha="center", va="center", fontsize=5.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    ax = axs[1]
    names = [k for k, _ in comps]
    vals = [v for _, v in comps]
    cols = ["#9ec5f4", "#5598e7", "#256abf", "#b5b4ae"]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.12, i, f"{v:.2f} pp² ({100 * v / tot:.0f}%)", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.6)
    ax.set_xlabel("variance of single-run G (pp²)")
    ax.set_title("(b) variance components (MoM)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_variance.pdf"))
    plt.close(fig)


V1 = {"GSL": 0.1206, "BdSL": 0.1429, "LSA64": 0.1763, "LIBRAS": 0.3151, "KArSL": 0.3445, "AUTSL": 0.4115,
      "LSFB": 0.4798, "BISINDO": 0.4495}


def _dodge(ys, gap):
    """Return label y-positions >= gap apart, preserving order, as close to ys as possible."""
    idx = np.argsort(ys)
    pos = np.array(ys, float)[idx]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            if pos[k] - pos[k - 1] < gap:
                m = (pos[k] + pos[k - 1]) / 2
                pos[k - 1], pos[k] = m - gap / 2, m + gap / 2
                moved = True
        if not moved:
            break
    out = np.empty(len(ys))
    out[idx] = pos
    return out


def fig_distance_steps():
    """Slope chart v1 (paper) -> v2 (Addendum A); intermediate steps are tabulated in the appendix."""
    Rd = json.load(open(DIST))["results"]
    langs = ["BdSL", "GSL", "LSA64", "LIBRAS", "KArSL", "AUTSL", "LSFB", "BISINDO"]
    v2 = {t: Rd["v2"]["pool_v2"][t]["pooled"]["mean"] for t in langs}
    addA = {"BdSL": 0.0810, "LSA64": 0.1572, "AUTSL": 0.1557, "KArSL": 0.4041, "GSL": 0.0913, "LSFB": 0.1447,
            "LIBRAS": 0.1373, "BISINDO": 0.3959}  # PREREG_addendum_A_distances.md
    for t in langs:
        assert abs(v2[t] - addA[t]) < 5e-5, (t, v2[t], addA[t])
    grp = {t: ("near in both" if V1[t] < 0.193 else ("far in both" if v2[t] > 0.3 else "far in v1, near in v2"))
           for t in langs}
    col = {"near in both": "#8a8984", "far in v1, near in v2": BLUE, "far in both": RED}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for t in langs:
        ax.plot([0, 1], [V1[t], v2[t]], color=col[grp[t]], lw=1.5, marker="o", ms=3.5)
    yl = _dodge([V1[t] for t in langs], 0.022)
    yr = _dodge([v2[t] for t in langs], 0.022)
    for t, a_, b_ in zip(langs, yl, yr):
        ax.text(-0.04, a_, f"{t} {V1[t]:.3f}", ha="right", va="center", fontsize=6.5, color=INK)
        ax.text(1.04, b_, f"{v2[t]:.3f} {t}", ha="left", va="center", fontsize=6.5, color=INK)
    for g, c in col.items():
        ax.plot([], [], color=c, lw=1.5, label=g)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["v1 (paper)", "v2 (pre-registered)"], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_distance_steps.pdf"))
    plt.close(fig)
    print("[distance] v2 =", {t: round(v, 4) for t, v in v2.items()}, "groups", grp)


# ---------------------------------------------------------------------------------------------- v2
V2_LANGS = ["BdSL", "GSL", "LIBRAS", "LSFB", "AUTSL", "LSA64", "BISINDO", "KArSL"]  # ordered by d^v2
# Values quoted in docs/p1_fix_2026-10/RESULTS.md and the paper, Sec. 7.4; asserted against refit_v2.json below.
V2_DOC = {"b": (-11.56, -4.59, 2.61), "a": (-3.87, -0.27, 3.33), "P_b_neg": 0.905, "lolo": (6, 8),
          "pooled": {"BdSL": 1.51, "LSA64": -1.90, "AUTSL": -2.40, "KArSL": -1.54, "GSL": -3.18, "LSFB": -1.72,
                     "LIBRAS": 1.18, "BISINDO": -2.13},
          "pseed_slopes": (-0.49, -8.80, -0.95), "pseed_r2": (0.001, 0.203, 0.003),
          "S8": (1.61, 12.93, 24.47), "S9": (-18.43, -10.15, -1.97)}


def v2_numbers(R2):
    """Derived (descriptive, not pre-registered) v2 numbers quoted in the paper; asserts the quoted values."""
    d, C, P = R2["distances_primary"], R2["cells"], R2["primary"]
    assert P["verdict"] == "NOT SUPPORTED" and not P["C1_CI_below_0"] and P["lolo"]["C2"]
    for k in ("b", "a"):
        assert all(abs(x - y) < 0.006 for x, y in zip(P[k], V2_DOC[k])), (k, P[k])
    assert abs(P["P_b_neg"] - V2_DOC["P_b_neg"]) < 5e-4
    assert (P["lolo"]["correct"], P["lolo"]["n"]) == V2_DOC["lolo"]
    # every LOLO prediction is negative -> C2 equals the score of the constant "G<0" rule
    assert all(v["pred"] < 0 for v in P["lolo"]["per_lang"].values())
    for t, g in V2_DOC["pooled"].items():
        assert abs(R2["pooled_G"][t] - g) < 0.006, t
    assert sum(c["nx"] + c["np"] for c in C) == 239 and len(R2["exclusions"]) == 1
    out = {"pseed": {}}
    x = np.array([d[t] for t in V2_LANGS])
    for p in R2["primary"]["pseeds"]:
        y = np.array([next(c["G"] for c in C if c["t"] == t and c["p"] == p) for t in V2_LANGS])
        a_, b_, r_ = ols(x, y)
        out["pseed"][p] = (a_, b_, r_)
        print(f"[v2] pseed {p}: OLS slope {b_:+.2f} intercept {a_:+.2f} R2 {r_:.3f}; negative cells {(y < 0).sum()}/8")
        assert abs(b_ - V2_DOC["pseed_slopes"][p]) < 0.006 and abs(r_ - V2_DOC["pseed_r2"][p]) < 6e-4
    # t-interval over the 3 pretraining-seed cell gains (the bootstrap over 3 pseeds is anti-conservative),
    # and Holm adjustment over the 8 languages (t-test p; bootstrap two-sided p = 2 min(P(G<0), P(G>0)))
    from scipy import stats
    tq = stats.t.ppf(0.975, 2)
    p_t, p_b = {}, {}
    for t in V2_LANGS:
        g = np.array([c["G"] for c in sorted(C, key=lambda c: c["p"]) if c["t"] == t])
        se = g.std(ddof=1) / np.sqrt(g.size)
        p_t[t] = 2 * stats.t.sf(abs(g.mean() / se), g.size - 1)
        pn = R2["boot_lang"][t]["P_G_neg"]
        p_b[t] = 2 * min(pn, 1 - pn)
        lo_, hi_ = g.mean() - tq * se, g.mean() + tq * se
        print(f"[v2] {t:7s} pooled {g.mean():+.2f}  t-CI [{lo_:+.2f}, {hi_:+.2f}]  p_t {p_t[t]:.3f}  "
              f"boot CI {R2['boot_lang'][t]['CI'][0]:+.2f},{R2['boot_lang'][t]['CI'][1]:+.2f}")
        assert lo_ < 0 < hi_  # no t-interval excludes zero

    def holm(pv):
        ks = sorted(pv, key=pv.get)
        out_, run = {}, 0.0
        for k, key in enumerate(ks):
            run = max(run, (len(ks) - k) * pv[key])
            out_[key] = min(1.0, run)
        return out_
    ht, hb = holm(p_t), holm(p_b)
    print(f"[v2] Holm (8 languages): min adjusted p, t-test {min(ht.values()):.2f}; bootstrap {min(hb.values()):.2f}")
    assert min(ht.values()) > 0.05 and min(hb.values()) > 0.05 and abs(min(hb.values()) - 0.23) < 0.005
    yp = np.array([R2["pooled_G"][t] for t in V2_LANGS])
    a_, b_, r_ = ols(x, yp)
    rank = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rank(x), rank(yp))[0, 1]   # no ties in d or G
    print(f"[v2] pooled-G OLS slope {b_:+.2f} intercept {a_:+.2f} R2 {r_:.3f}; Spearman(d, pooled G) {rho:+.3f}")
    assert abs(b_ + 3.42) < 0.006 and abs(r_ - 0.068) < 6e-4 and abs(rho + 0.190) < 6e-4
    # v1 vs v2 per language (v1: 7-pseed refit; pooled_gains.json G[pseed][target])
    V1g = json.load(open(V1_GAINS))
    G1 = np.array(V1g["G"])
    print("[v2] v1-vs-v2 G (pp): lang | v1 ps0 | v1 ps0-2 | v1 ps0-6 | v2 ps0-2 | v2 - v1(ps0-2)")
    diffs = {}
    for i, t in enumerate(V1g["targets"]):
        v1_02 = G1[:3, i].mean()
        diffs[t] = R2["pooled_G"][t] - v1_02
        print(f"   {t:7s} {G1[0, i]:+6.2f} {v1_02:+6.2f} {G1[:, i].mean():+6.2f} {R2['pooled_G'][t]:+6.2f} {diffs[t]:+6.2f}")
    common = V1g["targets"]
    m1 = np.mean([G1[:3, i].mean() for i in range(len(common))])
    m1all = G1.mean()
    m2 = np.mean([R2["pooled_G"][t] for t in common])
    print(f"   mean over the 7 v1 languages: v1 ps0-2 {m1:+.2f}, v1 ps0-6 {m1all:+.2f}, v2 {m2:+.2f}")
    assert abs(m1 + 0.36) < 0.006 and abs(m1all - 0.18) < 0.006 and abs(m2 + 1.15) < 0.006
    out["diffs"] = diffs
    return out


def fig_v2law(R2):
    d, C, P, S = R2["distances_primary"], R2["cells"], R2["primary"], R2["sensitivity"]
    der = v2_numbers(R2)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1.3, 1]})
    # (a) G vs d^v2
    ax = axs[0]
    ax.axhline(0, color=MUTED, lw=0.6)
    xs = np.linspace(0.06, 0.42, 50)
    marks = {0: "o", 1: "s", 2: "^"}
    greys = {0: "#9a9993", 1: "#b5b4ae", 2: "#cfcec8"}
    for p in P["pseeds"]:
        a_, b_, _ = der["pseed"][p]
        ax.plot(xs, a_ + b_ * xs, color=greys[p], lw=0.9, zorder=1)
        ax.scatter([d[c["t"]] for c in C if c["p"] == p], [c["G"] for c in C if c["p"] == p], s=9,
                   marker=marks[p], color=greys[p], zorder=2, label=f"pseed {p} cells + OLS ({b_:+.1f})")
    # posterior-mean line a + b d (linear, so the two end points define it exactly)
    pm = P["P_G_neg"]
    dl, dh = pm["d_min"]["d"], pm["d_max"]["d"]
    ml, mh = pm["d_min"]["mean"], pm["d_max"]["mean"]
    mid = pm["d_mid"]
    assert abs(ml + (mh - ml) * (mid["d"] - dl) / (dh - dl) - mid["mean"]) < 1e-6
    ax.plot([dl, dh], [ml, mh], color=BLUE, lw=1.6, zorder=3, label="posterior mean $a+b\,d$")
    pooled = np.array([R2["pooled_G"][t] for t in V2_LANGS])
    lo = np.array([R2["boot_lang"][t]["CI"][0] for t in V2_LANGS])
    hi = np.array([R2["boot_lang"][t]["CI"][1] for t in V2_LANGS])
    x = np.array([d[t] for t in V2_LANGS])
    ax.errorbar(x, pooled, yerr=[pooled - lo, hi - pooled], fmt="D", ms=3.8, color=INK, ecolor=INK,
                elinewidth=0.8, capsize=2, zorder=4, label="pooled G (95% CI)")
    # label positions in data coordinates (thin leader line where the label sits away from its point)
    lab = {"BdSL": (0.058, 5.2), "GSL": (0.097, -3.6), "LIBRAS": (0.112, 5.2), "LSFB": (0.098, -6.9),
           "AUTSL": (0.168, -3.4), "LSA64": (0.168, -1.2), "BISINDO": (0.312, -3.0), "KArSL": (0.385, 2.2)}
    for t, xi, yi in zip(V2_LANGS, x, pooled):
        lead = dict(arrowstyle="-", color="#b5b4ae", lw=0.5) if t in ("BdSL", "LIBRAS", "LSFB") else None
        ax.annotate(t, (xi, yi), xytext=lab[t], textcoords="data", fontsize=6.2, color="#52514e", arrowprops=lead)
    ax.set_xlim(0.05, 0.44)
    ax.set_xlabel("v2 pose-motion MMD$^2_u$ $d^{\mathrm{v2}}_\ell$ (pre-registered)")
    ax.set_ylabel("G = xling $-$ pool (pp, VAL @10%)")
    ax.set_title("(a) v2: G vs. corrected distance (3 pseeds x 5 ft seeds)", fontsize=8)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=1, handlelength=1.6)
    ax.set_ylim(-8.2, 6.8)
    # (b) slope forest
    ax = axs[1]
    rows = [("primary (binding)", P["b"], P), ("S1 no exclusions", S["S1_no_exclusions"]["b"], S["S1_no_exclusions"]),
            ("S2 + language effect", S["S2_language_RE"]["b"], S["S2_language_RE"]),
            ("S4 gap interpolation", S["S4_v2_interp"]["b"], S["S4_v2_interp"]),
            ("S5 v1-style source", S["S5_v2_desc_oldsrc"]["b"], S["S5_v2_desc_oldsrc"]),
            ("S6 drop KArSL, BISINDO", S["S6_drop_near_floor"]["b"], S["S6_drop_near_floor"]),
            ("S7 MixedLM (Wald)", [S["S7_mixedlm"]["b_CI"][0], S["S7_mixedlm"]["b"], S["S7_mixedlm"]["b_CI"][1]], None),
            ("S8 TEST signers", S["S8_test_split"]["b"], S["S8_test_split"]),
            ("S9 ft seeds 0-2", S["S9_ftseeds_012"]["b"], S["S9_ftseeds_012"])]
    assert all(abs(u - v) < 0.006 for u, v in zip(S["S8_test_split"]["b"], V2_DOC["S8"]))
    assert all(abs(u - v) < 0.006 for u, v in zip(S["S9_ftseeds_012"]["b"], V2_DOC["S9"]))
    assert S["S9_ftseeds_012"]["verdict"] == "SUPPORTED" and S["S8_test_split"]["b"][0] > 0
    ax.axvline(0, color=MUTED, lw=0.6)
    XL = (-45, 30)
    for i, (name, (l, m, h), res) in enumerate(rows):
        col = BLUE if i == 0 else INK
        ax.plot([max(l, XL[0]), min(h, XL[1])], [i, i], color=col, lw=1.1)
        if l < XL[0]:
            ax.annotate("", (XL[0], i), (XL[0] + 4, i), arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
        ax.plot(m, i, "o", ms=3.6, color=col)
        tag = "--" if res is None else f"{res['lolo']['correct']}/{res['lolo']['n']}"
        c1 = "" if res is None else (" C1" if res["C1_CI_below_0"] else "")
        ax.text(XL[1] + 1.5, i, f"{tag}{c1}", va="center", fontsize=6.0, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(*XL)
    ax.set_xlabel("slope $b$ (pp per unit $d^{\mathrm{v2}}$), 95% interval")
    ax.set_title("(b) slope: primary and sensitivities\n(right: LOLO sign; \"C1\" = interval below 0)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_v2law.pdf"))
    if os.environ.get("P1_FIG_PNG"):
        fig.savefig(os.environ["P1_FIG_PNG"], dpi=200)
    plt.close(fig)
    print("[v2law] primary b", [round(v, 2) for v in P["b"]], "a", [round(v, 2) for v in P["a"]],
          "P(b<0)", round(P["P_b_neg"], 3), "LOLO", P["lolo"]["correct"], "/", P["lolo"]["n"])


if __name__ == "__main__":
    import sys
    if "--out-dir" in sys.argv:
        HERE = sys.argv[sys.argv.index("--out-dir") + 1]
    os.makedirs(HERE, exist_ok=True)
    if "--only" not in sys.argv or sys.argv[sys.argv.index("--only") + 1] != "v2":
        R = json.load(open(REFIT))
        fig_seedlaws(R)
        fig_variance(R)
        fig_distance_steps()
    fig_v2law(json.load(open(REFIT_V2)))
