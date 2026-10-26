#!/usr/bin/env python
"""P1 distance-gating law: PRE-REGISTERED 7-pretraining-seed refit.

Implements docs/p1_refit_2026-10/PREREG.md exactly (written 2026-10-22 04:50 EDT, sha256 in
docs/p1_refit_2026-10/PREREG.sha256). Primary model:

    G_{l,p} ~ N(a + b d_l + u_p + v_l + w_{l,p}, s2_{l,p})      (cell gain, xling - pool, VAL @10%)
    u_p ~ N(0, su^2)  v_l ~ N(0, sv^2)  w_{l,p} ~ N(0, sw^2)     s2_{l,p} = plug-in fine-tune variance
    a ~ N(0, 20^2), b ~ N(0, 50^2), su, sv, sw ~ HalfNormal(5)

(a, b, u, v, w) marginalised analytically; posterior over (su, sv, sw) by deterministic grid
quadrature (61^3 on [0, 20] pp). Decision: KEEP-LAW iff P(b<0) > 0.95 AND >= 6/7 leave-one-language-out
sign predictions correct on pseed-pooled G; else REFRAME.

  python tools/p1_refit/refit.py --pseeds 0-6 --out docs/p1_refit_2026-10
"""
import argparse, json, os, subprocess, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")   # batched tiny linalg: 1 thread is ~100x faster than oversubscribed BLAS
from collections import defaultdict
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import p1_law_pretrain_seed_analysis as A  # noqa: E402  (same parser / names / d values as item6)

TARGETS, DIST, STEM = A.TARGETS, A.DIST, A.STEM
DX = DIST.copy()   # x-axis used by the Bayesian fit / LOLO (sensitivity runs may override)
NCLASS = {"BdSL": 60, "LSA64": 64, "KArSL": 70, "AUTSL": 226, "GSL": 310, "LSFB": 60, "LIBRAS": 20}
FT_SEEDS = {t: ([0, 1, 2, 3, 4] if t in ("GSL", "LSFB", "LIBRAS") else [0, 1, 2]) for t in TARGETS}
D_PRED = [0.15, 0.30, 0.45]


# ----------------------------------------------------------------------------- data
_COMPLETED = None


def _completed_list():
    """results/completed_runs.txt: runs whose final and best checkpoints existed when the results were frozen."""
    global _COMPLETED
    if _COMPLETED is None:
        p = os.path.join(ROOT, "results", "completed_runs.txt")
        _COMPLETED = set()
        if os.path.exists(p):
            _COMPLETED = {ln.strip() for ln in open(p) if ln.strip() and not ln.startswith("#")}
    return _COMPLETED


def complete(exp):
    """Run reached its final epoch: work_dir/<exp> has <exp>_model-best.pt and <exp>_model-99.pt.
    Without a local work_dir/<exp>, fall back to the archived list results/completed_runs.txt."""
    w = os.path.join(ROOT, "work_dir", exp)
    if not os.path.isdir(w):
        return exp in _completed_list()
    return os.path.exists(os.path.join(w, f"{exp}_model-best.pt")) and \
        os.path.exists(os.path.join(w, f"{exp}_model-99.pt"))


def exp_name(base, f):
    return f"{base}_seed{f}"


def gather(val, test, pseeds, seed_mode="all", rule="primary"):
    """Return runs[(t, p, arm)] = list of dicts(f, val, test, excl) and an exclusion log.
    rule: 'primary' (incomplete + collapse + gross), 'S1' (incomplete only)."""
    runs, log = defaultdict(list), []
    for t in TARGETS:
        seeds = FT_SEEDS[t] if seed_mode == "all" else [0, 1, 2]
        for p in pseeds:
            for arm in ("pool", "xling"):
                b = A.base(t, arm, p)
                for f in seeds:
                    e = exp_name(b, f)
                    v = val.get((b, f))
                    if v is None or not complete(e):
                        log.append((t, p, arm, f, "incomplete/missing", v)); continue
                    runs[(t, p, arm)].append(dict(f=f, val=v, test=test.get((b, f))))
    if rule == "primary":
        for t in TARGETS:
            allv = [r["val"] for (tt, _, _), rs in runs.items() if tt == t for r in rs]
            if not allv:
                continue
            med, chance3 = np.median(allv), 3 * 100.0 / NCLASS[t]
            for key in [k for k in runs if k[0] == t]:
                keep = []
                for r in runs[key]:
                    if r["val"] <= chance3:
                        log.append((t, key[1], key[2], r["f"], f"collapsed (<= 3x chance {chance3:.2f})", r["val"]))
                    elif r["val"] < 0.5 * med:
                        log.append((t, key[1], key[2], r["f"], f"gross failure (< 0.5 x median {med:.2f})", r["val"]))
                    else:
                        keep.append(r)
                runs[key] = keep
    return runs, log


def cells(runs, pseeds):
    """Cell gains G_{l,p} and plug-in s2_{l,p} (within-cell variance pooled over pseeds per lang x arm)."""
    S2 = {}
    for t in TARGETS:
        for arm in ("pool", "xling"):
            num = den = 0.0
            for p in pseeds:
                v = np.array([r["val"] for r in runs.get((t, p, arm), [])])
                if v.size >= 2:
                    num += ((v - v.mean()) ** 2).sum(); den += v.size - 1
            S2[(t, arm)] = num / den if den else np.nan
    out = []
    for li, t in enumerate(TARGETS):
        for pi, p in enumerate(pseeds):
            x = np.array([r["val"] for r in runs.get((t, p, "xling"), [])])
            q = np.array([r["val"] for r in runs.get((t, p, "pool"), [])])
            if x.size < 2 or q.size < 2:
                continue
            out.append(dict(t=t, li=li, p=p, pi=pi, G=x.mean() - q.mean(),
                            s2=S2[(t, "xling")] / x.size + S2[(t, "pool")] / q.size, nx=x.size, np=q.size))
    return out


# ----------------------------------------------------------------------------- Bayesian grid fit
PRIORS = {
    "primary": dict(sig="hn", scale=5.0, sa=20.0, sb=50.0),
    "S4a_hn10": dict(sig="hn", scale=10.0, sa=20.0, sb=50.0),
    "S4b_hc2.5": dict(sig="hc", scale=2.5, sa=20.0, sb=50.0),
    "S4c_b100": dict(sig="hn", scale=5.0, sa=20.0, sb=100.0),
}


def log_prior_sig(s, kind, scale):
    if kind == "hn":
        return -0.5 * (s / scale) ** 2
    return -np.log1p((s / scale) ** 2)


def fit_bayes(C, prior="primary", include_v=True, ngrid=61, smax=20.0, chunk=4000):
    pr = PRIORS[prior]
    y = np.array([c["G"] for c in C]); s2 = np.array([c["s2"] for c in C]); n = len(C)
    X = np.stack([np.ones(n), DX[[c["li"] for c in C]]], 1)
    lp = sorted({c["pi"] for c in C}); ll = sorted({c["li"] for c in C})
    Zu = np.array([[c["pi"] == k for k in lp] for c in C], float)
    Zv = np.array([[c["li"] == k for k in ll] for c in C], float)
    Ku, Kv = Zu @ Zu.T, Zv @ Zv.T
    Sb = np.diag([pr["sa"] ** 2, pr["sb"] ** 2]); Sbi = np.linalg.inv(Sb); XSX = X @ Sb @ X.T
    g = np.linspace(0, smax, ngrid)
    gv = g if include_v else np.array([0.0])
    SU, SV, SW = [m.ravel() for m in np.meshgrid(g, gv, g, indexing="ij")]
    lw = np.empty(SU.size); mb = np.empty((SU.size, 2)); Vb = np.empty((SU.size, 2, 2))
    I = np.eye(n)
    for s0 in range(0, SU.size, chunk):
        su, sv, sw = SU[s0:s0 + chunk], SV[s0:s0 + chunk], SW[s0:s0 + chunk]
        Sig = (su[:, None, None] ** 2 * Ku + sv[:, None, None] ** 2 * Kv + sw[:, None, None] ** 2 * I
               + np.diag(s2)[None])
        M = Sig + XSX[None]
        L = np.linalg.cholesky(M)
        z = np.linalg.solve(L, np.broadcast_to(y, (len(su), n))[..., None])[..., 0]
        ll_ = -0.5 * (z ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
        lpr = (log_prior_sig(su, pr["sig"], pr["scale"]) + log_prior_sig(sw, pr["sig"], pr["scale"]) +
               (log_prior_sig(sv, pr["sig"], pr["scale"]) if include_v else 0.0))
        lw[s0:s0 + chunk] = ll_ + lpr
        Si = np.linalg.inv(Sig)
        P = X.T[None] @ Si @ X[None] + Sbi[None]
        V = np.linalg.inv(P)
        mb[s0:s0 + chunk] = (V @ (X.T[None] @ Si @ y[None, :, None]))[..., 0]
        Vb[s0:s0 + chunk] = V
    w = np.exp(lw - lw.max()); w /= w.sum()
    s2bar = s2.mean(); s2med = np.median(s2)
    return dict(w=w, mb=mb, Vb=Vb, SU=SU, SV=SV, SW=SW, s2bar=s2bar, s2med=s2med, n=n,
                langs=[TARGETS[k] for k in ll], pseed_idx=lp)


def wquant(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); return np.interp(q, cw / cw[-1], x[o])


def summarize(F, rng, ndraw=400000):
    w = F["w"]
    idx = rng.choice(w.size, size=ndraw, p=w)
    L = np.linalg.cholesky(F["Vb"][idx] + 1e-12 * np.eye(2))
    beta = F["mb"][idx] + (L @ rng.standard_normal((ndraw, 2, 1)))[..., 0]
    a, b = beta[:, 0], beta[:, 1]
    pbneg = float(np.sum(w * norm.cdf(0, F["mb"][:, 1], np.sqrt(F["Vb"][:, 1, 1]))))
    tot = F["SU"] ** 2 + F["SV"] ** 2 + F["SW"] ** 2 + F["s2bar"]
    icc_sh, icc_ps = F["SU"] ** 2 / tot, (F["SU"] ** 2 + F["SW"] ** 2) / tot
    q = lambda arr: [float(v) for v in np.percentile(arr, [2.5, 50, 97.5])]
    qw = lambda arr: [float(wquant(arr, w, v)) for v in (0.025, 0.5, 0.975)]
    with np.errstate(divide="ignore", invalid="ignore"):
        dstar = -a / b
    ok = (b < 0)
    res = dict(n_cells=F["n"], P_b_neg=pbneg, b=q(b), a=q(a), b_mean=float(np.sum(w * F["mb"][:, 1])),
               a_mean=float(np.sum(w * F["mb"][:, 0])), b_CI_excludes_0=bool(q(b)[2] < 0 or q(b)[0] > 0),
               sigma_u=qw(F["SU"]), sigma_v=qw(F["SV"]), sigma_w=qw(F["SW"]), s2bar=float(F["s2bar"]),
               ICC_shared=qw(icc_sh), ICC_pseed=qw(icc_ps),
               dstar_given_bneg=(q(dstar[ok]) if ok.sum() > 100 else None), P_b_neg_mc=float(np.mean(b < 0)))
    pp = {}
    for d in D_PRED:
        x = np.array([1.0, d])
        m = F["mb"] @ x; vb = np.einsum("i,kij,j->k", x, F["Vb"], x)
        v_single = vb + F["SU"] ** 2 + F["SV"] ** 2 + F["SW"] ** 2 + F["s2med"]
        v_avg = vb + F["SV"] ** 2
        pp[str(d)] = dict(single_run=float(np.sum(w * norm.cdf(0, m, np.sqrt(v_single)))),
                          pseed_avg=float(np.sum(w * norm.cdf(0, m, np.sqrt(v_avg)))),
                          line=float(np.sum(w * norm.cdf(0, m, np.sqrt(vb)))),
                          mean=float(np.sum(w * m)))
    res["P_G_neg"] = pp
    return res


def lolo(C, prior="primary", include_v=True):
    out = {}
    for li, t in enumerate(TARGETS):
        Ct = [c for c in C if c["li"] == li]
        if not Ct:
            continue
        F = fit_bayes([c for c in C if c["li"] != li], prior, include_v)
        pred = float(np.sum(F["w"] * (F["mb"] @ np.array([1.0, DX[li]]))))
        obs = float(np.mean([c["G"] for c in Ct]))
        out[t] = dict(pred=pred, obs=obs, correct=bool(np.sign(pred) == np.sign(obs)))
    return out


def mixedlm(cells_):
    import pandas as pd, statsmodels.formula.api as smf, warnings
    df = pd.DataFrame(dict(G=[c["G"] for c in cells_], d=[DIST[c["li"]] for c in cells_],
                           lang=[c["t"] for c in cells_], ps=[c["p"] for c in cells_], one=1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("G ~ d", df, groups="one", re_formula="0",
                            vc_formula={"ps": "0 + C(ps)", "lang": "0 + C(lang)"}).fit(reml=True)
        ci = m.conf_int().loc["d"].tolist()
        return dict(b=float(m.params["d"]), se=float(m.bse["d"]), ci=ci, p=float(m.pvalues["d"]),
                    vc={k: float(v) for k, v in zip(m.model.exog_vc.names, m.vcomp)}, resid=float(m.scale))
    except Exception as e:  # noqa
        return dict(error=repr(e))


# ----------------------------------------------------------------------------- descriptive
def boot_unpaired(Xs, Ms, n, rng):
    """Two-level hierarchical bootstrap, arms unpaired, each arm with its own number of pseeds."""
    def arm(arrs):
        k = len(arrs); out = np.zeros(n)
        for j in range(k):
            sel = rng.integers(0, k, n)
            acc = np.zeros(n)
            for p in range(k):
                hit = sel == p
                if hit.any():
                    a = arrs[p]; acc[hit] = a[rng.integers(0, a.size, (hit.sum(), a.size))].mean(1)
            out += acc
        return out / k
    return arm(Xs) - arm(Ms)


def bdsl_headline(val, test, pseeds, rng, nb=20000):
    def arm_runs(arm, ps, which):
        out = []
        for p in ps:
            b = A.base("BdSL", arm, p); vals = []
            for f in [0, 1, 2]:
                if (b, f) in val and complete(exp_name(b, f)):
                    v = (val if which == "val" else test).get((b, f))
                    if v is not None:
                        vals.append(v)
            if len(vals) >= 2:
                out.append(np.array(vals))
        return out
    mono_ps = [p for p in [0, 1, 2]]
    res = {}
    for which in ("val", "test"):
        Xa, Ma = arm_runs("xling", pseeds, which), arm_runs("mono", mono_ps, which)
        Pa = arm_runs("pool", pseeds, which)
        Xm = arm_runs("xling", [p for p in pseeds if p in mono_ps], which)
        Sa = arm_runs("scratch", [0], which)
        r = {}
        for name, X_, M_ in [("xling-mono (xling all pseeds, mono ps0-2)", Xa, Ma),
                             ("xling-mono matched ps0-2", Xm, Ma),
                             ("pool-mono (G_data; pool all pseeds, mono ps0-2)", Pa, Ma),
                             ("mono-scratch (mono ps0-2; scratch has no pseed)", Ma, Sa)]:
            if not X_ or not M_:
                r[name] = None; continue
            d = boot_unpaired(X_, M_, nb, rng)
            est = np.mean([x.mean() for x in X_]) - np.mean([m.mean() for m in M_])
            r[name] = dict(est=float(est), ci=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                           P_le0=float(np.mean(d <= 0)), n_pseeds=[len(X_), len(M_)],
                           arm_means=[float(np.mean([x.mean() for x in X_])), float(np.mean([m.mean() for m in M_]))])
        res[which] = r
    return res


def descriptive(runs, pseeds, rng, nb=20000):
    X = {t: [np.array([r["val"] for r in runs.get((t, p, "xling"), [])]) for p in pseeds] for t in TARGETS}
    P = {t: [np.array([r["val"] for r in runs.get((t, p, "pool"), [])]) for p in pseeds] for t in TARGETS}
    G = np.array([[X[t][i].mean() - P[t][i].mean() if X[t][i].size and P[t][i].size else np.nan
                   for t in TARGETS] for i in range(len(pseeds))])
    out = dict(G=G.tolist(), per_lang={}, per_pseed={}, test={})
    for j, t in enumerate(TARGETS):
        ok = [i for i in range(len(pseeds)) if X[t][i].size and P[t][i].size]
        d = boot_unpaired([X[t][i] for i in ok], [P[t][i] for i in ok], nb, rng)
        out["per_lang"][t] = dict(d=float(DIST[j]), pooled=float(np.nanmean(G[:, j])),
                                  ci=[float(v) for v in np.percentile(d, [2.5, 97.5])], P_pos=float(np.mean(d > 0)),
                                  per_pseed=[None if np.isnan(g) else float(g) for g in G[:, j]],
                                  sd_over_pseeds=float(np.nanstd(G[:, j], ddof=1)),
                                  pool_mean=float(np.mean([P[t][i].mean() for i in ok])),
                                  xling_mean=float(np.mean([X[t][i].mean() for i in ok])))
    for i, p in enumerate(pseeds):
        if not np.isnan(G[i]).any():
            s = A.law_stats(G[i]); out["per_pseed"][p] = dict(b7=s["b7"], r2_7=s["r2_7"], sign_paper=s["sign_paper"])
    if not np.isnan(G).any():
        s = A.law_stats(G.mean(0))
        out["pooled_ols"] = {k: (v if not isinstance(v, dict) else v) for k, v in s.items()}
        from scipy.stats import spearmanr
        out["pooled_ols"]["rho7"] = float(spearmanr(DIST, G.mean(0)).correlation)
        out["frozen_gate_d0.193"] = int(np.sum((DIST < 0.193) == (G.mean(0) > 0)))
    for t in A.TEST_LANGS:
        Xt, Pt = [], []
        for p in pseeds:
            xs = np.array([r["test"] for r in runs.get((t, p, "xling"), []) if r["test"] is not None and r["f"] <= 2])
            ps_ = np.array([r["test"] for r in runs.get((t, p, "pool"), []) if r["test"] is not None and r["f"] <= 2])
            if xs.size and ps_.size:
                Xt.append(xs); Pt.append(ps_)
        if Xt:
            d = boot_unpaired(Xt, Pt, nb, rng)
            out["test"][t] = dict(pooled=float(np.mean([x.mean() - q.mean() for x, q in zip(Xt, Pt)])),
                                  ci=[float(v) for v in np.percentile(d, [2.5, 97.5])], P_pos=float(np.mean(d > 0)),
                                  n_pseeds=len(Xt), pool=float(np.mean([q.mean() for q in Pt])),
                                  xling=float(np.mean([x.mean() for x in Xt])))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "results_final.csv"))
    ap.add_argument("--pseeds", default="0-6")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "p1_refit_2026-10"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="skip sensitivity analyses (debug)")
    ap.add_argument("--alt-dist", help="json {variants: {name: {lang: d}}}: non-pre-registered sensitivity x-axes")
    a = ap.parse_args()
    pseeds = A.parse_pseeds(a.pseeds)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261022)
    val, test = A.load(a.csv)
    R = dict(prereg_sha256=open(os.path.join(ROOT, "docs/p1_refit_2026-10/PREREG.sha256")).read().split()[0],
             pseeds=pseeds, targets=TARGETS, dist=DIST.tolist())

    runs, log = gather(val, test, pseeds)
    C = cells(runs, pseeds)
    R["exclusions"] = [dict(lang=t, pseed=p, arm=arm, ft_seed=f, reason=why, val=v) for t, p, arm, f, why, v in log]
    R["cells"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()} for c in C]
    full = [p for p in pseeds if sum(c["p"] == p for c in C) == len(TARGETS)]
    R["pseeds_complete_all7"] = full
    R["data_sufficient"] = len(full) >= 5

    print(f"cells: {len(C)}; pseeds with all 7 languages: {full}; exclusions: {len(log)}", flush=True)
    F = fit_bayes(C)
    R["primary"] = summarize(F, rng)
    R["lolo"] = lolo(C)
    nlolo = sum(v["correct"] for v in R["lolo"].values())
    D1 = R["primary"]["P_b_neg"] > 0.95
    D2 = nlolo >= 6
    verdict = ("KEEP-LAW" if (D1 and D2) else "REFRAME") if R["data_sufficient"] else "INCONCLUSIVE (insufficient data)"
    R["decision"] = dict(D1_P_b_neg=R["primary"]["P_b_neg"], D1_pass=bool(D1), D2_lolo_correct=nlolo,
                         D2_pass=bool(D2), verdict=verdict)
    print(json.dumps(R["decision"], indent=1), flush=True)

    if not a.quick:
        R["mixedlm"] = mixedlm(C)
        R["S3_no_v"] = summarize(fit_bayes(C, include_v=False), rng)
        R["S3_no_v"]["lolo_correct"] = sum(v["correct"] for v in lolo(C, include_v=False).values())
        for pk in ("S4a_hn10", "S4b_hc2.5", "S4c_b100"):
            R[pk] = summarize(fit_bayes(C, prior=pk), rng)
        rep = [p for p in pseeds if p != 0]
        Crep = [c for c in C if c["p"] != 0]
        if len({c["p"] for c in Crep}) >= 2:
            R["replication_only_excl_ps0"] = summarize(fit_bayes(Crep), rng)
            R["replication_only_excl_ps0"]["lolo"] = lolo(Crep)
            R["replication_only_excl_ps0"]["pseeds"] = rep
        for nm, kw in (("S1_no_collapse_exclusion", dict(rule="S1")), ("S2_ftseeds_012", dict(seed_mode="012"))):
            r_, _ = gather(val, test, pseeds, **kw); C_ = cells(r_, pseeds)
            R[nm] = summarize(fit_bayes(C_), rng)
            R[nm]["lolo_correct"] = sum(v["correct"] for v in lolo(C_).values())
        if a.alt_dist:
            global DX
            alt = json.load(open(a.alt_dist))
            for key, dd in alt["variants"].items():
                DX = np.array([dd[t] for t in TARGETS])
                r_ = summarize(fit_bayes(C), rng); lo_ = lolo(C)
                r_["lolo"] = lo_; r_["lolo_correct"] = sum(v["correct"] for v in lo_.values()); r_["dist"] = dd
                R.setdefault("alt_dist", {})[key] = r_
            DX = DIST.copy()
        R["descriptive"] = descriptive(runs, pseeds, rng)
        R["bdsl_headline"] = bdsl_headline(val, test, pseeds, rng)
        # tab:transfer / tab:robust refit via the existing tool, against pooled G
        gj = os.path.join(a.out, f"pooled_gains{a.tag}.json")
        json.dump(dict(pseeds=pseeds, targets=TARGETS, G=R["descriptive"]["G"]), open(gj, "w"), indent=1)
        try:
            R["baseline_refit_stdout"] = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tools/p1_law_baseline_refit.py"), "--gains-json", gj],
                capture_output=True, text=True, cwd=ROOT, timeout=3600).stdout
        except Exception as e:  # noqa
            R["baseline_refit_stdout"] = f"ERROR {e!r}"

    def conv(o):
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.bool_): return bool(o)
        return o
    os.makedirs(a.out, exist_ok=True)
    jp = os.path.join(a.out, f"refit_results{a.tag}.json")
    json.dump(conv(R), open(jp, "w"), indent=1)
    print(f"[json] {jp}")


if __name__ == "__main__":
    main()
