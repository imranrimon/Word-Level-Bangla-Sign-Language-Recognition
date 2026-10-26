#!/usr/bin/env python
"""P1 distance-gating law v2 — analysis exactly as pre-registered in docs/p1_fix_2026-10/PREREG.md.

Primary model (cell level, fine-tune noise as known variance):
    G_{l,p} ~ N(a + b d_l + u_p + w_{l,p}, s2_{l,p});  u_p ~ N(0, su^2), w_{l,p} ~ N(0, sw^2)
    a ~ N(0, 20^2), b ~ N(0, (50/range(d))^2), su, sw ~ HalfNormal(5); grid 61^2 on [0, 20] pp.
SUPPORTED iff 95% CrI of b < 0 AND leave-one-language-out sign accuracy >= 6/8.

  python tools/p1_fix/refit_v2.py --out docs/p1_fix_2026-10/results        # the real analysis
  python tools/p1_fix/refit_v2.py --debug-v1 --out <scratch>                 # code check on seen v1 data
"""
import argparse, csv, json, math, os, re
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from collections import defaultdict
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANGS = ["BdSL", "LSA64", "AUTSL", "KArSL", "GSL", "LSFB", "LIBRAS", "BISINDO"]
STEM = {"BdSL": "bdsl_block_gcn_si", "LSA64": "lsa64_si", "AUTSL": "autsl_si", "KArSL": "karsl_si",
        "GSL": "gsl_si", "LSFB": "lsfb_si", "LIBRAS": "libras_si", "BISINDO": "bisindo_si"}
NCLASS = {"BdSL": 60, "LSA64": 64, "AUTSL": 226, "KArSL": 70, "GSL": 310, "LSFB": 60, "LIBRAS": 20, "BISINDO": 32}
V1_DIST = {"GSL": 0.1206, "BdSL": 0.1429, "LSA64": 0.1763, "LIBRAS": 0.3151, "KArSL": 0.3445,
           "AUTSL": 0.4115, "LSFB": 0.4798, "BISINDO": 0.4495}
_SEED = re.compile(r"_seed(\d+)(_test)?$")


# ------------------------------------------------------------------------------------------ data
def load_csv(path):
    val, test = defaultdict(lambda: -1.0), {}
    if not os.path.exists(path):
        return {}, {}
    for r in csv.reader(open(path)):
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


def base_v2(t, arm, p):
    return f"{STEM[t]}_f10_v2_{arm}_ps{p}"


def base_v1(t, arm, p):
    return f"{STEM[t]}_lr01_f10_{arm}" + ("" if p == 0 else f"_ps{p}")


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


def gather(val, test, langs, pseeds, seeds, basefn, rule="primary", check_files=True):
    runs, log = defaultdict(list), []
    for t in langs:
        for p in pseeds:
            for arm in ("pool", "xling"):
                b = basefn(t, arm, p)
                for f in seeds:
                    v = val.get((b, f))
                    if v is None or (check_files and not complete(f"{b}_seed{f}")):
                        log.append((t, p, arm, f, "incomplete/missing", v)); continue
                    runs[(t, p, arm)].append(dict(f=f, val=v, test=test.get((b, f))))
    if rule == "primary":
        for t in langs:
            allv = [r["val"] for (tt, _, _), rs in runs.items() if tt == t for r in rs]
            if not allv:
                continue
            med, c3 = np.median(allv), 3 * 100.0 / NCLASS[t]
            for key in [k for k in runs if k[0] == t]:
                keep = []
                for r in runs[key]:
                    if r["val"] <= c3:
                        log.append((t, key[1], key[2], r["f"], f"collapsed (<= 3x chance {c3:.2f})", r["val"]))
                    elif r["val"] < 0.5 * med:
                        log.append((t, key[1], key[2], r["f"], f"gross failure (< 0.5 x median {med:.2f})", r["val"]))
                    else:
                        keep.append(r)
                runs[key] = keep
    return runs, log


def cells(runs, langs, pseeds, field="val"):
    S2 = {}
    for t in langs:
        for arm in ("pool", "xling"):
            num = den = 0.0
            for p in pseeds:
                v = np.array([r[field] for r in runs.get((t, p, arm), []) if r[field] is not None])
                if v.size >= 2:
                    num += ((v - v.mean()) ** 2).sum(); den += v.size - 1
            S2[(t, arm)] = num / den if den else np.nan
    out = []
    for t in langs:
        for p in pseeds:
            x = np.array([r[field] for r in runs.get((t, p, "xling"), []) if r[field] is not None])
            q = np.array([r[field] for r in runs.get((t, p, "pool"), []) if r[field] is not None])
            if x.size < 2 or q.size < 2:
                continue
            out.append(dict(t=t, p=p, G=float(x.mean() - q.mean()), xm=float(x.mean()), pm=float(q.mean()),
                            s2=float(S2[(t, "xling")] / x.size + S2[(t, "pool")] / q.size), nx=int(x.size), np=int(q.size),
                            xs=x.tolist(), ps=q.tolist()))
    return out


# ------------------------------------------------------------------------------------------ model
def fit(C, dist, include_v=False, sb=None, sig_scale=5.0, ngrid=61, smax=20.0, chunk=3000):
    y = np.array([c["G"] for c in C]); s2 = np.array([c["s2"] for c in C]); n = len(C)
    d = np.array([dist[c["t"]] for c in C])
    X = np.stack([np.ones(n), d], 1)
    lp = sorted({c["p"] for c in C}); ll = sorted({c["t"] for c in C})
    Ku = np.array([[ci["p"] == cj["p"] for cj in C] for ci in C], float)
    Kv = np.array([[ci["t"] == cj["t"] for cj in C] for ci in C], float)
    if sb is None:
        rng_d = max(dist[t] for t in ll) - min(dist[t] for t in ll)
        sb = 50.0 / max(rng_d, 1e-6)
    Sb = np.diag([20.0 ** 2, sb ** 2]); Sbi = np.linalg.inv(Sb); XSX = X @ Sb @ X.T
    g = np.linspace(0, smax, ngrid); gv = g if include_v else np.array([0.0])
    SU, SV, SW = [m.ravel() for m in np.meshgrid(g, gv, g, indexing="ij")]
    lw = np.empty(SU.size); mb = np.empty((SU.size, 2)); Vb = np.empty((SU.size, 2, 2))
    I = np.eye(n)
    lpri = lambda s: -0.5 * (s / sig_scale) ** 2
    for s0 in range(0, SU.size, chunk):
        su, sv, sw = SU[s0:s0 + chunk], SV[s0:s0 + chunk], SW[s0:s0 + chunk]
        Sig = su[:, None, None] ** 2 * Ku + sv[:, None, None] ** 2 * Kv + sw[:, None, None] ** 2 * I + np.diag(s2)[None]
        L = np.linalg.cholesky(Sig + XSX[None])
        z = np.linalg.solve(L, np.broadcast_to(y, (len(su), n))[..., None])[..., 0]
        lw[s0:s0 + chunk] = (-0.5 * (z ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
                             + lpri(su) + lpri(sw) + (lpri(sv) if include_v else 0.0))
        Si = np.linalg.inv(Sig)
        V = np.linalg.inv(X.T[None] @ Si @ X[None] + Sbi[None])
        mb[s0:s0 + chunk] = (V @ (X.T[None] @ Si @ y[None, :, None]))[..., 0]
        Vb[s0:s0 + chunk] = V
    w = np.exp(lw - lw.max()); w /= w.sum()
    return dict(w=w, mb=mb, Vb=Vb, SU=SU, SV=SV, SW=SW, s2bar=float(s2.mean()), s2med=float(np.median(s2)),
                n=n, langs=ll, pseeds=lp, sb=sb, include_v=include_v)


def wq(x, w, qs):
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return [float(np.interp(q, cw, x[o])) for q in qs]


def summarize(F, dist, rng, ndraw=200000):
    w = F["w"]
    idx = rng.choice(w.size, size=ndraw, p=w)
    Lc = np.linalg.cholesky(F["Vb"][idx] + 1e-12 * np.eye(2))
    beta = F["mb"][idx] + (Lc @ rng.standard_normal((ndraw, 2, 1)))[..., 0]
    a, b = beta[:, 0], beta[:, 1]
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 50, 97.5])]
    tot_ps = F["SU"] ** 2 + F["SW"] ** 2 + F["SV"] ** 2 + F["s2bar"]
    res = dict(n_cells=F["n"], langs=F["langs"], pseeds=F["pseeds"], slope_prior_sd=F["sb"], include_v=F["include_v"],
               b=q(b), a=q(a), b_mean=float(b.mean()), a_mean=float(a.mean()),
               P_b_neg=float(np.sum(w * norm.cdf(0, F["mb"][:, 1], np.sqrt(F["Vb"][:, 1, 1])))),
               C1_CI_below_0=bool(q(b)[2] < 0),
               sigma_u=wq(F["SU"], w, (0.025, 0.5, 0.975)), sigma_w=wq(F["SW"], w, (0.025, 0.5, 0.975)),
               sigma_v=(wq(F["SV"], w, (0.025, 0.5, 0.975)) if F["include_v"] else None),
               ICC_pseed=wq((F["SU"] ** 2 + F["SW"] ** 2) / tot_ps, w, (0.025, 0.5, 0.975)),
               ICC_shared=wq(F["SU"] ** 2 / tot_ps, w, (0.025, 0.5, 0.975)), s2bar=F["s2bar"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = -a / b
    res["dstar_given_bneg"] = q(ds[b < 0]) if (b < 0).sum() > 1000 else None
    dl = sorted(dist[t] for t in F["langs"])
    grid = {f"d_{t}": dist[t] for t in F["langs"]}
    grid.update({"d_min": dl[0], "d_mid": 0.5 * (dl[0] + dl[-1]), "d_max": dl[-1]})
    pp = {}
    for k, dv in grid.items():
        x = np.array([1.0, dv]); m = F["mb"] @ x; vb = np.einsum("i,kij,j->k", x, F["Vb"], x)
        pp[k] = dict(d=dv, mean=float(np.sum(w * m)),
                     single_run=float(np.sum(w * norm.cdf(0, m, np.sqrt(vb + F["SU"] ** 2 + F["SW"] ** 2 + F["SV"] ** 2 + F["s2med"])))),
                     pseed_avg=float(np.sum(w * norm.cdf(0, m, np.sqrt(vb + F["SV"] ** 2)))),
                     line=float(np.sum(w * norm.cdf(0, m, np.sqrt(vb)))))
    res["P_G_neg"] = pp
    return res


def lolo(C, dist, **kw):
    out = {}
    langs = sorted({c["t"] for c in C}, key=LANGS.index)
    for t in langs:
        F = fit([c for c in C if c["t"] != t], dist, **kw)
        pred = float(np.sum(F["w"] * (F["mb"] @ np.array([1.0, dist[t]]))))
        obs = float(np.mean([c["G"] for c in C if c["t"] == t]))
        out[t] = dict(pred=pred, obs=obs, correct=bool(np.sign(pred) == np.sign(obs)))
    n = len(out); k = sum(v["correct"] for v in out.values())
    return dict(per_lang=out, correct=k, n=n, threshold=math.ceil(0.75 * n), C2=bool(k >= math.ceil(0.75 * n)))


def mom_icc(C):
    """Method-of-moments: var of cell G across pseeds within language vs mean s2."""
    by = defaultdict(list)
    for c in C:
        by[c["t"]].append(c)
    within = [np.var([c["G"] for c in v], ddof=1) for v in by.values() if len(v) >= 2]
    s2 = np.mean([c["s2"] for c in C])
    vb = float(np.mean(within)) if within else float("nan")
    return dict(var_G_across_pseeds_within_lang=vb, mean_s2=float(s2), pseed_share=float(max(vb - s2, 0) / vb) if vb > 0 else None)


def boot_lang(runs, langs, pseeds, nb, rng):
    out = {}
    for t in langs:
        cell = {p: (np.array([r["val"] for r in runs.get((t, p, "xling"), [])]),
                    np.array([r["val"] for r in runs.get((t, p, "pool"), [])])) for p in pseeds}
        cell = {p: v for p, v in cell.items() if v[0].size >= 2 and v[1].size >= 2}
        if not cell:
            continue
        P = list(cell)
        Gs = np.empty(nb)
        for i in range(nb):
            ps = rng.choice(P, size=len(P), replace=True)
            Gs[i] = np.mean([rng.choice(cell[p][0], cell[p][0].size).mean() - rng.choice(cell[p][1], cell[p][1].size).mean() for p in ps])
        obs = float(np.mean([cell[p][0].mean() - cell[p][1].mean() for p in P]))
        out[t] = dict(G=obs, CI=[float(x) for x in np.percentile(Gs, [2.5, 97.5])], P_G_neg=float(np.mean(Gs < 0)), n_pseeds=len(P))
    return out


def mixedlm(cl, dist):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as e:
        return dict(error=str(e))
    df = pd.DataFrame([dict(G=c["G"], d=dist[c["t"]], p=str(c["p"]), lp=f"{c['t']}_{c['p']}", g=1) for c in cl])
    try:
        m = smf.mixedlm("G ~ d", df, groups="g", vc_formula={"p": "0 + C(p)"}, re_formula="0").fit(reml=True)
        ci = m.conf_int().loc["d"].tolist()
        return dict(b=float(m.params["d"]), b_CI=[float(x) for x in ci], b_p=float(m.pvalues["d"]), a=float(m.params["Intercept"]),
                    C1=bool(ci[1] < 0), vc=[float(x) for x in m.vcomp], scale=float(m.scale))
    except Exception as e:
        return dict(error=str(e))


def analyse(C, dist, rng, include_v=False):
    F = fit(C, dist, include_v=include_v)
    s = summarize(F, dist, rng)
    lo = lolo(C, dist, include_v=include_v)
    s["lolo"] = lo
    s["verdict"] = "SUPPORTED" if (s["C1_CI_below_0"] and lo["C2"]) else "NOT SUPPORTED"
    return s


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "results_p1fix_v2.csv"))
    ap.add_argument("--dist", default=os.path.join(ROOT, "docs/p1_fix_2026-10/step2/distances_v2.json"))
    ap.add_argument("--pseeds", default="0,1,2")
    ap.add_argument("--debug-v1", action="store_true", help="code check on the (already seen) v1 data")
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261022)
    pseeds = [int(x) for x in a.pseeds.split(",")]
    DJ = json.load(open(a.dist)) if os.path.exists(a.dist) else None

    def dmap(desc="v2", src="pool_v2", bw="pooled"):
        return {t: DJ["results"][desc][src][t][bw]["mean"] for t in LANGS}

    R = {"prereg_sha256": open(os.path.join(ROOT, "docs/p1_fix_2026-10/PREREG.sha256")).read().split()[0]}
    if a.debug_v1:
        val, test = load_csv(os.path.join(ROOT, "results", "results_final.csv"))
        langs = [t for t in LANGS if t != "BISINDO"]
        runs, log = gather(val, test, langs, pseeds, [0, 1, 2], base_v1, check_files=False)
        dist = V1_DIST
    else:
        val, test = load_csv(a.csv)
        langs = LANGS
        runs, log = gather(val, test, langs, pseeds, [0, 1, 2, 3, 4], base_v2)
        dist = dmap()
    C = cells(runs, langs, pseeds)
    complete_ps = [p for p in pseeds if all(any(c["t"] == t and c["p"] == p for c in C) for t in langs)]
    R["exclusions"] = log
    R["cells"] = C
    R["distances_primary"] = dist
    R["pseeds_complete_all_langs"] = complete_ps
    R["primary"] = analyse(C, dist, rng)
    if len(complete_ps) < 2:
        R["primary"]["verdict"] = "INCONCLUSIVE (insufficient data)"
    R["mom_icc"] = mom_icc(C)
    R["boot_lang"] = boot_lang(runs, langs, pseeds, a.boot, rng)
    R["pooled_G"] = {t: float(np.mean([c["G"] for c in C if c["t"] == t])) for t in langs if any(c["t"] == t for c in C)}
    sens = {}
    # S1 no exclusions except incomplete
    if not a.debug_v1:
        r1, _ = gather(val, test, langs, pseeds, [0, 1, 2, 3, 4], base_v2, rule="S1")
        sens["S1_no_exclusions"] = analyse(cells(r1, langs, pseeds), dist, rng)
    sens["S2_language_RE"] = analyse(C, dist, rng, include_v=True)
    if DJ is not None and not a.debug_v1:
        for tag, (desc, src) in {"S3_old_descriptor_v2pool": ("old", "pool_v2"), "S4_v2_interp": ("v2_interp", "pool_v2"),
                                 "S5_v2_desc_oldsrc": ("v2", "oldsrc_wlasl_v2")}.items():
            sens[tag] = analyse(C, dmap(desc, src), rng)
    sens["S6_drop_near_floor"] = analyse([c for c in C if c["t"] not in ("KArSL", "BISINDO")], dist, rng)
    sens["S7_mixedlm"] = mixedlm(C, dist)
    Ct = cells(runs, langs, pseeds, field="test")
    sens["S8_test_split"] = analyse(Ct, dist, rng) if len({c["t"] for c in Ct}) >= 3 else None
    if not a.debug_v1:
        r9, _ = gather(val, test, langs, pseeds, [0, 1, 2], base_v2)
        sens["S9_ftseeds_012"] = analyse(cells(r9, langs, pseeds), dist, rng)
    R["sensitivity"] = sens
    json.dump(R, open(os.path.join(a.out, "refit_v2.json"), "w"), indent=1, default=str)
    P = R["primary"]
    print(f"VERDICT: {P['verdict']}")
    print(f"b = {P['b'][1]:+.2f} [{P['b'][0]:+.2f}, {P['b'][2]:+.2f}]  P(b<0)={P['P_b_neg']:.3f}  "
          f"LOLO {P['lolo']['correct']}/{P['lolo']['n']} (need {P['lolo']['threshold']})")
    print(f"ICC_pseed {P['ICC_pseed']}  sigma_u {P['sigma_u']} sigma_w {P['sigma_w']}  s2bar {P['s2bar']:.2f}")
    for t in langs:
        if t in R["pooled_G"]:
            lo = P["lolo"]["per_lang"].get(t, {})
            print(f"  {t:8s} d={dist[t]:.4f}  G_pooled={R['pooled_G'][t]:+6.2f}  LOLO pred={lo.get('pred', float('nan')):+6.2f} "
                  f"{'ok' if lo.get('correct') else 'WRONG'}  boot CI {R['boot_lang'].get(t, {}).get('CI')}")
    for k, v in sens.items():
        if v and "verdict" in v:
            print(f"  {k:28s} b={v['b'][1]:+.2f} [{v['b'][0]:+.2f},{v['b'][2]:+.2f}] LOLO {v['lolo']['correct']}/{v['lolo']['n']} -> {v['verdict']}")
        elif v:
            print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
