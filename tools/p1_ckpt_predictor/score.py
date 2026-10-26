#!/usr/bin/env python
"""P1 checkpoint-aware predictor of G = acc(xling) - acc(pool): scores + evaluation.

  score.py compute   per-(language, backbone) scores from the features written by extract.py
                     -> work_dir/p1_ckpt_predictor/scores_backbone.csv, scores_pair.csv (cached)
  score.py analyze   join with G per (language, pretraining seed) from results_final.csv and evaluate
                     every predictor against the data-only MMD and the fine-tune-noise ceiling
                     -> markdown on stdout (+ --json)

Per-backbone scores (target TRAIN clips only):
  unlabelled  ssl_loss / ssl_acc  masked code-prediction CE / acc of the backbone's own SSL head on target
                                  clips (pseeds>=1 only: the original seed-0 runs kept no head)
              ssl_gap             ssl_loss(target) - ssl_loss(source pool)
              mmd_ftj / mmd_ssl   RBF-MMD (median heuristic) source-pool vs target features
              md_ftj / md_ssl     normalised mean discrepancy ||mu_s-mu_t||^2 / mean(tr S_s, tr S_t)
              erank_ft/erank_ssl  effective rank (Roy & Vetterli 2007) of target features
              unif_ft             uniformity loss (Wang & Isola 2020) of L2-normalised target features
  10% labels  logme               LogME (You et al., ICML 2021; tools/logme_transfer.logme)
  (exact ft   ncti_*              NCTI (Wang et al., ICCV 2023), official recipe (github BUserName/NCTI):
   subset)                        PCA-64 -> shrinkage LDA; seli = nuclear norm of LDA features, ncc = mean
                                  LDA prob of true class, vc = log sum_c nuclear norm of class prob
                                  matrix; NCTI = mm(seli)+mm(ncc)-mm(vc), min-max over the model hub
                                  (= all pool/xling backbones scored on that language)
              probe_acc           3-fold CV logistic-regression accuracy
              ncc_acc / knn_acc   leave-one-out nearest-class-mean / 1-NN accuracy (cosine)
Pair scores (pool_p vs xling_p): linear CKA (Kornblith et al. 2019) of target features, ft & ssl views.
Predictor of G = dir * (score(xling) - score(pool)), dir = +1 if higher score means better transfer.
"""
from __future__ import annotations

import argparse, csv, glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "work_dir", "p1_ckpt_predictor")
TARGETS = ["BdSL", "LSA64", "KArSL", "AUTSL", "GSL", "LSFB", "LIBRAS"]
DIST_PAPER = dict(zip(TARGETS, [0.1429, 0.1763, 0.3445, 0.4115, 0.1206, 0.4798, 0.3151]))  # gating_law_fit.py
DSTAR = 0.193


# =============================================================== score primitives
def rbf_mmd(X, Y, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)].astype(np.float64)
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)].astype(np.float64)
    Z = np.vstack([X, Y]); sq = (Z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * Z @ Z.T, 0)
    s2 = np.median(d2[d2 > 0]) + 1e-12
    K = np.exp(-d2 / s2); a, b = len(X), len(Y)
    return float((K[:a, :a].sum() - np.trace(K[:a, :a])) / (a * (a - 1))
                 + (K[a:, a:].sum() - np.trace(K[a:, a:])) / (b * (b - 1)) - 2 * K[:a, a:].mean())


def mean_disc(X, Y):
    return float(((X.mean(0) - Y.mean(0)) ** 2).sum() / (0.5 * (X.var(0).sum() + Y.var(0).sum()) + 1e-12))


def erank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def uniformity(X, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    Z = X[rng.choice(len(X), min(n, len(X)), replace=False)].astype(np.float64)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    d2 = np.maximum(2 - 2 * Z @ Z.T, 0)
    iu = np.triu_indices(len(Z), 1)
    v = -2 * d2[iu]; m = v.max()
    return float(m + np.log(np.mean(np.exp(v - m))))


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def contiguous(y):
    return np.unique(y, return_inverse=True)[1]


def ncti_components(X, y):
    """Official NCTI_Score (BUserName/NCTI metrics.py) -> (seli, ncc, vc)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    y = contiguous(y); C = y.max() + 1
    pca = PCA(n_components=min(64, X.shape[0], X.shape[1]))
    Xp = pca.fit_transform(X.astype(np.float64))
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)
    clf = (LinearDiscriminantAnalysis(solver="svd") if temp == 1e-10
           else LinearDiscriminantAnalysis(solver="eigen", shrinkage=float(temp)))
    low = clf.fit_transform(Xp, y)
    low = low - low.mean(0, keepdims=True)
    seli = float(np.linalg.norm(low, ord="nuc"))
    pr = clf.predict_proba(Xp)
    ncc = float(pr[np.arange(len(y)), y].mean())
    vc = float(np.log(sum(np.linalg.norm(pr[y == c], ord="nuc") for c in range(C))))
    return seli, ncc, vc


def probe_acc(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    y = contiguous(y); acc = []
    for tr, te in KFold(3, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(X[tr]), y[tr])
        acc.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    return float(np.mean(acc))


def loo_ncc_knn(X, y):
    """Leave-one-out nearest-class-mean and 1-NN accuracy (cosine, globally centred features);
    queries from singleton classes are skipped (their class has no other member)."""
    y = contiguous(y)
    Z = X - X.mean(0); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    C = y.max() + 1
    S = np.zeros((C, Z.shape[1])); n = np.bincount(y, minlength=C).astype(float)
    np.add.at(S, y, Z)
    ok = n[y] > 1
    M = S / n[:, None]; M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Z @ M.T                                        # (N,C)
    own = (S[y] - Z) / np.maximum(n[y] - 1, 1)[:, None]   # LOO mean of own class
    sims[np.arange(len(y)), y] = (Z * own).sum(1) / (np.linalg.norm(own, axis=1) + 1e-12)
    ncc = float((sims.argmax(1) == y)[ok].mean())
    Gm = Z @ Z.T; np.fill_diagonal(Gm, -np.inf)
    knn = float((y[Gm.argmax(1)] == y)[ok].mean())
    return ncc, knn


def from_logme(X, y):
    from tools.logme_transfer import logme
    return logme(X, contiguous(y))


# =============================================================== compute
def compute(args):
    fb = os.path.join(OUT, "scores_backbone.csv")
    rows = {}
    if os.path.exists(fb):
        for r in csv.DictReader(open(fb)):
            rows[(r["lang"], r["arm"], int(r["pseed"]))] = r
    for lang in TARGETS:
        for f in sorted(glob.glob(os.path.join(OUT, lang, "*_ps*.npz"))):
            arm, p = os.path.basename(f)[:-4].split("_ps"); p = int(p)
            if (lang, arm, p) in rows:
                continue
            z = np.load(f)
            unl, lab = z["is_unl"], z["is_lab"]
            Ft, Fs = z["ft_tgt"][unl], None
            r = dict(lang=lang, arm=arm, pseed=p, n_unl=int(unl.sum()), n_lab=int(lab.sum()))
            if "ssl_loss_tgt" in z.files:
                r.update(ssl_loss=float(z["ssl_loss_tgt"][unl].mean()), ssl_acc=float(z["ssl_acc_tgt"][unl].mean()),
                         ssl_loss_src=float(z["ssl_loss_src"].mean()),
                         ssl_gap=float(z["ssl_loss_tgt"][unl].mean() - z["ssl_loss_src"].mean()))
            r.update(mmd_ftj=rbf_mmd(z["ftj_src"], z["ftj_tgt"][unl]), mmd_ssl=rbf_mmd(z["ssl_src"], z["ssl_tgt"][unl]),
                     md_ftj=mean_disc(z["ftj_src"], z["ftj_tgt"][unl]), md_ssl=mean_disc(z["ssl_src"], z["ssl_tgt"][unl]),
                     erank_ft=erank(Ft), erank_ssl=erank(z["ssl_tgt"][unl]), unif_ft=uniformity(Ft))
            X, y = z["ft_tgt"][lab].astype(np.float64), z["labels"][lab]
            seli, ncc, vc = ncti_components(X, y)
            nccl, knn = loo_ncc_knn(X, y)
            r.update(logme=from_logme(X, y), ncti_seli=seli, ncti_ncc=ncc, ncti_vc=vc,
                     probe_acc=probe_acc(X, y), ncc_acc=nccl, knn_acc=knn)
            rows[(lang, arm, p)] = r
            print(f"[score] {lang} {arm} ps{p}: " + " ".join(f"{k}={v:.4g}" for k, v in r.items()
                                                           if isinstance(v, float)), flush=True)
    keys = sorted({k for r in rows.values() for k in r}, key=lambda k: (k not in ("lang", "arm", "pseed"), k))
    with open(fb + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(fb + ".tmp", fb)
    # pair scores
    prs = []
    for lang in TARGETS:
        for f in sorted(glob.glob(os.path.join(OUT, lang, "pool_ps*.npz"))):
            p = int(os.path.basename(f)[:-4].split("_ps")[1])
            fx = os.path.join(OUT, lang, f"xling_ps{p}.npz")
            if not os.path.exists(fx):
                continue
            a, b = np.load(f), np.load(fx)
            u = a["is_unl"]
            prs.append(dict(lang=lang, pseed=p, cka_ft=linear_cka(a["ft_tgt"][u], b["ft_tgt"][u]),
                            cka_ssl=linear_cka(a["ssl_tgt"][u], b["ssl_tgt"][u])))
    with open(os.path.join(OUT, "scores_pair.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["lang", "pseed", "cka_ft", "cka_ssl"]); w.writeheader()
        w.writerows(prs)
    print(f"[score] {len(rows)} backbone rows, {len(prs)} pair rows")


# =============================================================== analyze
DIRS = {  # +1: higher = better transfer for that backbone
    "ssl_loss": -1, "ssl_acc": +1, "ssl_gap": -1, "mmd_ftj": -1, "mmd_ssl": -1, "md_ftj": -1, "md_ssl": -1,
    "erank_ft": +1, "erank_ssl": +1, "unif_ft": -1,
    "logme": +1, "ncti": +1, "ncti_seli": +1, "ncti_ncc": +1, "ncti_vc": -1,
    "probe_acc": +1, "ncc_acc": +1, "knn_acc": +1}
LABEL_USING = {"logme", "ncti", "ncti_seli", "ncti_ncc", "ncti_vc", "probe_acc", "ncc_acc", "knn_acc"}


def load_G(pseeds, csv_path):
    from tools.p1_law_pretrain_seed_analysis import load, base
    val, _ = load(csv_path)
    G, SE2, N = {}, {}, {}
    for t in TARGETS:
        seeds = [0, 1, 2, 3, 4] if t in ("GSL", "LSFB", "LIBRAS") else [0, 1, 2]
        for p in pseeds:
            arr = {}
            for arm in ("pool", "xling"):
                v = []
                for s in seeds:
                    exp = f"{base(t, arm, p)}_seed{s}"
                    done = os.path.exists(os.path.join(ROOT, "work_dir", exp, f"{exp}_model-99.pt"))
                    if (base(t, arm, p), s) in val and (done or p <= 2):
                        v.append(val[(base(t, arm, p), s)])
                arr[arm] = np.array(v, float)
            if len(arr["pool"]) == len(seeds) and len(arr["xling"]) == len(seeds):
                G[(t, p)] = arr["xling"].mean() - arr["pool"].mean()
                SE2[(t, p)] = arr["xling"].var(ddof=1) / len(seeds) + arr["pool"].var(ddof=1) / len(seeds)
                N[(t, p)] = arr
    return G, SE2, N


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def kendall(a, b):
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).correlation)


def cv_predict(x, g, groups):
    """Leave-one-group-out OLS g ~ a + b x; returns predictions and constant-mean baseline predictions."""
    pred, base = np.zeros_like(g), np.zeros_like(g)
    for k in np.unique(groups):
        te, tr = groups == k, groups != k
        if np.ptp(x[tr]) > 0:
            b, a = np.polyfit(x[tr], g[tr], 1)
        else:
            b, a = 0.0, g[tr].mean()
        pred[te] = a + b * x[te]; base[te] = g[tr].mean()
    return pred, base


def analyze(args):
    pseeds = list(range(7))
    G, SE2, NR = load_G(pseeds, args.csv)
    units = sorted(G, key=lambda u: (u[1], TARGETS.index(u[0])))
    lang = np.array([u[0] for u in units]); ps = np.array([u[1] for u in units])
    g = np.array([G[u] for u in units]); se2 = np.array([SE2[u] for u in units])
    print(f"# P1 checkpoint-aware G predictor\n\nunits (language x pretraining seed) with complete fine-tunes: "
          f"{len(units)}; pretraining seeds present: {sorted(set(ps.tolist()))}\n")

    # ---------------- scores -> per-unit predictors
    B = {}
    for r in csv.DictReader(open(os.path.join(OUT, "scores_backbone.csv"))):
        B[(r["lang"], r["arm"], int(r["pseed"]))] = {k: (float(v) if v not in ("", None) else np.nan)
                                                     for k, v in r.items() if k not in ("lang", "arm", "pseed")}
    for t in TARGETS:  # NCTI min-max over the model hub of that language
        hub = [k for k in B if k[0] == t]
        for comp in ("ncti_seli", "ncti_ncc", "ncti_vc"):
            v = np.array([B[k][comp] for k in hub]); lo, hi = v.min(), v.max()
            for k in hub:
                B[k]["_mm_" + comp] = (B[k][comp] - lo) / (hi - lo + 1e-12)
        for k in hub:
            B[k]["ncti"] = B[k]["_mm_ncti_seli"] + B[k]["_mm_ncti_ncc"] - B[k]["_mm_ncti_vc"]
    PR = {(r["lang"], int(r["pseed"])): r for r in csv.DictReader(open(os.path.join(OUT, "scores_pair.csv")))}
    dm = json.load(open(os.path.join(OUT, "data_mmd.json")))

    preds = {}
    for s, d in DIRS.items():
        v = np.array([d * (B.get((t, "xling", p), {}).get(s, np.nan) - B.get((t, "pool", p), {}).get(s, np.nan))
                      for t, p in units])
        preds[s] = v
    for s in ("cka_ft", "cka_ssl"):
        preds["1-" + s] = np.array([1 - float(PR[(t, p)][s]) if (t, p) in PR else np.nan for t, p in units])
    data_only = {"mmd_paper": np.array([-DIST_PAPER[t] for t in lang]),
                 "mmd_realpool": np.array([-dm[t]["realpool"] for t in lang]),
                 "mmd_realpool_masked": np.array([-dm[t]["realpool_masked"] for t in lang])}
    preds.update(data_only)

    # ---------------- noise ceiling
    rel = 1 - se2.mean() / g.var(ddof=1)
    rng = np.random.default_rng(0)
    gm = g.mean(); sims = []
    for _ in range(4000):
        gt = gm + np.sqrt(max(rel, 0)) * (g - gm)
        sims.append(spearman(gt, gt + rng.normal(0, np.sqrt(se2))))
    rho_ceiling = float(np.mean(sims))
    mae_floor = float(np.mean(np.sqrt(2 / np.pi) * np.sqrt(se2)))
    # variance components
    lm = {t: g[lang == t].mean() for t in set(lang)}; pm = {p: g[ps == p].mean() for p in set(ps)}
    resid = g - np.array([lm[t] for t in lang]) - np.array([pm[p] for p in ps]) + g.mean()
    var_lang = np.var([lm[t] for t in lm], ddof=1); var_ps = np.var([pm[p] for p in pm], ddof=1)
    print("## Outcome G and noise\n")
    print(f"* G over units: mean {g.mean():+.2f}pp, SD {g.std(ddof=1):.2f}pp; mean fine-tune SE of G "
          f"{np.sqrt(se2.mean()):.2f}pp")
    print(f"* variance of language means {var_lang:.2f}, of pretraining-seed means {var_ps:.2f}, "
          f"interaction+noise residual {resid.var(ddof=1):.2f}, mean SE^2 {se2.mean():.2f} (pp^2)")
    print(f"* reliability of observed G = 1 - mean(SE^2)/Var(G) = {rel:.2f} -> Pearson ceiling "
          f"{np.sqrt(max(rel, 0)):.2f}, simulated Spearman ceiling {rho_ceiling:.2f}; MAE floor of a perfect "
          f"predictor (fine-tune noise only) {mae_floor:.2f}pp")
    # split-half reliability (fine-tune seeds even/odd), Spearman-Brown corrected
    ga, gb = [], []
    for u in units:
        x, p_ = NR[u]["xling"], NR[u]["pool"]
        ga.append(x[0::2].mean() - p_[0::2].mean()); gb.append(x[1::2].mean() - p_[1::2].mean())
    r_sh = float(np.corrcoef(ga, gb)[0, 1]); r_sb = 2 * r_sh / (1 + r_sh)
    print(f"* split-half (fine-tune seeds even vs odd) r = {r_sh:.2f}, Spearman-Brown reliability {r_sb:.2f} "
          f"-> ceiling sqrt = {np.sqrt(max(r_sb, 0)):.2f}\n")

    # language-mean oracle for LOSeedO (best any data-only predictor could do)
    orc = np.array([g[(lang == t) & (ps != p)].mean() for t, p in units])
    _, cst = cv_predict(np.zeros_like(g), g, ps)
    print(f"* LOSeedO reference: constant (training-mean) MAE {np.mean(np.abs(cst - g)):.2f}pp; "
          f"language-mean-from-other-seeds oracle MAE {np.mean(np.abs(orc - g)):.2f}pp, "
          f"sign {np.mean(np.sign(orc) == np.sign(g)):.2f} (a data-only predictor can use only language identity; "
          f"this oracle knows each language's mean G at the other seeds); constant-mean sign "
          f"{np.mean(np.sign(cst) == np.sign(g)):.2f}\n")
    print("* the data-only law within each pretraining seed (Spearman of -MMD_paper vs G over languages): " +
          ", ".join(f"ps{q} {spearman(-np.array([DIST_PAPER[t] for t in lang[ps == q]]), g[ps == q]):+.2f}"
                    for q in sorted(set(ps.tolist())) if (ps == q).sum() >= 4) + "\n")

    # ---------------- evaluation table
    def stats(x, sel):
        xs, gs, ls, pss = x[sel], g[sel], lang[sel], ps[sel]
        out = dict(n=int(sel.sum()), rho=spearman(xs, gs), tau=kendall(xs, gs))
        # seed-clustered bootstrap CI
        up = np.unique(pss); bs = []
        for _ in range(args.n_boot):
            dr = rng.choice(up, len(up))
            idx = np.concatenate([np.where(pss == q)[0] for q in dr])
            if np.ptp(xs[idx]) > 0 and np.ptp(gs[idx]) > 0:
                bs.append(spearman(xs[idx], gs[idx]))
        out["ci"] = tuple(np.percentile(bs, [2.5, 97.5])) if bs else (np.nan, np.nan)
        out["sign0"] = float(np.mean(np.sign(xs) == np.sign(gs)))
        pl, bl = cv_predict(xs, gs, ls); pp, bp = cv_predict(xs, gs, pss)
        out.update(lolo_mae=np.mean(np.abs(pl - gs)), lolo_base=np.mean(np.abs(bl - gs)),
                   lolo_sign=np.mean(np.sign(pl) == np.sign(gs)),
                   loso_mae=np.mean(np.abs(pp - gs)), loso_base=np.mean(np.abs(bp - gs)),
                   loso_sign=np.mean(np.sign(pp) == np.sign(gs)))
        # decomposition (scale-free): mean per-language Spearman across pretraining seeds (can the score
        # track the backbone-pair effect?), mean per-seed Spearman across languages (the paper's law view),
        # and Spearman of the per-seed means (n = #pretraining seeds)
        def mean_rho(key, other_min=4):
            vals = []
            for q in np.unique(key):
                m = key == q
                if m.sum() >= other_min and np.ptp(xs[m]) > 1e-12 and np.ptp(gs[m]) > 0:
                    vals.append(spearman(xs[m], gs[m]))
            return float(np.mean(vals)) if vals else np.nan
        out["rho_within_lang"] = mean_rho(ls)
        out["rho_within_seed"] = mean_rho(pss)
        xp = np.array([xs[pss == q].mean() for q in up]); gp = np.array([gs[pss == q].mean() for q in up])
        out["rho_seedmeans"] = spearman(xp, gp) if np.ptp(xp) > 1e-12 else np.nan
        return out

    res = {}
    print("## Predictor evaluation (units = language x pretraining seed; CI = 95% pretraining-seed-clustered "
          "bootstrap of Spearman)\n")
    print("| predictor | type | n | Spearman ρ [95% CI] | Kendall τ | sign@0 | LOLO MAE (const) | LOLO sign | "
          "LOSeedO MAE (const) | LOSeedO sign | mean ρ within-lang (across pseeds) | mean ρ within-seed (across langs) | ρ seed-means |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = list(data_only) + [k for k in preds if k not in data_only]
    for name in order:
        x = preds[name]; sel = ~np.isnan(x)
        if sel.sum() < 6:
            continue
        s = stats(x, sel); res[name] = s
        typ = ("data-only" if name in data_only else "pair (unsigned)" if name.startswith("1-cka")
               else "ckpt+10% labels" if name in LABEL_USING else "ckpt, label-free")
        sign0 = (f"{s['sign0']:.2f}" if name not in data_only and not name.startswith("1-cka") else
                 (f"{np.mean(np.where(-x[sel] < DSTAR, 1, -1) == np.sign(g[sel])):.2f} (gate d*)" if name == "mmd_paper"
                  else "–"))
        fm = lambda v: "–" if v != v else f"{v:+.2f}"
        print(f"| {name} | {typ} | {s['n']} | {s['rho']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | {s['tau']:+.2f} | "
              f"{sign0} | {s['lolo_mae']:.2f} ({s['lolo_base']:.2f}) | {s['lolo_sign']:.2f} | "
              f"{s['loso_mae']:.2f} ({s['loso_base']:.2f}) | {s['loso_sign']:.2f} | {fm(s['rho_within_lang'])} | "
              f"{fm(s['rho_within_seed'])} | {fm(s['rho_seedmeans'])} |")
    # Westfall-Young family-wise adjusted p for |Spearman| (free permutation of G over units)
    fam = [k for k in res if not k.startswith("1-cka")]
    obs = {k: abs(res[k]["rho"]) for k in fam}
    from scipy.stats import rankdata
    R = {k: rankdata(preds[k][~np.isnan(preds[k])]) for k in fam}
    S = {k: ~np.isnan(preds[k]) for k in fam}
    mx, raw = [], {k: 0 for k in fam}
    for _ in range(args.n_perm):
        gp = g[rng.permutation(len(g))]
        m = 0.0
        for k in fam:
            r = abs(np.corrcoef(R[k], rankdata(gp[S[k]]))[0, 1]); m = max(m, r)
            raw[k] += r >= obs[k] - 1e-12
        mx.append(m)
    mx = np.array(mx)
    print(f"\n* permutation p-values for |Spearman| ({args.n_perm} free permutations of G; raw / Westfall-Young "
          f"family-wise over {len(fam)} predictors): " +
          "; ".join(f"{k} {raw[k] / args.n_perm:.3f}/{np.mean(mx >= obs[k] - 1e-12):.3f}"
                    for k in sorted(fam, key=lambda k: -obs[k])[:6]))
    for k in fam:
        res[k]["p_raw"] = raw[k] / args.n_perm; res[k]["p_fwer"] = float(np.mean(mx >= obs[k] - 1e-12))
    # |G| for the unsigned pair scores
    for name in ("1-cka_ft", "1-cka_ssl"):
        x = preds[name]; sel = ~np.isnan(x)
        if sel.sum() >= 6:
            print(f"\n* {name} vs |G|: Spearman {spearman(x[sel], np.abs(g[sel])):+.2f} (n={sel.sum()})")
    # per-unit dump
    print("\n## Per-unit G and predictors\n")
    cols = ["mmd_paper", "ssl_loss", "mmd_ssl", "logme", "ncti", "probe_acc"]
    print("| lang | pseed | G | SE | " + " | ".join(cols) + " |")
    print("|---|---|---|---|" + "---|" * len(cols))
    for i, (t, p) in enumerate(units):
        print(f"| {t} | {p} | {g[i]:+.2f} | {np.sqrt(se2[i]):.2f} | " +
              " | ".join("–" if np.isnan(preds[c][i]) else f"{preds[c][i]:+.4g}" for c in cols) + " |")
    print("\n## Data-only MMD values (pool -> target)\n")
    print("| target | paper (gating_law_fit) | real pool | real pool, dropout-masked |")
    print("|---|---|---|---|")
    for t in TARGETS:
        print(f"| {t} | {DIST_PAPER[t]:.3f} | {dm[t]['realpool']:.3f} ± {dm[t]['realpool_sd']:.3f} | "
              f"{dm[t]['realpool_masked']:.3f} ± {dm[t]['realpool_masked_sd']:.3f} |")
    if args.json:
        json.dump(dict(units=[list(u) for u in units], G=g.tolist(), SE=np.sqrt(se2).tolist(), rel=rel,
                       rho_ceiling=rho_ceiling, mae_floor=mae_floor, split_half=r_sh, spearman_brown=r_sb,
                       results={k: {kk: (list(vv) if isinstance(vv, tuple) else float(vv)) for kk, vv in v.items()}
                                for k, v in res.items()},
                       preds={k: v.tolist() for k, v in preds.items()}), open(args.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compute", "analyze"])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "results_final.csv"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--json")
    a = ap.parse_args()
    compute(a) if a.cmd == "compute" else analyze(a)


if __name__ == "__main__":
    main()
