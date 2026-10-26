#!/usr/bin/env python
"""Exact REML for the v1 7-pseed refit table (paper Sec. 5.2 / 6.1).

Model on the 7 x 7 table of cell gains G[p, l] (docs/p1_refit_2026-10/pooled_gains.json):
    G = mu + b * d_l + lang_l + pseed_p + resid,  crossed random effects, REML.
statsmodels' MixedLM (docs/p1_refit_2026-10/RESULTS.md) converged to the sigma_lang = 0 boundary; this
maximises the REML likelihood directly (multi-start Nelder-Mead on log variances) and checks that, on this
balanced design, it equals the method-of-moments (ANOVA, lack-of-fit) estimates.
  python tools/paper/exact_reml_v1.py   ->  results/provenance/v1_exact_reml_2026-10-23.txt
"""
import json, os
import numpy as np
from scipy import optimize, stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = json.load(open(os.path.join(ROOT, "docs/p1_refit_2026-10/pooled_gains.json")))
R = json.load(open(os.path.join(ROOT, "docs/p1_refit_2026-10/refit_results.json")))
T, G = V["targets"], np.array(V["G"])
P, L = G.shape
dist = dict(zip(R["targets"], R["dist"]))
d = np.array([dist[t] for t in T])
y = G.ravel()
n = y.size
X = np.column_stack([np.ones(n), np.tile(d, P)])
Zl = np.kron(np.ones((P, 1)), np.eye(L))
Zp = np.kron(np.eye(P), np.ones((L, 1)))


def neg_reml(th, ret=False, X=X):
    sl, sp, se = np.exp(th)
    Vm = sl * Zl @ Zl.T + sp * Zp @ Zp.T + se * np.eye(n)
    Vi = np.linalg.inv(Vm)
    XtVX = X.T @ Vi @ X
    beta = np.linalg.solve(XtVX, X.T @ Vi @ y)
    r = y - X @ beta
    if ret:
        return beta, np.linalg.inv(XtVX)
    return 0.5 * (np.linalg.slogdet(Vm)[1] + np.linalg.slogdet(XtVX)[1] + r @ Vi @ r)


best = min((optimize.minimize(neg_reml, np.array(s, float), method="Nelder-Mead",
                              options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
            for s in [(0, 0, 2), (1, 1, 2), (-3, 0, 2), (0, -3, 2)]), key=lambda o: o.fun)
vc = np.exp(best.x)
beta, cov = neg_reml(best.x, ret=True)
se = np.sqrt(cov[1, 1])
p = 2 * stats.norm.sf(abs(beta[1] / se))
# method of moments with d as a fixed effect (lack-of-fit MS for language)
gm = G.mean(0)
b_ols, a_ols = np.polyfit(d, gm, 1)
ms_lof = P * np.sum((gm - a_ols - b_ols * d) ** 2) / (L - 2)
res = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
ms_res = np.sum(res ** 2) / ((L - 1) * (P - 1))
ms_ps = L * np.sum((G.mean(1) - G.mean()) ** 2) / (P - 1)
mom = ((ms_lof - ms_res) / P, (ms_ps - ms_res) / L, ms_res)
print(f"exact REML VCs (pp^2): lang {vc[0]:.3f}  pseed {vc[1]:.3f}  resid {vc[2]:.3f}")
print(f"MoM (with d)    (pp^2): lang {mom[0]:.3f}  pseed {mom[1]:.3f}  resid {mom[2]:.3f}")
print(f"b = {beta[1]:.3f}, Wald 95% CI [{beta[1] - 1.96 * se:.2f}, {beta[1] + 1.96 * se:.2f}], p = {p:.3f}; "
      f"OLS slope on pooled G {b_ols:.3f}")
assert np.allclose(vc, mom, atol=2e-3) and abs(beta[1] - b_ols) < 1e-6
# without the distance term (as in Fig. fig:variance): language VC = ANOVA language MS
X0 = X[:, :1]
best0 = min((optimize.minimize(neg_reml, np.array(s_, float), args=(False, X0), method="Nelder-Mead",
                               options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
             for s_ in [(0, 0, 2), (1, 1, 2), (-3, 0, 2), (0, -3, 2)]), key=lambda o: o.fun)
vc0 = np.exp(best0.x)
ms_lang = P * np.sum((gm - gm.mean()) ** 2) / (L - 1)
mom0 = ((ms_lang - ms_res) / P, mom[1], ms_res)
print(f"no distance term: exact REML lang {vc0[0]:.3f} pseed {vc0[1]:.3f} resid {vc0[2]:.3f}; "
      f"MoM lang {mom0[0]:.3f} pseed {mom0[1]:.3f} resid {mom0[2]:.3f}")
assert np.allclose(vc0, mom0, atol=2e-3)
