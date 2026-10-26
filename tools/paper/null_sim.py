#!/usr/bin/env python
"""Exploratory (NOT pre-registered): how surprising is the published single-run law under the
variance structure estimated by the pre-registered refit, if the true slope were zero?

Model (posterior medians from docs/p1_refit_2026-10/refit_results.json, primary model):
    G_l = v_l + w_l + e_l   (one pretraining run; the shared u_p and intercept a do not affect R^2)
    v_l ~ N(0, sigma_v^2), w_l ~ N(0, sigma_w^2), e_l ~ N(0, s2_l)
with s2_l = mean over pretraining seeds of the cell fine-tune variance s2_{l,p}.
x = the v1 paper distances. Reports P(R^2_7 >= 0.887 and slope < 0) and P(R^2_5 >= 0.997, slope < 0)
for the 5 calibration languages, plus the same under pure iid noise (Beta reference).

  python tools/paper/null_sim.py  > results/provenance/null_sim.txt
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(ROOT, "docs/p1_refit_2026-10/refit_results.json")))
T, d = R["targets"], np.array(R["dist"])
pr = R["primary"]
sv, sw = pr["sigma_v"][1], pr["sigma_w"][1]
s2 = np.array([np.mean([c["s2"] for c in R["cells"] if c["t"] == t]) for t in T])
calib = [T.index(t) for t in ["GSL", "BdSL", "LSA64", "KArSL", "AUTSL"]]

rng = np.random.default_rng(20261022)
N = 2_000_000


def r2_slope(x, Y):
    xc = x - x.mean()
    Yc = Y - Y.mean(1, keepdims=True)
    b = Yc @ xc / (xc @ xc)
    ss_res = ((Yc - b[:, None] * xc[None, :]) ** 2).sum(1)
    ss_tot = (Yc ** 2).sum(1)
    return 1 - ss_res / ss_tot, b


sd = np.sqrt(sv ** 2 + sw ** 2 + s2)
Y = rng.standard_normal((N, len(T))) * sd[None, :]
r7, b7 = r2_slope(d, Y)
r5, b5 = r2_slope(d[calib], Y[:, calib])
print(f"sigma_v={sv:.3f} sigma_w={sw:.3f} s2_l={np.round(s2, 2).tolist()} N={N}")
print(f"P(R2_7 >= 0.887 & b<0) = {np.mean((r7 >= 0.887) & (b7 < 0)):.5f}")
print(f"P(R2_5 >= 0.997 & b<0) = {np.mean((r5 >= 0.997) & (b5 < 0)):.6f}")
print(f"P(R2_5 >= 0.997 & b<0 & R2_7 >= 0.887) = {np.mean((r5 >= 0.997) & (b5 < 0) & (r7 >= 0.887)):.6f}")
# observed replication seeds, same statistic
G = np.array(R["descriptive"]["G"])
rr, bb = r2_slope(d, G)
print("observed per-pseed R2_7:", np.round(rr, 3).tolist(), "slopes:", np.round(bb, 2).tolist())
print(f"null quantiles of R2_7: 50% {np.quantile(r7, .5):.3f}, 90% {np.quantile(r7, .9):.3f}, 99% {np.quantile(r7, .99):.3f}")
