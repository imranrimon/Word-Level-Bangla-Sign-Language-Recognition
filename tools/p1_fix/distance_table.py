#!/usr/bin/env python
"""Markdown tables from docs/p1_fix_2026-10/step2/distances_v2.json: step-by-step decomposition of the
distance fix (v1 replica -> WLASL fix -> real pool source -> dropout mask -> per-second+scale -> 25 Hz = v2),
plus rank correlations with the v1 paper distances.
  python tools/p1_fix/distance_table.py [--json ...] > docs/p1_fix_2026-10/step2/distances_v2.md
"""
import argparse, json
from scipy.stats import spearmanr

LANGS = ["GSL", "BdSL", "LSA64", "LIBRAS", "KArSL", "AUTSL", "LSFB", "BISINDO", "ASL-Citizen-60"]
V1 = {"GSL": 0.1206, "BdSL": 0.1429, "LSA64": 0.1763, "LIBRAS": 0.3151, "KArSL": 0.3445, "AUTSL": 0.4115,
      "LSFB": 0.4798, "BISINDO": 0.4495}
STEPS = [("A v1 replica (old desc, 400 WLASL-v1 + 400 BdSLW60)", "old", "oldsrc_wlasl_v1"),
         ("B + WLASL skeleton fix", "old", "oldsrc_wlasl_v2"),
         ("C + real pool source (v1 WLASL)", "old", "pool_v1"),
         ("D + real pool source (v2 WLASL)", "old", "pool_v2"),
         ("E + dropout mask (per frame)", "masked_frame", "pool_v2"),
         ("F + per-second + shoulder scale", "masked_sec", "pool_v2"),
         ("G = v2 (+25 Hz resampling) PRIMARY", "v2", "pool_v2"),
         ("H v2_interp (bridge gaps <=0.2 s)", "v2_interp", "pool_v2"),
         ("I v2 descriptor, v1-style source (WLASL v2)", "v2", "oldsrc_wlasl_v2")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default="docs/p1_fix_2026-10/step2/distances_v2.json")
    ap.add_argument("--bw", default="pooled")
    a = ap.parse_args()
    R = json.load(open(a.json))["results"]
    langs = [t for t in LANGS if t in R["v2"]["pool_v2"]]
    print(f"MMD^2_u (mean ± SD over draws, bandwidth={a.bw}); law languages first; ASL-Citizen-60 = control\n")
    print("| step | " + " | ".join(langs) + " | Spearman vs v1 (8 law langs) |")
    print("|---|" + "---|" * (len(langs) + 1))
    print("| v1 paper (tools/gating_law_fit.py) | " + " | ".join(f"{V1[t]:.4f}" if t in V1 else "–" for t in langs) + " | 1 |")
    for name, d, s in STEPS:
        if d not in R or s not in R[d]:
            continue
        cells = [R[d][s][t][a.bw] for t in langs]
        law = [t for t in langs if t in V1]
        rho = spearmanr([V1[t] for t in law], [R[d][s][t][a.bw]["mean"] for t in law]).correlation
        print(f"| {name} | " + " | ".join(f"{c['mean']:.4f} ± {c['sd']:.4f}" for c in cells) + f" | {rho:+.2f} |")
    print("\nRank order (v2 primary):", " < ".join(sorted([t for t in langs if t in V1], key=lambda t: R['v2']['pool_v2'][t][a.bw]['mean'])))


if __name__ == "__main__":
    main()
