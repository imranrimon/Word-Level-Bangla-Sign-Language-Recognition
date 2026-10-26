#!/usr/bin/env python3
"""Cross-domain gate analysis (scale-normalized redo). Reads best-Top1 per hrnet_ft
run from results_final.csv but ONLY rows on/after --since (default 2026-08-21) so the
old, confounded pixel-scale run's rows are excluded (append-only CSV reuses exp names).
Pairs gain = SSL-init - scratch with the fresh scale-normalized MMD and tests the gate.
"""
import csv, collections, statistics as st, re, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-08-21")
ap.add_argument("--csv", default="results/results_final.csv", help="run log (results_final.csv for your own runs)")
ap.add_argument("--mmd-log", default="results/hrnet_domain_mmd_norm.log",
                help="stdout of tools/compute_hrnet_domain_distance.py")
a = ap.parse_args()

best = collections.defaultdict(float)
n_rows = 0
for r in csv.reader(open(a.csv)):
    if len(r) < 4 or not r[1].startswith("hrnet_ft_"):
        continue
    if r[0] < a.since:                       # timestamp string compare (ISO-ish)
        continue
    try:
        best[r[1]] = max(best[r[1]], float(r[3])); n_rows += 1
    except ValueError:
        pass
print(f"[gate] {n_rows} clean rows (since {a.since}), {len(best)} exps")

def m(tgt, tag):
    v = [best.get(f"hrnet_ft_{tgt}_{tag}_seed{s}") for s in (0, 1)]
    v = [x for x in v if x]
    return st.mean(v) * 100 if v else None

mmd = {}
try:
    for line in open(a.mmd_log):
        mo = re.search(r"(\w+)\s*->\s*(\w+)\s+MMD=([\d.]+)", line)
        if mo:
            mmd[(mo.group(1), mo.group(2))] = float(mo.group(3))
except FileNotFoundError:
    print(f"[warn] {a.mmd_log} not found; run tools/compute_hrnet_domain_distance.py first")

pairs = [("ntu_hrnet", "hmdb51", "fromNTU"), ("ntu_hrnet", "ucf101", "fromNTU"),
         ("ntu_hrnet", "gym", "fromNTU"), ("ucf101", "hmdb51", "fromUCF"),
         ("ucf101", "gym", "fromUCF"), ("ucf101", "ntu_hrnet", "fromUCF")]
print(f"\n{'src->tgt':26}{'MMD':>7}{'scratch':>9}{'SSL':>8}{'gain':>8}")
rows = []
for src, tgt, tag in sorted(pairs, key=lambda p: mmd.get((p[0], p[1]), 9)):
    d = mmd.get((src, tgt)); sc = m(tgt, "scratch"); ss = m(tgt, tag)
    if None in (d, sc, ss):
        print(f"  {src}->{tgt:16} incomplete (d={d} scratch={sc} ssl={ss})"); continue
    g = ss - sc; rows.append((d, g))
    print(f"  {src}->{tgt:16}{d:7.3f}{sc:9.2f}{ss:8.2f}{g:+8.2f}")
if len(rows) >= 4:
    import numpy as np
    d = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows])
    b, a0 = np.polyfit(d, g, 1); pr = a0 + b * d
    r2 = 1 - ((g - pr) ** 2).sum() / ((g - g.mean()) ** 2).sum() if g.var() > 0 else 0.0
    print(f"\nGATE fit: gain = {a0:.2f} {b:+.2f}*MMD   R2={r2:.3f}")
    print(f"close (MMD<0.25) mean gain {g[d<0.25].mean():+.2f} | far (MMD>0.25) {g[d>0.25].mean():+.2f}")
    print("VERDICT:", "GATE HOLDS (neg slope, far backfires)" if b < 0 and g[d>0.25].mean() < g[d<0.25].mean()
          else "NULL (distance does not govern cross-domain transfer)")
