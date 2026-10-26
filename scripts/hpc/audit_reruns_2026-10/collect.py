#!/usr/bin/env python
"""Collect 2026-10 audit-rerun results (seed-mean of per-seed best-epoch val Top-1,
the P1 convention) from the run log and write r1_autsl_leakfree.md / r2_true_mono.md.

  python scripts/hpc/audit_reruns_2026-10/collect.py [--csv results/results_final.csv] [--out DIR] [--monotr-seeds 0,1,2]
"""
import argparse, csv, collections, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/results_final.csv", help="run log")
ap.add_argument("--out", default="docs/audit_reruns_2026-10", help="output directory")
ap.add_argument("--monotr-seeds", default="", help="restrict *_monotr runs to these fine-tune seeds, e.g. 0,1,2 "
                "(the paper's true-mono table uses seeds 0-2; the log also has GSL seeds 3-4)")
a = ap.parse_args()
CSV, OUT = a.csv, a.out
MONO_SEEDS = {int(x) for x in a.monotr_seeds.split(",") if x != ""}

rows = collections.defaultdict(dict)
for r in csv.reader(open(CSV)):
    if len(r) < 4:
        continue
    m = re.fullmatch(r"(.*)_seed(\d+)", r[1])
    if not m:
        continue
    try:
        acc = float(r[3]) * 100
    except ValueError:
        continue
    if MONO_SEEDS and m[1].endswith("_monotr") and int(m[2]) not in MONO_SEEDS:
        continue
    rows[m[1]][int(m[2])] = max(rows[m[1]].get(int(m[2]), 0.0), acc)


def cell(stem):
    v = rows.get(stem)
    if not v:
        return "pending", None
    xs = [v[s] for s in sorted(v)]
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{st.mean(xs):.2f} ± {sd:.2f} (n={len(xs)}: " + "/".join(f"{x:.2f}" for x in xs) + ")", st.mean(xs)


def table(title, spec):
    out = [f"### {title}", "", "| condition | original (leaky) | leak-free rerun | Δ |", "|---|---|---|---|"]
    for name, old, new in spec:
        o, om = cell(old) if old else ("—", None)
        n, nm = cell(new) if new else ("—", None)
        d = f"{nm - om:+.2f}" if (om is not None and nm is not None) else ""
        out.append(f"| {name} | {o} | {n} | {d} |")
    return "\n".join(out) + "\n"


r1 = ["# R1 — leak-free AUTSL gate / CORAL experiments (2026-10 audit)", "",
      "Original runs pretrained on `data/ssl_pool_manifest_multisrc.json`, which contains ALL 36,289 AUTSL clips "
      "(incl. 3,741 official TEST + 4,418 VAL). Reruns: **noautsl** = same pool with every AUTSL clip removed "
      "(AUTSL genuinely foreign); **autsltrain** = AUTSL restricted to its 31 TRAIN signers (CORAL needs "
      "target-language clips, so it cannot run on noautsl). Fine-tune recipe identical (`config/autsl_si_lr01.yaml`, "
      "base_lr 0.01, 100 ep). Metric: seed-mean of per-seed best-epoch AUTSL val Top-1 (P1 convention).", ""]
for tag, frac in (("f10", "@10%"), ("f25", "@25%")):
    r1.append(table(f"AUTSL {frac}", [
        ("uniform-weight align (pool6_uniform)", f"ft_autsl_pool6_{tag}", f"ft_autsl_pool6_noautsl_{tag}"),
        ("DG-XL gate, BdSL-keyed (dgxl)", f"ft_autsl_dgxl_{tag}", f"ft_autsl_dgxl_noautsl_{tag}"),
        ("AUTSL-anchored gate (dgxl_autsl)", f"ft_autslA_{tag}", f"ft_autslA_noautsl_{tag}"),
        ("corrected continuous gate (lambdad)", f"lambdad_autsl_{tag}", f"lambdad_autsl_noautsl_{tag}"),
        ("no-alignment pool, same data (new reference)", None, f"ft_autsl_poolnoalign_noautsl_{tag}"),
        ("CORAL (autsltrain pool)", f"coral_autsl_{tag}", f"coral_autsl_autsltrain_{tag}"),
        ("no-alignment pool, autsltrain (CORAL reference)", None, f"ft_autsl_poolnoalign_autsltrain_{tag}"),
        ("[ref] tab:law pool, Bangla+ASL (unaffected)", f"autsl_si_lr01_{tag}_pool", None),
        ("[ref] tab:law xling, Bangla+ASL (unaffected)", f"autsl_si_lr01_{tag}_xling", None),
    ]))
open(f"{OUT}/r1_autsl_leakfree.md", "w").write("\n".join(r1))

r2 = ["# R2 — true (train-signers-only) monolingual SSL (2026-10 audit)", "",
      "Original `mono` for LSA64/KArSL/AUTSL loaded `bdsl_shubert_pretrain_bdsl_only` (Bangla SSL, not target-only). "
      "Original GSL mono (`src_gsl`) WAS target-only but its pool contained all 7 GSL signers incl. val (6) and test (7). "
      "Reruns pretrain on each target's TRAIN signers only (same SSL recipe as src_gsl: 30 ep, mask 0.3, K=64 "
      "pose_motion codes) and fine-tune with the original recipe. G_data = pool − mono.", ""]
for t, fr in (("lsa64", ("f10", "f25", "f50")), ("karsl", ("f10", "f25")), ("autsl", ("f10", "f25")), ("gsl", ("f10",))):
    spec = []
    for f in fr:
        spec += [(f"{t} {f} mono (old)→true mono", f"{t}_si_lr01_{f}_mono", f"{t}_si_lr01_{f}_monotr"),
                 (f"{t} {f} pool [ref]", f"{t}_si_lr01_{f}_pool", None),
                 (f"{t} {f} xling [ref]", f"{t}_si_lr01_{f}_xling", None)]
    r2.append(table(t.upper(), spec))
open(f"{OUT}/r2_true_mono.md", "w").write("\n".join(r2))
print("\n".join(r1)); print("\n".join(r2))
