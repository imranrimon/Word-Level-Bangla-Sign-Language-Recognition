# 2026-10 audit reruns R1 + R2 (expensive re-pretraining) — job map

Driver: `scripts/hpc/audit_reruns_2026-10/slurm_audit_reruns.sbatch` (JOBSET=A/B/C). Pools/configs: `scripts/hpc/audit_reruns_2026-10/prep.py`. Results: `python scripts/hpc/audit_reruns_2026-10/collect.py` → `r1_autsl_leakfree.md`, `r2_true_mono.md`. Per-run logs were written to `slurm_logs/audit_reruns/` (not distributed).

| job | id | contents |
|---|---|---|
| audit_A | J03 | R1 noautsl: prep (manifest, MMD src→AUTSL-train, configs) → k-means → 5 pretrains (pool6_uniform, dgxl, dgxl_autsl, lambdad_autsl, poolnoalign; 3 concurrent) → 30 AUTSL fine-tunes (@10/25% × 3 seeds) |
| audit_B | J02 | R2 train-signers-only mono: 4 k-means → 4 pretrains (LSA64, KArSL, AUTSL, GSL) → 24 fine-tunes (LSA64 10/25/50, KArSL 10/25, AUTSL 10/25, GSL 10; × 3 seeds) |
| audit_C | J01 | R1 autsltrain: k-means → CORAL + matched no-align pretrains → 12 AUTSL fine-tunes |

## Leak facts established while building these

| pool (original) | used as | contains |
|---|---|---|
| `ssl_pool_manifest_multisrc.json` (125,662) | AUTSL-target gate/CORAL backbones | all 36,289 AUTSL clips incl. 3,741 TEST (signers 6/14/27/30/34/39) + 4,418 VAL (1/11/16/18/25/35) |
| `ssl_pool_manifest_gsl.json` (40,826; `src_gsl` = GSL "mono") | GSL mono 75.35 | all 7 GSL signers incl. VAL signer 6 (5,828) and TEST signer 7 (5,823). The parent audit's basename check missed this because GSL label names (`glossesXXXX`) are not unique. |
| `ssl_pool_manifest_{lsa64,karsl,autsl}.json` | pairft source backbones (other targets only) | each target's full val+test |

Leak-free pools built (counts equal the official train splits exactly): LSA64 2,240; KArSL 6,212 (signers 1–2 minus the 10% seen-signer val carve); AUTSL 28,130; GSL 29,175; multisrc_noautsl 89,373; multisrc_autsltrain 117,503.

## Deviations from the directive (and why)
- **CORAL is run on `autsltrain`, not `noautsl`.** Its loss matches source vs *target-language* clip covariance (`model/shubert_pretrain.py`, `is_tgt = lang == target_lang`), so with zero AUTSL clips it has no target batch and degenerates to plain pooling. The leak-free CORAL keeps AUTSL TRAIN-signer clips only (standard SSL footing: unlabeled target-train data is allowed, val/test signers are not).
- **Matched no-alignment pools added** (`poolnoalign_noautsl`, `poolnoalign_autsltrain`) as same-data references; the paper's tab:anchor compared against the Bangla+ASL pool (69.95), which is a different pool.
- **GSL added to R2** (its "true mono" also leaked).
- `dgxl` keeps its BdSL-keyed distance vector (as the original); `dgxl_autsl`/`lambdad_autsl` get MMD(source→AUTSL-train) recomputed on the AUTSL-free pool (`r1_autsl_distances.json`, written by job A).
- Hard-gate AUTSL entry stays 0.0 with no AUTSL clips present: the hard gate is a relative re-weight, so ratios among present sources are unchanged.

## Outcome (2026-10-10)
All 11 pretrains and 66/66 fine-tunes completed with no failures. Wall-clock: audit_A 2d 14h41m, audit_B 19h28m, audit_C 1d 9h35m (k-means on scratch was slow: GSL ~2h). Results and readings: `r1_autsl_leakfree.md`, `r2_true_mono.md`; recomputed distances in `r1_autsl_distances.json`.
