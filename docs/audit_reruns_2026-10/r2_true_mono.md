# R2 — true (train-signers-only) monolingual SSL (2026-10 audit)

Original `mono` for LSA64/KArSL/AUTSL loaded `bdsl_shubert_pretrain_bdsl_only` (Bangla SSL, not target-only). Original GSL mono (`src_gsl`) WAS target-only but its pool contained all 7 GSL signers incl. val (6) and test (7). Reruns pretrain on each target's TRAIN signers only (same SSL recipe as src_gsl: 30 ep, mask 0.3, K=64 pose_motion codes) and fine-tune with the original recipe. G_data = pool − mono.

### LSA64

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| lsa64 f10 mono (old)→true mono | 60.00 ± 1.11 (n=3: 60.94/58.44/60.62) | 66.67 ± 1.79 (n=3: 64.38/68.75/66.87) | +6.67 |
| lsa64 f10 pool [ref] | 60.84 ± 2.52 (n=3: 59.38/64.38/58.75) | — |  |
| lsa64 f10 xling [ref] | 61.77 ± 1.66 (n=3: 63.75/61.88/59.69) | — |  |
| lsa64 f25 mono (old)→true mono | 83.02 ± 1.56 (n=3: 83.44/80.94/84.69) | 86.04 ± 1.45 (n=3: 84.06/87.50/86.56) | +3.02 |
| lsa64 f25 pool [ref] | 82.81 ± 1.35 (n=3: 84.06/83.44/80.94) | — |  |
| lsa64 f25 xling [ref] | 83.02 ± 0.97 (n=3: 82.50/82.19/84.38) | — |  |
| lsa64 f50 mono (old)→true mono | 91.87 ± 0.76 (n=3: 91.87/90.94/92.81) | 92.92 ± 0.30 (n=3: 92.50/93.13/93.13) | +1.05 |
| lsa64 f50 pool [ref] | 92.71 ± 2.13 (n=3: 90.62/91.87/95.63) | — |  |
| lsa64 f50 xling [ref] | 93.33 ± 0.39 (n=3: 93.75/93.44/92.81) | — |  |

### KARSL

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| karsl f10 mono (old)→true mono | 60.05 ± 3.89 (n=3: 59.34/55.67/65.13) | 72.54 ± 3.08 (n=3: 76.83/69.74/71.04) | +12.49 |
| karsl f10 pool [ref] | 69.86 ± 3.44 (n=3: 70.21/65.48/73.88) | — |  |
| karsl f10 xling [ref] | 65.52 ± 1.98 (n=3: 64.18/64.07/68.32) | — |  |
| karsl f25 mono (old)→true mono | 92.43 ± 0.42 (n=3: 92.67/92.79/91.84) | 95.07 ± 0.29 (n=3: 95.39/94.68/95.15) | +2.64 |
| karsl f25 pool [ref] | 94.68 ± 1.10 (n=3: 95.63/95.27/93.14) | — |  |
| karsl f25 xling [ref] | 93.81 ± 1.54 (n=3: 95.74/91.96/93.74) | — |  |

### AUTSL

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| autsl f10 mono (old)→true mono | 68.98 ± 1.92 (n=3: 70.78/69.83/66.32) | 66.31 ± 0.18 (n=3: 66.57/66.21/66.16) | -2.66 |
| autsl f10 pool [ref] | 69.95 ± 2.27 (n=3: 72.54/67.02/70.30) | — |  |
| autsl f10 xling [ref] | 63.52 ± 3.50 (n=3: 58.58/66.23/65.75) | — |  |
| autsl f25 mono (old)→true mono | 84.31 ± 0.24 (n=3: 84.02/84.61/84.31) | 83.48 ± 0.20 (n=3: 83.75/83.43/83.27) | -0.83 |
| autsl f25 pool [ref] | 84.37 ± 0.23 (n=3: 84.65/84.36/84.09) | — |  |
| autsl f25 xling [ref] | 82.59 ± 2.86 (n=3: 83.05/85.83/78.88) | — |  |

### GSL

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| gsl f10 mono (old)→true mono | 75.35 ± 0.57 (n=3: 75.84/74.55/75.65) | 71.59 ± 6.21 (n=3: 76.22/75.74/62.82) | -3.75 |
| gsl f10 pool [ref] | 72.00 ± 4.23 (n=5: 77.37/73.13/71.88/64.40/73.22) | — |  |
| gsl f10 xling [ref] | 74.06 ± 2.79 (n=5: 74.00/70.28/76.87/71.71/77.42) | — |  |

## Reading (2026-10-10; job J02 = B)

Train-only mono pools match the official train splits exactly: LSA64 2,240 / KArSL 6,212 / AUTSL 28,130 / GSL 29,175 clips. BdSL's mono was already true same-language (`bdsl_only`), so it is unchanged.

| @10% | old mono | **true mono** | pool | xling | G_data old → **true** | xling−mono old → **true** |
|---|--:|--:|--:|--:|--:|--:|
| GSL | 75.35 | **71.59** | 72.00 | 74.06 | −3.35 → **+0.41** | −1.29 → **+2.47** |
| BdSL | 50.79 | 50.79 | 53.30 | 55.00 | +2.51 | +4.21 |
| LSA64 | 60.00 | **66.67** | 60.84 | 61.77 | +0.84 → **−5.83** | +1.77 → **−4.90** |
| KArSL | 60.05 | **72.54** | 69.86 | 65.52 | +9.81 → **−2.68** | +5.47 → **−7.02** |
| AUTSL | 68.98 | **66.31** | 69.95 | 63.52 | +0.97 → **+3.64** | −5.46 → **−2.79** |

R²(distance → G_data) goes from 0.270 to **0.044**. The law's G_align = xling − pool does not involve mono and is **unaffected**.

**Consequences for P1:**
- **"Cross-lingual pretraining beats monolingual at low labels" (C3) holds only for BdSL and GSL.** On LSA64, KArSL and AUTSL, true monolingual SSL beats xling @10% by 4.9, 7.0 and 2.8pp.
- **The LSA64 "scarcest target" argument is refuted.** It claimed mono ≈ scratch, so the second language is what helps. True mono is 66.67 vs scratch 60.5 (+6.2), and it beats xling (61.77).
- **"Pooling foreign data is broadly safe / take the data" is not supported.** With true mono, pooling HURTS LSA64 (−5.8) and KArSL (−2.7) @10%. KArSL's showcase "G_data +9.8" came from comparing against a Bangla backbone.
- **Higher budgets.** True mono ≥ pool and xling on LSA64 @25% (86.04 vs 82.81 / 83.02) and KArSL @25% (95.07 vs 94.68 / 93.81). At LSA64 @50% it ties (92.92 vs 92.71 / 93.33). On AUTSL @25%, true mono 83.48 is slightly below pool 84.37.
- **The GSL true mono has one collapsed seed** (62.82; the other two are 76.22 / 75.74 ≈ old 75.35). Its 71.59 mean is therefore noisy. Without the collapsed seed it is ≈ 76.0, which would make GSL G_data ≈ −4.0 and xling−mono ≈ −1.9. Re-seed before relying on it.
- These are single pretraining runs per backbone.
