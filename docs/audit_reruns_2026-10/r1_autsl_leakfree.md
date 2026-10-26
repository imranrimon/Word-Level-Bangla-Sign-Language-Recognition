# R1 — leak-free AUTSL gate / CORAL experiments (2026-10 audit)

Original runs pretrained on `data/ssl_pool_manifest_multisrc.json`, which contains ALL 36,289 AUTSL clips (incl. 3,741 official TEST + 4,418 VAL). Reruns: **noautsl** = same pool with every AUTSL clip removed (AUTSL genuinely foreign); **autsltrain** = AUTSL restricted to its 31 TRAIN signers (CORAL needs target-language clips, so it cannot run on noautsl). Fine-tune recipe identical (`config/autsl_si_lr01.yaml`, base_lr 0.01, 100 ep). Metric: seed-mean of per-seed best-epoch AUTSL val Top-1 (P1 convention).

### AUTSL @10%

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| uniform-weight align (pool6_uniform) | 70.24 ± 2.08 (n=3: 70.62/67.52/72.57) | 65.53 ± 2.56 (n=3: 64.96/62.72/68.92) | -4.70 |
| DG-XL gate, BdSL-keyed (dgxl) | 66.73 ± 3.63 (n=3: 63.35/65.07/71.77) | 72.21 ± 1.57 (n=3: 73.43/73.22/69.99) | +5.48 |
| AUTSL-anchored gate (dgxl_autsl) | 65.78 ± 0.76 (n=3: 65.50/65.01/66.82) | 69.91 ± 1.27 (n=3: 70.35/68.18/71.21) | +4.14 |
| corrected continuous gate (lambdad) | 68.41 ± 2.48 (n=3: 65.01/69.38/70.85) | 69.24 ± 1.68 (n=3: 67.11/69.40/71.21) | +0.83 |
| no-alignment pool, same data (new reference) | — | 69.10 ± 1.71 (n=3: 66.73/70.71/69.85) |  |
| CORAL (autsltrain pool) | 69.47 ± 0.35 (n=3: 69.69/69.74/68.97) | 65.49 ± 1.43 (n=3: 63.78/65.41/67.27) | -3.98 |
| no-alignment pool, autsltrain (CORAL reference) | — | 62.91 ± 1.57 (n=3: 63.01/60.93/64.78) |  |
| [ref] tab:law pool, Bangla+ASL (unaffected) | 69.95 ± 2.27 (n=3: 72.54/67.02/70.30) | — |  |
| [ref] tab:law xling, Bangla+ASL (unaffected) | 63.52 ± 3.50 (n=3: 58.58/66.23/65.75) | — |  |

### AUTSL @25%

| condition | original (leaky) | leak-free rerun | Δ |
|---|---|---|---|
| uniform-weight align (pool6_uniform) | 83.75 ± 1.02 (n=3: 84.22/82.34/84.70) | 80.86 ± 2.47 (n=3: 77.82/80.90/83.86) | -2.89 |
| DG-XL gate, BdSL-keyed (dgxl) | 83.30 ± 1.17 (n=3: 83.54/84.59/81.76) | 85.51 ± 0.67 (n=3: 85.06/86.46/85.02) | +2.22 |
| AUTSL-anchored gate (dgxl_autsl) | 83.70 ± 0.30 (n=3: 84.09/83.36/83.66) | 84.41 ± 0.70 (n=3: 83.48/84.61/85.15) | +0.71 |
| corrected continuous gate (lambdad) | pending | 83.94 ± 0.48 (n=3: 84.25/84.31/83.27) |  |
| no-alignment pool, same data (new reference) | — | 85.22 ± 1.49 (n=3: 84.27/84.07/87.32) |  |
| CORAL (autsltrain pool) | pending | 83.65 ± 0.75 (n=3: 82.64/83.88/84.43) |  |
| no-alignment pool, autsltrain (CORAL reference) | — | 83.83 ± 1.56 (n=3: 82.93/86.03/82.53) |  |
| [ref] tab:law pool, Bangla+ASL (unaffected) | 84.37 ± 0.23 (n=3: 84.65/84.36/84.09) | — |  |
| [ref] tab:law xling, Bangla+ASL (unaffected) | 82.59 ± 2.86 (n=3: 83.05/85.83/78.88) | — |  |

## Reading (2026-10-10; jobs J03 = A, J01 = C)

All comparisons are **one pretraining run per backbone** × 3 fine-tune seeds, so differences of a few points are within pretraining-run variance (see the P1 pretrain-seed fragility finding). Recomputed MMD(source→AUTSL-train), with the original value in parentheses: BdSL 0.145 (0.121), ASL 0.851 (0.836), LSA64 0.434 (0.446), KArSL 0.639 (0.724).

1. **The leak hid the far-target backfire.** On the genuinely foreign (AUTSL-free) pool, full-strength uniform alignment scores **65.53**, versus **69.10** for the matched no-alignment pool. G_align is therefore **−3.6pp @10%** and **−4.4pp @25%** (80.86 vs 85.22). The leaky originals showed no backfire: uniform 70.24 ≈ pool. P1's own prediction ("far ⇒ alignment backfires") is *confirmed* once the leak is removed. P1's tab:anchor claim that uniform-weight align "matches the true no-alignment pool" is an artifact of AUTSL being inside the pool.
2. **Boundary #2 reverses.** The leaky table had anchored 65.8 < uniform 70.2. Leak-free, anchoring is better:
   - AUTSL-anchored gate 69.91 vs uniform 65.53, i.e. +4.4.
   - BdSL-keyed DG-XL 72.21 vs uniform 65.53, i.e. +6.7.
   - The continuous λ(d) gate gives 69.24 ≈ pool 69.10. It reduces to pooling, as designed.

   **"No gate variant beats plain pooling" no longer holds @10%:** DG-XL is 72.21 vs pool 69.10 (+3.1, SDs about 1.6/1.7, n=3, single pretrain). At @25% every gate ties the pool (83.9–85.5 vs 85.22). Only uniform alignment is lower.
3. **CORAL (autsltrain pool, AUTSL train signers only).** CORAL scores 65.49 vs 62.91 for its matched no-alignment pool, so +2.6 @10% and a tie @25% (83.65 vs 83.83). CORAL still does not backfire relative to its own matched pool, so the qualitative "gentler operator avoids the penalty" survives. But the paper's comparison (CORAL 69.47 vs Bangla+ASL pool 69.95 / DANN 63.5) compared different pools; it needs restating on matched pools.
4. **Surprise.** Adding AUTSL *train*-signer clips to the pool *lowered* the no-alignment backbone (@10% 62.91 vs 69.10 without any AUTSL; @25% 83.83 vs 85.22). That is opposite to what in-language unlabeled data should do. It most likely reflects single-pretraining-run variance (or k-means codebook differences, since the codebook is refit per pool), but it is unexplained and should be checked with a second pretraining seed before anyone cites "target data hurts".
5. **@25% lambdad/CORAL** had no original runs (the paper only reported @10% for them); the leak-free @25% numbers above are new.
