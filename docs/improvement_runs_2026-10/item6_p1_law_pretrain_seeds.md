# Item 6: does P1's distance-gating law survive pretraining-seed noise?

Date: 2026-10-10/11. **No .tex edited, nothing committed.**

## Jobs (all COMPLETED)

| Job | What it ran |
|---|---|
| J10_1, J10_2 | Pool backbone (`bdsl_shubert_pretrain_bdsl_asl`) pretrained at seeds 1 and 2. Took 6h14m and 6h05m. |
| J11 / J15 | xling fine-tunes, pseed 1 / pseed 2. 6 languages (BdSL excluded, reused from item 5), fine-tune seeds 0–2. |
| J12 / J16 | xling fine-tunes, pseed 1 / pseed 2. GSL, LSFB, LIBRAS, fine-tune seeds 3–4. |
| J13 / J17 | pool fine-tunes, pseed 1 / pseed 2. 7 languages, fine-tune seeds 0–2. Ran `afterok` on J10_1 / J10_2. |
| J14 / J18 | pool fine-tunes, pseed 1 / pseed 2. GSL, LSFB, LIBRAS, fine-tune seeds 3–4. Ran `afterok` on the pretrains. |
| J19 | Control. The original seed-0 pool backbone, re-fine-tuned with the new materialised configs: GSL, KArSL and LSFB, fine-tune seed 0. |

Scripts and tools:

* Pretrain sbatch: `scripts/hpc/slurm_p1_pool_pretrain_seed.sbatch`
* Fine-tune sbatch: `scripts/hpc/slurm_p1_law_pseed_ft.sbatch`
* New configs: `config/{gsl,lsa64,karsl,autsl,lsfb,libras}_si_lr01_f10.yaml`. BdSL reuses item 5's `bdsl_block_gcn_si_lr01_f10.yaml`.
* Analysis: `tools/p1_law_pretrain_seed_analysis.py` (`--seeds all|012`). Raw outputs are in `raw/item6_analysis_{allseeds,seeds012}.txt`.

## Step 0: what the law fits

Source: `tools/gating_law_fit.py`, which reproduces the published single-run numbers.

* **Law quantity.** G_align = **xling − pool**, measured as best-epoch **VAL** Top-1 at **10% labels**.
  * Fine-tune recipe: `<lang>_si_lr01` configs + `--train-fraction 0.1`, subset_seed 42, lr 0.01, 100 epochs, fc re-initialised, **no gradient clipping**.
  * Values are seed-means over 3 fine-tune seeds, or 5 for GSL, LSFB and LIBRAS.
  * Mono is not part of the law; it only enters `tab:gdecomp` as G_data = pool − mono.
* **Backbones.** All seven targets fine-tune the **same two** backbones:
  * pool: `work_dir/bdsl_shubert_pretrain_bdsl_asl/pretrained_epoch30.pt` (BdSL+ASL pool, λ=0)
  * xling: `work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt` (same pool + GRL-adv 0.1 + contrastive 0.1)

  Both were pretrained once, at seed 0. Because the backbones are shared, a pretraining seed is a shared random effect across all seven law points.
* **Languages.** Seven in the fit.
  * Five calibration languages: BdSL, LSA64, KArSL, AUTSL, GSL.
  * Two "prospective" languages: LSFB and LIBRAS.
  * BISINDO is excluded (below the learnability floor), not used as a counterexample.
* **x-axis.** A fixed one-pass RBF-MMD on per-joint speed, measured pool→target: GSL 0.121, BdSL 0.143, LSA64 0.176, LIBRAS 0.315, KArSL 0.344, AUTSL 0.411, LSFB 0.480. It is label- and backbone-free, so it does not change with pretraining seed.
* **Reported statistics:**
  * calib-5 OLS: gain = 5.99 − 30.18·d, R² 0.997, d* 0.198, Spearman −1.00
  * all-7 OLS: gain = 4.78 − 24.78·d, R² 0.887, d* 0.193
  * calib-5 LOO MAE: 0.235pp
  * sign: 5/5 calib-LOO + 2/2 prospective = **7/7**
  * all-7 LOO MAE: 1.42
* **TEST.** P1 reports held-out TEST only for BdSL, LSA64 and AUTSL (`tab:test`). Those three languages were TEST-scored here as well, with the best-val checkpoint via `run_resumable_seed.py --test-config`.

## Integrity checks

* **102 new fine-tunes.** All reached epoch 99, have `model-best.pt`, and have 20 eval rows. All 9 TEST languages×arms×pseeds have their `_test` rows. There are no `.partial_*` directories.
* **Control (J19).** The original seed-0 pool backbone, re-fine-tuned with the new configs and runner, reproduces the paper's runs **bit-for-bit**: GSL 77.37, KArSL 70.21, LSFB 41.66. Item 5 showed the same for BdSL. So the differences below come from the pretraining seed, not from pipeline drift.
* **Backbone health.** Final pool pretraining loss is 0.3019 for the seed-0 original, 0.2982 for pseed 1 and 0.3055 for pseed 2, with the same ~735 s/epoch. The new backbones are as well-trained as the original.
* **Leak check.** The pool manifest (`ssl_pool_manifest_bdsl_asl.json`, 75,589 clips) contains 0 BdSL val/test-signer clips, and `tests/test_ssl_pool_no_leak.py` passes 3/3. The manifest and k-means targets are the same files as for seed 0.
* **Fine-tune seeds.** New pseeds use 5 fine-tune seeds for GSL, LSFB and LIBRAS, as the paper does. A seeds-0–2-only variant is also given.

## Per-language law quantity G = xling − pool (VAL, pp)

Pool/xling cells are seed-means. The pooled value is the mean over the 3 pretraining seeds. The hierarchical 95% CI resamples pretraining seeds, then fine-tune seeds within each drawn seed, with arms unpaired (20k draws). P(G>0) comes from the same bootstrap.

| target | MMD | pool / xling, ps0 (paper) | **G ps0** | pool / xling, ps1 | **G ps1** | pool / xling, ps2 | **G ps2** | **G pooled** | 95% hier CI | P(G>0) |
|---|---|---|---|---|---|---|---|---|---|---|
| GSL | 0.121 | 72.00 / 74.06 | **+2.06** | 75.39 / 74.41 | **−0.98** | 71.09 / 66.06 | **−5.02** | **−1.32** | [−6.83, +3.39] | 0.33 |
| BdSL | 0.143 | 53.28 / 54.96 | **+1.68** | 55.17 / 52.06 | **−3.10** | 53.74 / 57.91 | **+4.18** | **+0.92** | [−2.51, +4.36] | 0.69 |
| LSA64 | 0.176 | 60.84 / 61.77 | **+0.94** | 55.52 / 60.83 | **+5.31** | 57.81 / 53.86 | **−3.95** | **+0.77** | [−4.58, +5.39] | 0.62 |
| LIBRAS | 0.315 | 46.17 / 41.49 | **−4.69** | 41.03 / 42.06 | **+1.03** | 41.94 / 38.86 | **−3.08** | **−2.25** | [−5.90, +1.26] | 0.11 |
| KArSL | 0.344 | 69.86 / 65.52 | **−4.33** | 63.28 / 70.25 | **+6.98** | 66.11 / 67.41 | **+1.30** | **+1.31** | [−3.62, +6.74] | 0.68 |
| AUTSL | 0.411 | 69.95 / 63.52 | **−6.43** | 66.88 / 69.63 | **+2.75** | 69.29 / 64.88 | **−4.41** | **−2.70** | [−6.62, +1.30] | 0.10 |
| LSFB | 0.480 | 40.84 / 35.78 | **−5.06** | 34.80 / 40.95 | **+6.16** | 33.90 / 34.96 | **+1.05** | **+0.72** | [−4.34, +5.66] | 0.61 |

The ps0 column matches the paper. The table uses unrounded CSV values, so BdSL is +1.68 against the paper's +1.70 (computed from rounded 53.3/55.0).

Per-run VAL values for fine-tune seeds 0..4 are in `raw/item6_analysis_allseeds.txt`. They can be regenerated from `results_final.csv`.

**Sign per pretraining seed** (gate side at d*=0.193 in brackets):

| target | gate side | ps0 / ps1 / ps2 |
|---|---|---|
| GSL | [+] | + − − |
| BdSL | [+] | + − + |
| LSA64 | [+] | + + − |
| LIBRAS | [−] | − + − |
| KArSL | [−] | − + + |
| AUTSL | [−] | − + − |
| LSFB | [−] | − + + |

**Every one of the seven languages has the opposite sign at one or more pretraining seeds.**

**Pooled sign flips relative to the paper:**

* **GSL** goes from + to − (−1.32).
* **KArSL** goes from − to + (+1.31).
* **LSFB** goes from − to + (+0.72).

BdSL and LSA64 stay weakly positive, and AUTSL and LIBRAS stay negative. Every pooled CI contains 0. AUTSL and LIBRAS come closest to a robust backfire, with P(G>0) of about 0.10.

**Variance decomposition:**

* The SD of G across pretraining seeds averages **4.4pp** per language (range 3.0–5.7).
* The within-pretraining-seed fine-tune SE of G is **2.3pp**, so pretraining-seed variance is about 1.9× the fine-tune noise.
* The spread of pooled G across languages is only 1.67pp (SD), versus 3.65pp at seed 0. The distance-structured spread at seed 0 is not reproduced.

There is also a backbone factor. The seed-0 pool backbone is the best of the three pool seeds on 5/7 targets (all four far targets plus LSA64). The seed-0 xling backbone is the best xling seed on only 1/7. The paper's seed 0 combined an unusually good pool with a mediocre xling on the far targets, which manufactured the far-side backfires.

## Law refits

The paper protocol computes sign as calib-5 LOO plus the 2 prospective targets predicted from the frozen calib-5 fit. The frozen gate applies the paper's own d*=0.193 to each G.

| G source | all-7 OLS | R²(7) | d*(7) | calib-5 OLS | R²(5) | d*(5) | ρ(5) | LOO MAE(5) | sign (paper protocol) | LOO MAE(7) | frozen paper gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **paper (P1)** | +4.78 −24.78·d | 0.887 | 0.193 | +5.99 −30.18·d | 0.997 | 0.198 | −1.00 | 0.23 | 5/5+2/2 = **7/7** | 1.42 | 7/7 |
| pseed 0 (recomputed) | +4.73 −24.59·d | 0.889 | 0.192 | +5.91 −29.81·d | 0.997 | 0.198 | −1.00 | 0.22 | **7/7** | 1.39 | 7/7 |
| pseed 1 | −2.32 **+17.28**·d | 0.403 | 0.134 | −2.34 **+18.95**·d | 0.344 | 0.124 | +0.60 | 4.89 | 2/5+2/2 = **4/7** | 3.10 | **1/7** |
| pseed 2 | −2.12 +2.45·d | 0.009 | (0.864) | −0.94 −2.67·d | 0.007 | (−0.354) | +0.10 | 5.91 | 1/5+1/2 = **2/7** | 4.32 | **3/7** |
| **pooled (3 pseeds)** | +0.10 −1.62·d | **0.018** | 0.059 | +0.88 −4.51·d | **0.116** | 0.194 | −0.10 | **2.52** | 0/5+1/2 = **1/7** | 2.09 | **4/7** |

Notes on the refits:

* Spearman ρ over all 7 languages is −0.93 at ps0, **+0.71 at ps1** (the slope inverts), +0.18 at ps2, and **−0.18 pooled** (p=0.70).
* Under the paper protocol at pseed 1, the "4/7" includes both prospective languages called correctly. That only happens because the calib-5 fit has a positive slope there and predicts their positive gains. The direction of the law is reversed.
* **Fine-tune-seed sensitivity even at seed 0.** Restricting GSL, LSFB and LIBRAS to fine-tune seeds 0–2 (`--seeds 012`) gives GSL −0.41 at ps0. The ps0 result then drops to R²(5) 0.898, LOO MAE 1.34 and **5/7**. The paper's 7/7 relies on GSL's seeds 3–4.

**Bootstrap of the pooled law (5,000 draws).**

* Primary method (the "joint" draw): pretraining seeds are drawn once and shared across all 7 languages, since they use the same backbones. The draw is unpaired across arms. Fine-tune seeds are then resampled within each language × drawn pretraining seed.
* The variant that resamples each language independently gives essentially the same intervals.

| statistic | median | 95% interval | paper |
|---|---|---|---|
| R²(7) | 0.09 | [0.00, 0.64] | 0.887 |
| slope(7) | −1.7 | [−17.0, +13.1] | −24.78 |
| d*(7) | 0.19 | [−1.48, +1.92] (undefined) | 0.193 |
| R²(5) | 0.15 | [0.00, 0.88] | 0.997 |
| slope(5) | −4.5 | [−24.3, +14.0] | −30.18 |
| LOO MAE(5) | 3.1pp | [0.9, 6.4] | 0.235 |
| LOO MAE(7) | 2.6pp | [1.3, 4.3] | 1.42 |
| P(slope(7) < 0) | 0.59 | | — |
| P(7/7 sign, paper protocol) | **0.036** | | 7/7 |

The full distribution of the paper-protocol sign count is: 7/7 0.04, 6/7 0.08, 5/7 0.17, 4/7 0.19, 3/7 0.24, 2/7 0.18.

## Held-out TEST (P1 `tab:test` languages): G_test = xling − pool, pp

Each cell shows G_test, with pool / xling TEST Top-1 underneath.

| target | ps0 (paper) | ps1 | ps2 | pooled | 95% hier CI |
|---|---|---|---|---|---|
| BdSL | +7.33 (40.5 / 47.8) | −1.79 (48.0 / 46.2) | +0.12 (45.2 / 45.3) | **+1.89** | [−1.86, +6.14] |
| LSA64 | +5.00 (58.1 / 63.1) | −1.40 (58.1 / 56.7) | −7.50 (57.1 / 49.6) | **−1.30** | [−8.24, +5.32] |
| AUTSL | −7.01 (69.3 / 62.3) | +2.11 (66.5 / 68.6) | −4.10 (68.5 / 64.4) | **−3.00** | [−6.87, +0.85] |

Pooled TEST means:

| target | pool | xling |
|---|---|---|
| BdSL | 44.6 | 46.4 |
| LSA64 | 57.8 | 56.5 |
| AUTSL | 68.1 | 65.1 |

## Verdict

**The law fails under pretraining-seed replication.** The R²=0.997 / d*≈0.20 / 7/7 result is a property of one pretraining run of the shared pool/xling backbone pair, not of the target languages:

* **Seed 1** has a distance slope with the **opposite sign** (+17.3·d, ρ(7)=+0.71). Alignment helps the *far* targets (KArSL +7.0, LSFB +6.2, AUTSL +2.8) and hurts BdSL (−3.1).
* **Seed 2** has **no distance relation** (R² 0.01).
* **Pooled over 3 pretraining seeds:**
  * R²(5) drops from 0.997 to 0.12 and R²(7) from 0.89 to 0.02. Slope(7) goes from −24.8 to −1.6, with a CI spanning [−17, +13].
  * d* is undefined: the CI covers the whole real line.
  * Calib-5 LOO MAE rises from 0.24 to 2.5pp.
  * Paper-protocol sign falls from 7/7 to **1/7**. The frozen paper gate still gets **4/7**, roughly chance.
  * The bootstrap probability of reproducing 7/7 is about **4%**.
* **Languages whose pooled sign flips:** GSL (+→−), KArSL (−→+), LSFB (−→+). All seven flip at one or more individual pretraining seeds.
* **What partially survives:**
  * AUTSL (d=0.411) and LIBRAS (d=0.315) lean toward backfire. Their pooled G is −2.7 and −2.3, with P(G>0) about 0.10 on VAL, and AUTSL TEST is −3.0, CI [−6.9, +0.8]. Neither is significant at 95%.
  * No near target has a reliably positive alignment gain. BdSL is +0.9 and LSA64 +0.8, both with CIs spanning 0, and GSL is negative.
  * The defensible residual claim is weak: alignment is unlikely to help much anywhere at this budget, and may hurt some distant targets. "Distance predicts the sign" is not supported.
* **Root cause.** Every law point fine-tunes the same two backbones. The per-language G at a given pretraining seed is dominated by a backbone-pair effect: pretraining-seed SD of G is 4.4pp per language, against a seed-0 cross-language spread of 3.7pp. So the 7 points are not 7 independent observations. They are one draw of a correlated vector, and a single draw can show an arbitrarily clean line.

## P1 text that would change (old → new; nothing edited)

Line numbers refer to the earlier (single-run) version of the paper.

1. **Header comment, lines 10–11.** "gain=5.99-30.18*MMD (R^2=0.997 calib-5, 0.887 over 7), d*~0.20, LOOCV sign 7/7". Change to: pooled over 3 pretraining seeds, R² 0.12 (calib-5) / 0.02 (7), d* undefined, 1/7.
2. **Abstract, line 51.** Four claims are affected:
   * "As a predictor of the alignment gain it outperforms general-purpose transferability scores (R²=0.997 vs. LogME 0.11 and A-distance 0.31)". The pooled R² is about 0.12, roughly on par with LogME. The claim fails.
   * "approximately linear in distance, with a critical distance d*≈0.20". Not supported: the slope CI spans 0 and d* is undefined.
   * "correct on all seven sign languages … (7/7 sign-correct)". This is 1/7 pooled, and the bootstrap gives P(7/7) ≈ 0.04.
   * The framing needs to change from "label-free predictor" to a single-pretraining-run observation that does not replicate.
3. **Intro, line 65.** "distance→gain R²=0.997", "gets the helps-vs-backfires sign right 7/7", "a genuine out-of-sample, label-free selector, not a curve fit". All of these are contradicted by the pooled values above.
4. **Contribution C1, line 69.** R² 0.997 / 0.89, d*≈0.20, "validated 7/7", "correctly chooses between our two pre-specified regimes". Same change as item 2.
5. **Setup, line 109.** "Three seeds throughout (five for GSL, LSFB and LIBRAS)". Add that every law point uses a single pretraining seed of a shared backbone pair, and that pretraining-seed SD of the gain is about 4.4pp per language.
6. **Law intro, line 114.** "strongly linear (R²=0.89 over seven; near-perfect R²=0.997 over the five calibration languages)" becomes R² 0.02 / 0.12 pooled.
7. **Table `tab:law`, lines 117–129.** The pool / xling / gain columns become pooled values, and the helps/backfires labels change accordingly:

   | target | old gain | new gain (pooled) | label change |
   |---|---|---|---|
   | GSL | +2.06 | −1.32 | "helps" → "backfires" |
   | BdSL | +1.70 | +0.92 | |
   | LSA64 | +1.00 | +0.77 | |
   | LIBRAS | −4.68 | −2.25 | |
   | KArSL | −4.40 | +1.31 | "backfires" → "helps" |
   | AUTSL | −6.50 | −2.70 | |
   | LSFB | −5.06 | +0.72 | "backfires" → "helps" |

   Add a pretraining-seed CI column using the hierarchical CIs above.
8. **Line 135 (estimator / linear form).** Several statements fail:
   * "correct on all seven", the boxed "gain = 4.78 − 24.78·MMD", "5.99 − 30.18·MMD (R²=0.997, ρ=−1.000)", "d*≈0.20 (OLS 0.198 / 0.193, jackknife 0.192±0.006)", and the transition bracket [0.176, 0.315]. Replace them with the pooled fits: +0.10 − 1.62·d, R² 0.02; calib-5 +0.88 − 4.51·d, R² 0.12, ρ −0.10.
   * The LIBRAS "interpolated within 1.2pp" and LSFB "direction right" stories. Pooled LSFB is +0.72, against a calib-5 prediction of −1.29, so the sign is wrong.
9. **Line 139 + `tab:robust` (lines 423–445) + footnote.** "sign-correct 5/5 with d* confined to [0.177, 0.209]" and "R²≥0.99". Every robustness refit regresses on the seed-0 gains. Rerun `tools/distance_robustness.py` against the pooled gains; the R²s will collapse the same way.
10. **`tab:transfer`, lines 144–158.** The MMD R² of 0.997 becomes about 0.12 pooled. The energy (0.932), A-distance (0.310), kNN (0.317), LogME (0.108), #classes and size rows were all fit to seed-0 gains and need refitting against pooled G. The claim that motion distance "outperforms the standard estimators" is not supported.
11. **Figure `fig:law` (`fig1_gating_law.pdf`) and its caption/description, lines 171–181.** Both lines (all-7 and calib-5), d*=0.193, and the green/red regions need replotting with pooled points plus pretraining-seed error bars (done in the replication paper, `tools/paper/make_figs.py`). The caption sentences "R²=0.997 … d*≈0.20 cleanly separates … 7/7" change accordingly.
12. **`tab:loocv` and line 188.** Old values: calib LOO preds/actuals (GSL +2.55/+2.06 … AUTSL −6.31/−6.50), "MAE 0.24", "5/5", "+ prospective 7/7", "Spearman −1.000 / −0.929", and "LSFB −8.5 (measured −5.06), LIBRAS −3.5 (measured −4.68)". New pooled values: calib LOO **0/5**, MAE **2.52**, prospective 1/2 (LIBRAS correct, LSFB wrong), and Spearman −0.10 (5) / −0.18 (7).
13. **`tab:gdecomp`, lines 217–229.** The G_align column changes to the pooled values in item 7. "distance→gain R² 0.997" becomes about 0.12. G_data = pool − mono also changes, because pool is re-seeded here (pooled pool values: GSL 72.8, BdSL 54.1, LSA64 58.1, KArSL 66.4, AUTSL 68.7). Mono is only re-seeded for BdSL (item 5).
14. **Budget surface and abstention, lines 260–262.**
    * "d0(10%) = 0.199 [0.171, 0.220]", "b3 = +29.3", and the align/pool/uncertain bands all come from seed-0 backbones at every budget.
    * The 10% slice is now contradicted, and the target-clustered CI understates the true uncertainty because it ignores the shared-backbone (pretraining-seed) random effect.
    * At minimum, state that the surface is a single-pretraining-run estimate. The residual σ=1.60pp in the abstention rule is far below the 4.4pp pretraining-seed SD.
15. **Line 338 + `tab:test`, lines 340–350.**
    * Old sentence: "xling is the best condition @10% on the close targets (BdSL, LSA64)". New: pooled TEST xling is 46.4 vs pool 44.6 on BdSL, and 56.5 vs 57.8 on LSA64. LSA64 no longer holds.
    * Old sentence: "pooling alone trails monolingual SSL on both (40.5 vs 43.9; …)". Pooled BdSL pool TEST is 44.6, against pooled mono 45.75 from item 5, so the gap shrinks to about 1pp.
    * AUTSL "confirms the backfire on test" is weaker: pooled −3.0, CI [−6.9, +0.8].
16. **Line 402 ("What the estimator buys").** "validated out of sample (7/7 sign-correct: 5/5 … MAE 0.24pp … errors of 1.16 and 3.43pp)" and "genuine label-free selector … the central contribution". These do not survive.
17. **Limitations, line 406.** "The decision is correct on all seven languages", and the d* bracket discussion. This must become an explicit statement that the law does not replicate across pretraining seeds.
18. **Limitations, line 410.** Two claims change:
    * "the align-vs-pool sign is LOO-stable at 7/7"; "the robust claim is the sign-gate".
    * "Subset-composition variance … alignment gain positive in every draw (+1.0 to +5.4)". Those subset draws all use the seed-0 backbone. Across pretraining seeds, the BdSL G is +1.7 / −3.1 / +4.2.

    Add that pretraining-seed variance (about 4.4pp SD on G) exceeds fine-tune-seed variance by about 2×, and that every law point is a single-pretraining-run estimate.
19. **Downstream uses of "d*" as a reference.** These lean on a d* that is no longer identified:
    * line 287: AUTSL "MMD 0.411 ≫ d*"
    * line 303: λ-ablation "full-strength λ helps close targets but over-regularises a distant one"
    * line 488: the directed-matrix "predictor's extreme prediction holds"
    * line 586: BdSLW60↔BdSLW401 "MMD 0.096 ≪ d*"

    Those experiments also use single seed-0 backbones and were not re-seeded here.

## Not done / caveats

* **No reruns are needed**: every job completed, and the controls are bit-exact.
* Only the pool and xling arms were re-seeded, which is all the law needs. Mono was re-seeded only for BdSL (item 5), so G_data for the other languages still mixes pooled pool with seed-0 mono.
* The transferability-baseline R²s (`tab:transfer`) and the robustness sweep (`tab:robust`) were not recomputed against pooled G. Both are CPU-only refits once someone decides to update them.
* Three pretraining seeds is still a small n. Even so, one seed inverts the slope and another flattens it, so more seeds cannot restore R² ≈ 1. At most they would show a weak negative trend. The pooled point estimate is slope −1.6 with P(slope<0) of only 0.59.
