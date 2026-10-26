# P1: a checkpoint-aware predictor of the alignment gain G

Date: 2026-10-22. **No .tex edited, nothing committed.**

GPU used: 1.26 GPU-h (job J33, 48 min, 9 backbones; job J40, 28 min, the remaining 5 backbones). The full analysis output is in `analysis_output.md` (this directory).

## Question

Item 6 showed that the data-only "distance-gating law" fails across pretraining seeds. G = acc(xling) − acc(pool) depends on which backbone pair was drawn, and a backbone-free distance cannot see that.

This study asks whether any score that does look at the two checkpoints predicts G, using only target TRAIN clips plus the 10% labels that fine-tuning is allowed to use.

## Setup

**Units.** 7 languages × 7 pretraining seeds = **49 units**.

* Each unit's G is a mean over fine-tune seeds (BdSL / LSA64 / KArSL / AUTSL: seeds 0–2; GSL / LSFB / LIBRAS: seeds 0–4), as in the paper.
* Arms are the pool backbone (`bdsl_shubert_pretrain_bdsl_asl[_pseedN]`) and the xling backbone (`bdsl_shubert_pretrain_xlingual[_pseedN]`).
* Pretraining seed 0 is the original pair. Seeds 1–6 are the 7-seed replication (`p1_law_ft7`).
* G comes from VAL Top-1 in `results_final.csv`, parsed with `tools/p1_law_pretrain_seed_analysis.load`.
* A run counts only if its `model-99.pt` exists. All 49 units are complete, and seeds 0–2 match the item-6 table exactly.

**Clips.** Only TRAIN-split clips are used.

* The labelled set is the exact 10% subset each fine-tune trains on: a replica of `Feeder._subsample` with frac 0.1, seed 42.
* The unlabelled set is 3,000 random train clips (all 773 for LIBRAS).
* The source set is 2,000 clips drawn uniformly from the **real** pretraining pool, `data/ssl_pool_manifest_bdsl_asl.json`. Uniform sampling makes it proportional to the pool: BdSLW401 51,098, WLASL 21,083, BdSLW102a 3,408.

**Feature views.** Code is in `tools/p1_ckpt_predictor/extract.py`.

* **ft view:** fine-tune preprocessing (center_choose to 120 frames + joint-0 centering) and the fine-tune architecture (stride True). BatchNorm statistics are re-estimated on the target clips (AdaBN), which approximates train-mode BN at the start of fine-tuning.
* **ftj view:** the same, but BN is estimated on source + target together. It is used for source-vs-target distances.
* **ssl view:** what pretraining saw. Raw pose, stride False, pretrained BN statistics, per-frame features mean-pooled.

**Predictors.** Each predictor is dir × (score(xling) − score(pool)), with the direction fixed a priori. Code is in `tools/p1_ckpt_predictor/score.py`.

| family | score | notes |
|---|---|---|
| ckpt, label-free | `ssl_loss`, `ssl_acc`, `ssl_gap` | Masked code-prediction cross-entropy / accuracy of each backbone's **own SSL head** on target clips: k-means codes from the pool's centres (pose_motion), 4 fixed masks at ratio 0.3. `ssl_gap` = target loss − source loss. The head exists only in `train_state.pt` for pseeds 1–6; the original seed-0 runs saved the backbone only, so n = 42. |
| | `mmd_ftj`, `mmd_ssl`, `md_ftj`, `md_ssl` | Feature-space RBF-MMD and normalised mean discrepancy, source pool → target, under each backbone. |
| | `erank_ft`, `erank_ssl`, `unif_ft` | Effective rank (Roy & Vetterli 2007) and uniformity (Wang & Isola 2020) of the target features. |
| | `1-cka_ft`, `1-cka_ssl` | Linear CKA (Kornblith et al. 2019) between the pool and xling target features. This is a pair score with no sign. |
| ckpt + 10% labels | `logme` | LogME (You et al., ICML 2021), using `tools/logme_transfer.logme`. |
| | `ncti` (+ components `seli` / `ncc` / `vc`) | NCTI (Wang, Luo, Zheng, Huang, Baktashmotlagh, ICCV 2023), following the official recipe in github.com/BUserName/NCTI `metrics.py`. Features are reduced with PCA-64, then fit with shrinkage LDA. NCTI = mm(seli) + mm(ncc) − mm(vc), where mm is min-max over the model hub of all 14 backbones per language. |
| | `probe_acc`, `ncc_acc`, `knn_acc` | 3-fold cross-validated logistic-probe accuracy, and leave-one-out nearest-class-mean and 1-NN accuracy. |
| data-only | `mmd_paper` | The paper's one-pass values from `gating_law_fit.py`. |
| | `mmd_realpool` | Recomputed against the real pool (n=600, mean of 5 draws). Code: `tools/p1_ckpt_predictor/datammd.py`. |
| | `mmd_realpool_masked` | The same, but speed is used only where the joint is detected in both frames (dropout-masked). |

**Evaluation.**

* Spearman ρ and Kendall τ, with a 95% CI from a bootstrap that resamples pretraining seeds (5,000 draws).
* Sign accuracy at 0. For `mmd_paper` the paper's frozen gate d* = 0.193 is used instead.
* Leave-one-language-out (LOLO) and leave-one-pretraining-seed-out (LOSeedO) OLS prediction, reporting MAE and sign. The reference is a constant that predicts the training mean.
* A scale-free decomposition: mean within-language ρ (across seeds), mean within-seed ρ (across languages), and ρ of the seed means.
* Westfall–Young family-wise permutation p-values over the 21 signed predictors.

## Outcome noise: the ceiling

**Size of G.** Over the 49 units, G has mean +0.18pp and SD 3.62pp. The mean fine-tune SE of G is **2.57pp**.

**Variance components** (method of moments on the 7×7 table, in pp²):

| component | pp² | share |
|---|---|---|
| fine-tune noise | 6.6 | ≈49% |
| language × pretraining-seed interaction | 4.3 | |
| pretraining-seed main effect | 1.9 | |
| language main effect | 0.65 | ≈5% |

**A data-only predictor can only target the language main effect, which is about 5% of the observed variance.** The signal a checkpoint-aware score could capture, the seed effect plus the interaction, is about 46%.

**Ceilings for a perfect predictor of the true G:**

* Reliability of observed G is 1 − mean SE² / Var G = **0.50**.
  * Pearson ceiling 0.70; simulated Spearman ceiling **0.73**.
  * Split-half check (fine-tune seeds even vs odd): r = 0.24, Spearman–Brown 0.39, ceiling about **0.62**.
* MAE floor from fine-tune noise alone: **1.85pp**.

**Reference MAEs:**

* Constant predictor: LOLO 3.06pp, LOSeedO 3.07pp.
* Language-mean oracle: knowing each language's mean G at the other 6 seeds gives LOSeedO MAE **3.12pp**, which is worse than the constant (3.07pp). Language identity has no transferable information about G across pretraining seeds.

## Results (n=49; 42 for the SSL-head scores)

LOLO and LOSeedO MAEs are in pp, with the constant baseline in brackets. The last three columns are the scale-free decomposition.

| predictor | type | Spearman ρ [95% seed-clustered CI] | sign@0 | LOLO MAE (const) | LOSeedO MAE (const) | LOSeedO sign | ρ within-lang | ρ within-seed | ρ seed-means (n=7) |
|---|---|---|---|---|---|---|---|---|---|
| **mmd_paper** | data-only | +0.06 [−0.30, +0.42] | 0.57 (gate d*) | 3.10 (3.06) | 3.17 (3.07) | 0.39 | – | −0.03 | – |
| mmd_realpool | data-only | +0.03 [−0.38, +0.33] | – | 3.18 (3.06) | 3.17 (3.07) | 0.33 | – | −0.09 | – |
| mmd_realpool_masked | data-only | −0.03 [−0.43, +0.25] | – | 3.25 (3.06) | 3.16 (3.07) | 0.35 | – | −0.16 | – |
| ssl_loss | ckpt | −0.15 [−0.45, +0.11] | 0.50 | 3.16 (2.97) | 2.88 (2.99) | 0.55 | −0.10 | −0.23 | −0.09 |
| ssl_acc | ckpt | −0.10 [−0.38, +0.22] | 0.50 | 3.07 (2.97) | 2.95 (2.99) | 0.55 | −0.17 | −0.11 | +0.09 |
| ssl_gap | ckpt | −0.13 [−0.41, +0.13] | 0.43 | 3.09 (2.97) | 2.85 (2.99) | 0.55 | −0.09 | −0.23 | −0.09 |
| mmd_ftj | ckpt | −0.09 [−0.44, +0.21] | 0.41 | 3.13 (3.06) | 3.23 (3.07) | 0.35 | −0.01 | −0.11 | −0.11 |
| mmd_ssl | ckpt | −0.18 [−0.55, +0.21] | 0.47 | 3.21 (3.06) | 3.30 (3.07) | 0.39 | −0.22 | −0.04 | −0.21 |
| md_ftj | ckpt | −0.05 [−0.41, +0.28] | 0.45 | 3.15 | 3.24 | 0.35 | −0.06 | +0.01 | −0.11 |
| md_ssl | ckpt | −0.07 [−0.46, +0.31] | 0.51 | 3.32 | 3.24 | 0.41 | −0.20 | +0.09 | −0.18 |
| erank_ft | ckpt | −0.16 [−0.36, +0.04] | 0.51 | 3.10 | 3.04 | 0.47 | +0.04 | −0.20 | −0.21 |
| erank_ssl | ckpt | +0.01 [−0.35, +0.34] | 0.45 | 3.16 | 3.22 | 0.33 | +0.22 | +0.01 | +0.11 |
| unif_ft | ckpt | −0.02 [−0.20, +0.22] | 0.47 | 3.11 | 3.11 | 0.37 | −0.19 | +0.03 | −0.57 |
| 1−CKA ft / ssl | pair | −0.01 / +0.04 | – | 3.14 / 3.18 | 3.12 / 3.11 | 0.35 / 0.37 | | | |
| **logme** | ckpt + 10% lab | −0.08 [−0.34, +0.20] | 0.45 | 3.16 (3.06) | 2.99 (3.07) | 0.45 | +0.14 | −0.11 | −0.43 |
| **ncti** | ckpt + 10% lab | −0.02 [−0.40, +0.29] | 0.51 | 3.13 (3.06) | 3.14 (3.07) | 0.33 | +0.22 | −0.11 | −0.11 |
| ncti_seli / ncc / vc | ckpt + 10% lab | +0.05 / +0.06 / −0.12 | 0.53 / 0.45 / 0.41 | 3.05 / 3.08 / 3.16 | 3.11 / 3.15 / 3.06 | | | | |
| probe_acc | ckpt + 10% lab | +0.05 [−0.27, +0.45] | 0.57 | 3.10 | 3.17 | 0.33 | +0.16 | +0.18 | −0.36 |
| ncc_acc | ckpt + 10% lab | −0.04 [−0.43, +0.37] | 0.53 | 3.08 | 3.26 | 0.39 | +0.19 | +0.05 | −0.11 |
| knn_acc | ckpt + 10% lab | −0.28 [−0.60, +0.14] | 0.45 | 3.08 | 3.22 | 0.39 | −0.08 | −0.22 | −0.54 |
| *noise ceiling* | | *≈0.62–0.73* | | *floor 1.85* | *floor 1.85* | | | | |

**Significance after multiple comparisons.**

* The largest |ρ| is `knn_acc` at −0.28, which is in the wrong a-priori direction. Its raw permutation p is 0.047, but its family-wise p is 0.55.
* Every other predictor has a family-wise p of 0.98 or more.
* No 95% CI excludes 0.

**Best LOSeedO MAE gains over the constant** (pretraining-seed-clustered 95% CIs):

* `ssl_gap`: −0.14pp [−0.30, +0.02]
* `ssl_loss`: −0.10pp [−0.27, +0.08]
* `logme`: −0.08pp [−0.18, +0.03]

These are tiny (3–5% of the MAE), not significant, and their sign accuracy is about chance at 0.55 / 0.55 / 0.45.

**The paper's data-only law.**

* Within individual pretraining seeds, Spearman(−MMD, G) is +0.93, −0.71, −0.18, −0.46, −0.29, +0.18 and +0.32. Only seed 0, the paper's seed, shows the law.
* The frozen gate d* = 0.193 is sign-correct on 0.57 of units.

**Distances recomputed against the real pool.** The real-pool distances reorder the languages:

| target | paper | real pool | real pool, dropout-masked |
|---|---|---|---|
| BdSL | 0.143 | 0.063 | 0.084 |
| LSA64 | 0.176 | 0.222 | 0.449 |
| KArSL | 0.344 | 0.509 | 0.500 |
| AUTSL | 0.411 | **0.173** | 0.250 |
| GSL | 0.121 | 0.113 | 0.177 |
| LSFB | 0.480 | 0.672 | **0.261** |
| LIBRAS | 0.315 | **0.140** | 0.246 |

* Rank agreement with the paper's values is ρ = 0.79 for the real-pool variant and only 0.57 for the dropout-masked variant.
* AUTSL and LIBRAS move to the near side on the real pool. LSFB moves to the middle once dropout jumps are masked.
* So the paper's "far" group depends on which source is used and how detector dropout is handled. None of the three variants predicts G (table above).

## Verdict

**No score predicts G across pretraining seeds.** This holds for label-free checkpoint-aware scores (the backbone's own masked-prediction loss, feature MMD, CKA, effective rank, uniformity), for label-using transferability estimators (LogME, NCTI, probe / NCC / kNN on the 10% labels), and for the data-only MMD in any of its three variants.

* All |ρ| are 0.28 or less, against a noise ceiling of about 0.62–0.73. No CI excludes 0, and no predictor survives family-wise correction.
* No predictor beats a constant on LOLO or LOSeedO MAE by a significant margin.
* No predictor tracks the per-seed mean of G: the seed-means column has no positive association.
* The thing to predict is mostly the backbone-pair effect: seed main effect about 1.9 pp², interaction about 4.3 pp², language effect only about 0.65 pp². Even scores computed from the actual checkpoints, on the actual target clips, cannot see it.
* At 10% labels, G behaves like an unpredictable random effect of the pretraining draw plus fine-tune noise. "Which arm wins" is not a stable, estimable property of a (language, backbone) pair.

## Caveats

1. **What a checkpoint-aware predictor saves.** It needs both pretrained checkpoints, so at best it saves the fine-tuning runs, not the pretraining. Even a perfect one would not answer the paper's original practical question ("should I pretrain with alignment for target X?") before pretraining. Here it does not work either way.
2. **The backbones include scrambled WLASL data.** Per the code audit, every backbone (pool and xling, all 7 seeds) was pretrained on a pool in which the 21,083 WLASL clips (28%) have a scrambled skeleton: the converter used the wrong Kaggle V3 indices, so the "body" slots move like a hand and the "hand" slots are nearly static.
   * The scores measure the backbones as actually trained, so the null result above is about these backbones.
   * Whether correctly-converted WLASL would make G more stable or more predictable is untested.
   * The source-pool sample used for the feature-space and real-pool distances also contains these scrambled clips (proportionally), as the pretraining did.
3. **SSL-head scores are missing for seed 0.** They exist for pseeds 1–6 only (n=42), because the original seed-0 runs saved backbone-only checkpoints.
4. **Small-sample degeneracy.**
   * NCTI on LIBRAS (79 labelled clips, 20 classes) and LogME there (n < d) are degenerate. NCTI's in-sample LDA reaches ncc = 1 and vc is constant, so for LIBRAS NCTI reduces to seli.
   * LIBRAS LogME differences are about 100× larger than other languages'. That outlier inflates LogME's LOLO MAE in interim runs but not the final pooled ρ.
   * The other languages have 256–2,851 labelled clips.
5. **Arbitrary preprocessing choices.** AdaBN (ft), joint AdaBN (ftj) and the raw / pretrained-BN view (ssl) are three reasonable ways to probe a checkpoint before fine-tuning. None of the three views works.
   * A few-epoch fine-tune proxy was not tried, because that is fine-tuning.
   * Alignment-type scores that need augmentation pairs were not tried.
6. **Different targets from the paper.** G is VAL Top-1 at 10% labels, as in the paper's law. TEST is not used here.

## Suggested paper framing (not inserted)

* Replace the "label-free selector" claim with a negative result. Across 7 pretraining seeds, the align-vs-pool gain at 10% labels is dominated by the pretraining draw and fine-tune noise:
  * fine-tune noise is about 49% of the variance, the seed effect plus interaction about 46%, and the language effect about 5%;
  * neither a data-only distance nor checkpoint-aware transferability scores predict it (LogME, NCTI, own-SSL-loss, feature MMD / CKA: all |ρ| ≤ 0.28 vs a ceiling of about 0.65, none above the constant baseline).
* Present this as a methodological lesson: transfer laws fitted on a single pretraining run of a shared backbone pair are one draw of a correlated random effect. Report pretraining-seed replication, and evaluate predictors on (target × pretraining-seed) units against a noise ceiling.
* If a positive claim is needed, it is the weak one from item 6: alignment does not help reliably anywhere at this budget, and the paper's d* and "far" group also depend on the source definition (real pool vs WLASL + BdSLW60) and on dropout handling.

## Files

* `tools/p1_ckpt_predictor/extract.py` extracts features: GPU, idempotent, only TRAIN clips.
* `tools/p1_ckpt_predictor/datammd.py` computes the data-only MMD against the real pool: plain and dropout-masked.
* `tools/p1_ckpt_predictor/score.py` has two steps: `compute` (per-backbone scores, written to `work_dir/p1_ckpt_predictor/scores_{backbone,pair}.csv`) and `analyze` (the evaluation, written to `work_dir/p1_ckpt_predictor/analysis.json`).
* `scripts/hpc/slurm_p1_ckpt_predictor.sbatch` runs extraction + data-MMD + compute. It is requeue-safe and resumable.
* Feature caches are in `work_dir/p1_ckpt_predictor/<lang>/<arm>_ps<p>.npz` (2.1 GB), plus `data_mmd.json` and `_src_inputs.npz`.
* To reproduce: `python tools/p1_ckpt_predictor/score.py analyze --n-boot 5000 --n-perm 5000 --json work_dir/p1_ckpt_predictor/analysis.json`.
