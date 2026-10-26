# Item 5 — P1 headline BdSL @10% xling−mono gain: pretraining-seed replication

Date: 2026-10-08/09. Jobs: J04 (mono pretrain seeds 1,2), J05 (xling pretrain seeds 1,2),
J06 / J07 (fine-tunes on the new mono / xling backbones, `afterok` on J04 / J05),
J08 (control: original backbones re-fine-tuned with current code), J09 (control: width-study
1× backbones with the headline unclipped recipe). All six jobs COMPLETED (pretrains took 8h41m and 12h11m; each fine-tune job about 36 min).

## What the headline is (step 0)

* **P1 Table `tab:sig`, BdSLW60 @10%: xling−mono = +4.2pp** (and SSL mono−scratch = +4.8pp). These are
  **seed-means of best-epoch VAL Top-1** (val signer 15) over fine-tune seeds 0,1,2 of
  `bdsl_block_gcn_si_lr01_f10_{xling,mono,scratch}_seed{0,1,2}`: xling 54.96, mono 50.79, scratch
  45.95 → +4.18 / +4.84 (aggregation = `tools/p1_results_io.seed_mean`). The held-out TEST
  version is P1 Table `tab:test` (`*_seed<S>_test`, best-val checkpoint scored on test signers 2,13):
  xling 47.79, mono 43.93 → **+3.9pp**.
* Backbones (from the saved `work_dir/*/config.yaml`):
  mono = `work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt`
  (config `bdsl_shubert_pretrain_bdsl_only.yaml`, BdSL-only pool), xling =
  `work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt` (config
  `bdsl_shubert_pretrain_xlingual.yaml`, BdSL+ASL pool, GRL adv 0.1 + contrast 0.1). Both were
  pretrained once, with `--seed 0`.
* Fine-tune recipe: `config/bdsl_block_gcn_si_lr01.yaml` + `--train-fraction 0.1` (stratified
  subset, `fraction_seed` 42), lr 0.01, 100 epochs, fc re-initialised, **no gradient clipping** (the
  runs predate the `grad_clip_norm` knob).
* **The `_d0/_d1/_d2/_d4/_d43` variants are label-subset draws, not pretraining seeds.** They use
  the same seed-0 backbones and differ only in `subset_seed` (0/1/2/4/43, versus the default 42). Their
  VAL xling−mono gains are +4.7, +1.4, +5.4, +3.3, +4.3. So for this one pretraining run, the gain
  holds up across different 10% label subsets. No `_d44` runs exist for BdSL.
* The width study's 1× arm (`bdsl_ft10_*_w1`, −0.2pp) did re-pretrain, again with `--seed 0`.
  **`mono_w1` is bit-identical to the original mono backbone** (max |Δw| = 0). `xling_w1` differs
  (max |Δw| = 2.7) because its pretraining was resumed mid-run, which changed the RNG stream. Its
  fine-tune also used grad-clip 5.0.

## What was run

* 4 new pretrains via `main_pretrain.py --seed {1,2} --work-dir work_dir/<orig>_pseed{1,2}`, using
  the original, unedited configs. The manifests and k-means target `.npz` files are the same ones the
  seed-0 backbones used (the targets were built on Aug 3 at 19:00, before the originals). Only the
  init, masks, crops and shuffle change. The SSL-pool leak test passes (`tests/test_ssl_pool_no_leak.py`, 3/3).
  Code drift since the originals is behaviour-neutral for these configs (the gate, latent and CORAL
  terms are all off).
* 12 fine-tunes: for each new backbone, seeds 0,1,2 with the headline recipe, materialised as
  `config/bdsl_block_gcn_si_lr01_f10.yaml` (lr01 + train_fraction 0.1 + subset_seed 42 + grad_clip 0).
  They were run via `tools/run_resumable_seed.py --init-weights <backbone>` and then scored on TEST
  with `config/bdsl_block_gcn_si_lr01_test.yaml`. Experiment names are `bdsl_block_gcn_si_lr01_f10_{mono,xling}_ps{1,2}_seed{S}[_test]`.
* Controls (not pooled):
  * `_rerun`: the original seed-0 backbones run through the current code. This reproduces the
    headline **bit-for-bit** (all 12 val and test numbers identical), so the fine-tune pipeline
    has not drifted.
  * `_w1bb`: the width-study 1× backbones with the headline *unclipped* recipe.
* Scripts: `scripts/hpc/slurm_p1_pretrain_seed.sbatch`, `scripts/hpc/slurm_p1_pseed_ft.sbatch`,
  analysis `tools/p1_pretrain_seed_analysis.py --clip-boot`.

## Results (Top-1 %, mean ± SD over 3 fine-tune seeds)

| pretrain seed | VAL mono | VAL xling | **VAL xling−mono** | TEST mono | TEST xling | **TEST xling−mono** |
|---|---|---|---|---|---|---|
| 0 (original headline) | 50.79 ± 0.79 | 54.96 ± 0.70 | **+4.18** | 43.93 ± 4.24 | 47.79 ± 2.21 | **+3.86** |
| 1 (new) | 56.89 ± 0.77 | 52.06 ± 1.91 | **−4.83** | 45.59 ± 2.69 | 46.20 ± 0.53 | **+0.61** |
| 2 (new) | 54.91 ± 2.55 | 57.91 ± 2.08 | **+3.00** | 47.74 ± 0.71 | 45.35 ± 3.30 | **−2.39** |
| **pooled (3 pseeds × 3 ft seeds)** | 54.20 | 54.98 | **+0.78** [−3.27, +4.91] | 45.75 | 46.45 | **+0.69** [−1.98, +3.70] |
| control: rerun of seed 0 (current code) | 50.79 | 54.96 | +4.18 | 43.93 | 47.79 | +3.86 |
| control: width-1× backbones, unclipped | 50.79 | 55.01 | +4.23 | 43.93 | 45.79 | +1.86 |

Per-run values (fine-tune seeds 0/1/2):
pseed1 mono val 57.71/56.79/56.18, xling val 51.91/54.05/50.23; mono test 47.11/47.18/42.49, xling test 45.93/45.86/46.81.
pseed2 mono val 52.06/56.95/55.73, xling val 56.79/60.31/56.64; mono test 47.91/48.35/46.96, xling test 41.54/47.47/47.03.

Pooled CIs come from an unpaired **hierarchical bootstrap**: within each arm, pretrain seeds are
resampled first, then fine-tune seeds within each drawn pretrain seed (20k resamples).
* VAL: +0.78pp, 95% CI [−3.27, +4.91], P(gain ≤ 0) = 0.36. Adding a clip-level third tier gives [−3.72, +5.38], P = 0.38.
* TEST: +0.69pp, 95% CI [−1.98, +3.70], P(gain ≤ 0) = 0.33. With the clip tier: [−2.33, +3.91], P = 0.35.
* The SD of the per-pretrain-seed gain is 4.9pp on VAL and 3.1pp on TEST. That is larger than the
  headline effect itself.

## Verdict on +4.2

**+4.2 does not hold up. Once pretraining-seed variance is included, it effectively vanishes.** It
comes from a single pretraining run in which the monolingual backbone happened to be the weakest of
three:
* Seed-0 mono scores 50.8 VAL; the other two mono seeds score 56.9 and 54.9.
* Across seeds, the gain has the sign pattern +4.2 / −4.8 / +3.0 on VAL and +3.9 / +0.6 / −2.4 on TEST.
* The pooled gain is about +0.7–0.8pp, and both CIs comfortably include 0.

The SSL-over-scratch benefit does survive, and on VAL it grows. Pooled mono VAL is 54.2 against
45.95 for scratch, so mono−scratch is about +8.2pp versus +4.8pp in the paper. On TEST it is small:
45.75 versus 44.25, about +1.5pp. Scratch has no pretraining-seed dimension, so these numbers are
not re-seeded.

The controls also explain the width-study discrepancy. Using the same seed-0 mono backbone, the
headline unclipped recipe on the width-1× xling backbone gives +4.2 VAL and +1.9 TEST. So the 1×
"−0.2" in the width study comes from grad-clip 5.0 fine-tuning plus a perturbed xling run. It is not
a different mono backbone. Neither +4.2 nor −0.2 is reproducible at the pretraining-seed level.

## P1 text that would change (no .tex edited)

1. **Table `tab:sig`, BdSLW60 @10% row** (earlier version, line ~272). "+4.2pp, p<10⁻⁴" should become about
   +0.8pp (hierarchical CI [−3.3, +4.9]). The clip-wise p<10⁻⁴ comes from a single pretraining run and
   is not a valid test of the claim. "+4.8pp" mono−scratch becomes about +8.2pp (VAL, pooled mono).
2. **Table `tab:test`, BdSLW60 row** (line ~347): "+3.9pp" becomes about +0.7pp, CI [−2.0, +3.7]. The
   sentence at line ~338, "xling is the best condition @10% on the close targets (BdSL, …)", no
   longer holds for BdSL. Pooled TEST is xling 46.5 versus mono 45.8.
3. **Line ~381 (backbone/capacity paragraph)** and the **`tab:backbone` caption (line ~384)**. These
   currently say "at the same 1× width the headline recipe gives +4.2pp, so the BdSL alignment gain is
   itself sensitive to the pretraining run and fine-tuning clipping". That should be restated as:
   across 3 pretraining seeds, the gain is +0.8pp [−3.3, +4.9]. The capacity-floor argument should
   also be softened, because the width deltas (−5.4 / −0.2 / +0.6) all come from single pretraining
   runs, and the per-seed gain SD here is about 4.9pp.
4. **Every use of mono = 50.8 / xling = 55.0 as the BdSL anchor**:
   * the law table rows at lines ~124 and ~224 (mono 50.79, pool 53.30, xling 55.00, G_align = +1.70);
   * `tab:lambda` (line ~312–314);
   * the CORAL table (line ~331);
   * line ~87 (50.8 → 44.6);
   * lines ~520 and ~593.

   xling alone spans 52.1–57.9 across pretraining seeds (SD of seed-means about 2.9), which exceeds
   the BdSL G_align of +1.70 used in the distance-law fit. The pool backbone was **not** re-seeded
   here, so G_align at other seeds is unmeasured. The BdSL calibration point should at least carry a
   pretraining-seed error bar. In the λ-ablation, adv_only 56.8 and con_only 57.1 are "+6pp over mono"
   only against the unlucky 50.8; against pooled mono (54.2) they are about +2.6 to +2.9, within
   pretraining-seed noise.
5. **Limitations / statistical-footing bullet** (line ~410): this should state that BdSL headline
   cells are single-pretraining-run estimates, and that pretraining-seed variance (SD about 3–5pp on
   the gain) dominates fine-tune-seed variance.
