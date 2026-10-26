# P1 distance-gating law, v2 (fixed inputs): RESULTS

**VERDICT (binding, pre-registered): NOT SUPPORTED.** C1 fails: the 95% CrI of the slope b is
[−11.56, +2.61] pp per unit MMD²_u (posterior median −4.59, P(b<0) = 0.905). C2 passes exactly at the
threshold: LOLO sign accuracy is 6/8 (needs 6). Under the decision rule both must hold.

* Analysis: `tools/p1_fix/refit_v2.py --out docs/p1_fix_2026-10/results`. It ran once on v2 data, on a compute
  node (job J45, 2026-10-23 08:43:09–08:44:07 EDT). Log: the job's stdout (not distributed).
  Output: `docs/p1_fix_2026-10/results/refit_v2.json`. **Every number below comes from that JSON.** The
  exception is the section marked *derived*, whose numbers are computed from the JSON (and from the v1
  refit's `pooled_gains.json`) by `tools/paper/make_figs.py`, which asserts them.
* Pre-registration: `PREREG.md`. SHA-256 `bd28232ae8703eaf28cf6b261238f8eb41f5f456970ad497fe4d403d6c95fd75`,
  mtime 2026-10-22 05:17:19 EDT, read-only. The hash was re-verified for this report (`sha256sum`, 2026-10-23), and the same hash is
  embedded in `refit_v2.json` (`prereg_sha256`), so the file is **unchanged**. Addendum A (the frozen
  distances, `PREREG_addendum_A_distances.md`) has SHA-256 `e19f78f8e749183022e0955be59f8c3725083108b9458aa9f0f459962c0d9757`, written at 05:48:22 EDT. Both
  files were locked before any v2 backbone or fine-tune existed.
* Units: G = xling − pool, best-epoch VAL Top-1 (%) at 10% labels. Cells are (language ℓ, pretraining seed p),
  each with 5 fine-tune seeds per arm. G_ℓp is the difference of arm means (arms are unpaired). "Pooled G" is
  the mean of G_ℓp over p = 0, 1, 2.

## Completeness and exclusions

* **240 of 240 fine-tunes are complete** (8 languages × 2 arms × 5 fine-tune seeds × 3 pretraining seeds).
  Each has a CSV row in `results_p1fix_v2.csv` plus `_model-best.pt` and `_model-99.pt`. The completeness
  gate logged no "incomplete/missing" entry. Held-out TEST evaluations: 210 of 210 (7 languages × 30 runs;
  KArSL-v2a has no separate test signer).
* All six v2 backbones (pool and xling × pseeds 0–2) completed (job J41, `incomplete: 0`). Each of the four
  jobs (J41, J42, J43, J44) was requeued once at its wall-time limit and resumed from its
  checkpoints. The fine-tune jobs ended with `incomplete: 0` (ps0 07:09, ps1 08:24, ps2 08:19 EDT,
  2026-10-23). No fine-tune needed a manual resubmission.
* **1 exclusion** (PREREG §4 rule 3, gross failure): KArSL, pseed 0, pool, fine-tune seed 1. Its best VAL is
  5.10%, below 0.5 × the KArSL median of 11.29%. It is not collapsed (3 × chance = 4.29%). The cell
  (KArSL, 0) therefore keeps 5 xling and 4 pool runs. Every one of the 24 cells meets the ≥2-runs-per-arm
  rule, and every pretraining seed has all 8 languages (`pseeds_complete_all_langs` = [0, 1, 2]). The
  analysis therefore used **239 runs in 24 cells**. Sensitivity S1 shows the same analysis without this
  exclusion.

## Primary model (PREREG §5)

G_ℓp ~ N(a + b·d_ℓ + u_p + w_ℓp, s²_ℓp). The slope prior SD is 50/Δd = 154.78. The σ's have Half-Normal(5)
priors, evaluated on a 61×61 grid.

| quantity | posterior median | 95% CrI |
|---|---|---|
| slope b (pp per unit d^v2) | **−4.59** | **[−11.56, +2.61]** |
| posterior mean of b | −4.56 | — |
| P(b < 0) | **0.905** | — |
| intercept a (pp) | −0.27 | [−3.87, +3.33] |
| σ_u (shared pretraining-seed effect) | 1.67 | [0.00, 6.67] |
| σ_w (language × pretraining seed) | 1.00 | [0.00, 2.33] |
| fine-tune variance per cell, s̄² (plug-in) | 3.89 pp² | — |
| ICC_pseed = (σ_u² + σ_w²)/(σ_u² + σ_w² + s̄²) | 0.54 | [0.10, 0.93] |
| ICC_shared = σ_u²/(…) | 0.36 | [0.00, 0.90] |
| d* = −a/b, given b < 0 | −0.03 | [−3.16, +0.81] |

Method of moments (`mom_icc`): G_ℓp varies across pretraining seeds within a language with variance 5.23 pp²
(mean over languages). The mean fine-tune variance s² is 3.89 pp², so the pretraining-seed share is
(5.23 − 3.89)/5.23 = **0.26**. With 3 pretraining seeds the variance components are weakly identified: the
CrIs above are wide, and the Bayesian median (0.54) and the method-of-moments value (0.26) differ by a factor
of 2. (For comparison, the v1 7-pseed refit gave ICC_pseed 0.51 [0.24, 0.73], σ_w 2.67 [1.67, 3.67] and
s̄² 6.62; see `docs/p1_refit_2026-10/RESULTS.md`.)

### Posterior predictive P(G < 0 | d) (PREREG §7.2)

"Single run" means one new pretraining run with 5 fine-tune seeds; it adds u*, w* and ε* at the median s².
"Line" means a + b·d. Without v_ℓ the pretraining-seed-averaged value equals the line value.

| d | at | E[a + b·d] | P(G<0), single new run | P(line < 0) = P(pseed-avg G < 0) |
|---|---|---|---|---|
| 0.0810 | BdSL (= d_min) | −0.64 | 0.58 | 0.71 |
| 0.0913 | GSL | −0.68 | 0.58 | 0.72 |
| 0.1373 | LIBRAS | −0.90 | 0.61 | 0.79 |
| 0.1447 | LSFB | −0.93 | 0.62 | 0.80 |
| 0.1557 | AUTSL | −0.98 | 0.62 | 0.81 |
| 0.1572 | LSA64 | −0.99 | 0.62 | 0.81 |
| 0.2425 | mid-range | −1.38 | 0.67 | 0.87 |
| 0.3959 | BISINDO | −2.08 | 0.74 | 0.92 |
| 0.4041 | KArSL (= d_max) | −2.12 | 0.74 | 0.92 |

The posterior-mean line is negative over the whole observed range, but the intercept's CrI spans 0. For a
single new pretraining run, the sign of G is close to a coin flip at the near end of the range (0.58) and
leans negative at the far end (0.74).

## Leave-one-language-out (C2)

| held-out | d^v2 | predicted a + b·d (fit on the other 7) | observed pooled G | sign correct |
|---|---|---|---|---|
| BdSL | 0.081 | −1.39 | +1.51 | **no** |
| GSL | 0.091 | −0.34 | −3.18 | yes |
| LIBRAS | 0.137 | −1.26 | +1.18 | **no** |
| LSFB | 0.145 | −0.73 | −1.72 | yes |
| AUTSL | 0.156 | −0.79 | −2.40 | yes |
| LSA64 | 0.157 | −0.81 | −1.90 | yes |
| BISINDO | 0.396 | −1.99 | −2.13 | yes |
| KArSL (v2a) | 0.404 | −2.42 | −1.54 | yes |

**6/8** (threshold ⌈0.75·8⌉ = 6), so **C2 passes**. *Derived observation (not pre-registered):* all eight LOLO
predictions are negative, so C2 here equals the score of the constant rule "G < 0". The two misses are the
two languages with positive pooled G. C2's pass therefore reflects the predominance of negative gains, not
distance ordering.

## Per-language G (VAL, pp)

Bootstrap: pretraining seeds resampled, then fine-tune seeds within cells; 20,000 draws (`boot_lang`).
P(G>0) = 1 − P(Ḡ<0) from the bootstrap. The pool/xling columns give the mean over the 3 cells of the arm
means. *SD_ps* is the SD of G_ℓp over the 3 pseeds (derived). With only 3 pretraining seeds the bootstrap is
anti-conservative, so the table also gives a 95% *t*-interval (2 df) over the three cell gains (derived;
`make_figs.py` `v2_numbers`, asserted).

| target | d^v2 | ps0 | ps1 | ps2 | pooled G | 95% boot CI | P(G>0) | 95% t-CI (pseeds) | SD_ps | pool / xling |
|---|---|---|---|---|---|---|---|---|---|---|
| BdSL | 0.081 | +2.72 | +1.13 | +0.67 | **+1.51** | [−0.58, +3.56] | 0.92 | [−1.16, +4.18] | 1.07 | 54.98 / 56.49 |
| GSL | 0.091 | −3.59 | −5.52 | −0.44 | **−3.18** | [−6.83, +0.59] | 0.05 | [−9.55, +3.18] | 2.56 | 71.24 / 68.06 |
| LIBRAS | 0.137 | +3.43 | −0.11 | +0.23 | **+1.18** | [−1.41, +4.00] | 0.80 | [−3.67, +6.03] | 1.95 | 41.26 / 42.44 |
| LSFB | 0.145 | −0.31 | −4.15 | −0.70 | **−1.72** | [−4.28, +0.60] | 0.09 | [−6.98, +3.53] | 2.12 | 38.28 / 36.56 |
| AUTSL | 0.156 | −3.31 | +0.08 | −3.96 | **−2.40** | [−5.15, +0.81] | 0.07 | [−7.78, +2.98] | 2.17 | 68.66 / 66.26 |
| LSA64 | 0.157 | +1.31 | −1.75 | −5.25 | **−1.90** | [−5.36, +1.52] | 0.15 | [−10.06, +6.26] | 3.28 | 57.94 / 56.04 |
| BISINDO | 0.396 | −0.75 | −3.94 | −1.69 | **−2.13** | [−4.15, −0.21] | 0.01 | [−6.19, +1.94] | 1.64 | 16.58 / 14.46 |
| KArSL (v2a) | 0.404 | +0.64 | −4.66 | −0.61 | **−1.54** | [−4.81, +1.66] | 0.18 | [−8.43, +5.34] | 2.77 | 12.90 / 11.36 |

* Pooled G is negative for **6 of 8** languages.
  * **Only BISINDO's bootstrap interval excludes 0.**
  * **No t-interval over pretraining seeds excludes 0.**
  * After Holm correction over the 8 languages, nothing is significant under either interval. The smallest
    adjusted p is 0.23 for the bootstrap (two-sided p = 2·min(P(Ḡ<0), P(Ḡ>0))) and 1.00 for the t-test.

  BISINDO is a *low-accuracy target*, meaning 10%-label VAL Top-1 below 20%, as are KArSL-v2a and no other
  language: pool 16.6%, 32 classes, chance 3.1%. BISINDO's held-out-test G (S8) is +3.56, the opposite sign.
  GSL, AUTSL and LSFB have P(G>0) ≤ 0.09 (bootstrap, uncorrected).
* Nearest-first ordering: BdSL (nearest, d = 0.081) has the largest positive G, but GSL (second nearest,
  0.091) has the most negative one. LIBRAS (0.137) is positive, and the two far languages are moderately
  negative. *Derived:* Spearman(d, pooled G) = −0.19 over 8 languages. OLS on pooled G gives slope −3.42 with
  R² 0.068 (equal to the S7 GLS estimate, as expected for a balanced design).
* KArSL-v2a (pool 12.9%, 70 classes, chance 1.4%) and BISINDO are the two far points, and both are
  low-accuracy targets. The far end of the x-axis therefore rests on low accuracies; S6 is the pre-registered
  guard against this.

### Per-pretraining-seed 8-language OLS (derived; `make_figs.py`)

| pseed | slope | intercept | R² | negative cells |
|---|---|---|---|---|
| 0 | −0.49 | +0.11 | 0.001 | 4/8 |
| 1 | −8.80 | −0.64 | 0.203 | 6/8 |
| 2 | −0.95 | −1.28 | 0.003 | 6/8 |

No single v2 pretraining run reproduces the v1 seed-0 picture (slope −24.6, R² 0.89 on v1 distances).

## Sensitivity analyses (PREREG §7.3; none decides)

| analysis | P(b<0) | b median [95% CrI] | a median [95% CrI] | C1 | LOLO (C2) | C1∧C2 |
|---|---|---|---|---|---|---|
| **primary** | 0.905 | −4.59 [−11.56, +2.61] | −0.27 [−3.87, +3.33] | no | 6/8 | NOT SUPPORTED |
| S1 no exclusions except incomplete | 0.872 | −4.07 [−11.16, +3.31] | −0.34 [−4.07, +3.42] | no | 6/8 | NOT SUPPORTED |
| S2 + language effect v_ℓ | 0.805 | −4.34 [−15.65, +7.62] | −0.34 [−4.63, +3.92] | no | 6/8 | NOT SUPPORTED |
| S3 x = v1 (`old`) descriptor on the v2 pool¹ | 0.903 | −2.26 [−5.75, +1.25] | −0.45 [−3.97, +3.11] | no | 6/8 | NOT SUPPORTED |
| S4 x = `v2_interp` | 0.835 | −3.37 [−10.35, +3.85] | −0.51 [−4.17, +3.11] | no | 6/8 | NOT SUPPORTED |
| S5 x = v2 descriptor, v1-style source | 0.918 | −4.42 [−10.81, +2.03] | −0.41 [−3.95, +3.13] | no | 6/8 | NOT SUPPORTED |
| S6 drop KArSL-v2a and BISINDO (6 langs) | 0.912 | −27.30 [−65.36, +14.05] | +2.61 [−3.56, +8.39] | no | 3/6 (need 5) | NOT SUPPORTED |
| S7 MixedLM (REML), Wald 95% CI² | — | −3.42 [−11.42, +4.59], p = 0.40 | −0.60 | no | — | — |
| S8 held-out TEST-signer G (7 langs) | **0.014** | **+12.93 [+1.61, +24.47]** | −2.12 [−6.55, +2.36] | no | 3/7 (need 6) | NOT SUPPORTED |
| S9 fine-tune seeds 0–2 only | **0.992** | **−10.15 [−18.43, −1.97]** | +0.66 [−3.76, +5.07] | **yes** | 6/8 | **SUPPORTED** |

¹ S3's x is on a different scale (slope prior SD 69.0), so its b is not comparable in magnitude.
² S7 is a cell-level MixedLM with a pseed variance component (0.72 pp²) and residual scale 5.83 pp².
The language × pseed component is not separately identifiable from the residual with one observation per
cell (see Deviations).

**Note on the primary model.** The binding v2 model omits the language effect v_ℓ. The v1 refit's PREREG §4
included v_ℓ because "without v_l the 49 cells would pseudo-replicate 7 x-values". Omitting it makes C1
anti-conservative: with v_ℓ (S2) the CrI widens to [−15.65, +7.62]. S9 was also fitted without v_ℓ.

**The result is not robust, in either direction.**

* **S9 passes both criteria.** Using only fine-tune seeds 0–2, as the v1 law did for four of its languages,
  gives b = −10.15 [−18.43, −1.97] and LOLO 6/8. Its LOLO predictions are again all negative. The change is
  driven mostly by the far end:
  * KArSL's pooled G is −4.48 with seeds 0–2 and −1.54 with all five.
  * BdSL's is +2.19 versus +1.51.

  Had the pre-registration fixed three fine-tune seeds, the binding verdict would have been "SUPPORTED". This
  is the same forking path that the v1 audit found for GSL (seeds 3 → 5), here in the opposite direction. S9,
  like the primary fit, omits v_ℓ, so its interval is anti-conservative and should not be over-read.
* **S8 has the opposite sign.** On the held-out TEST signers the slope is positive, +12.93 [+1.61, +24.47],
  with P(b<0) = 0.014 and LOLO 3/7. S8 has no KArSL (no separate test signer), so BISINDO is its only point
  beyond d = 0.16, and BISINDO's TEST G is +3.56 versus −2.13 on VAL (its test split is one signer). TEST G
  per language:

  | | BdSL | LSA64 | AUTSL | GSL | LSFB | LIBRAS | BISINDO |
  |---|---|---|---|---|---|---|---|
  | TEST G | +0.07 | −0.42 | −2.66 | −0.75 | −0.88 | +1.53 | +3.56 |

* **S6** is the pre-registered guard. Without the two low-accuracy far languages, the remaining six span only d = 0.081–0.157. The slope is
  then unidentified ([−65.4, +14.0]) and LOLO falls to 3/6.
* S1–S5 and S7 all agree with the primary analysis: a negative median slope whose interval includes 0, and
  P(b<0) of 0.80–0.92.

**Reading.** With corrected inputs there is still no reliable (detectable) distance-dependent effect. The binding test
fails. One pre-registered robustness check (S9) passes, and another (S8) gives a significant slope of the
opposite sign. What the data do show, weakly, is that alignment does not pay relative to pooling at this
label budget:

* the posterior-mean line is below 0 across the observed range;
* 6/8 pooled G are negative.

But the intercept's CrI spans 0. Only one language's bootstrap interval excludes 0 (BISINDO's, which reverses
on TEST), and no t-interval over pretraining seeds does. This is "no detectable distance-dependent effect;
alignment does not pay", not a significant negative effect of alignment. No language shows a reliable gain
from alignment either. BdSL leans positive in both studies (v1 +2.26, v2 +1.51), and the lean vanishes on test
signers (v1 +0.96, v2 +0.07).

## Old vs. new (PREREG §7.4: pre-specified, descriptive; computed by `make_figs.py`)

v1 values come from the 7-pseed refit (`docs/p1_refit_2026-10/pooled_gains.json`, VAL @10%, v1 backbones with
the scrambled WLASL cache and `pose_motion` targets). Their fine-tune seeds are 0–2 for BdSL, LSA64, KArSL and
AUTSL, and 0–4 for the others. v1 KArSL used the signer-dependent validation set. "v1 ps0" is the published
single run (unrounded).

| target | d^v1 | d^v2 | v1 ps0 (published run) | v1 ps0–2 | v1 ps0–6 | **v2 ps0–2** | v2 − v1(ps0–2) |
|---|---|---|---|---|---|---|---|
| BdSL | 0.143 | 0.081 | +1.68 | +0.92 | +2.26 | **+1.51** | +0.59 |
| LSA64 | 0.176 | 0.157 | +0.94 | +0.77 | +0.82 | **−1.90** | −2.66 |
| KArSL³ | 0.344 | 0.404 | −4.33 | +1.31 | +1.53 | **−1.54** | −2.86 |
| AUTSL | 0.411 | 0.156 | −6.43 | −2.70 | +0.01 | **−2.40** | +0.30 |
| GSL | 0.121 | 0.091 | +2.06 | −1.32 | −0.72 | **−3.18** | −1.87 |
| LSFB | 0.480 | 0.145 | −5.06 | +0.72 | −0.48 | **−1.72** | −2.44 |
| LIBRAS | 0.315 | 0.137 | −4.69 | −2.25 | −2.14 | **+1.18** | +3.43 |
| BISINDO⁴ | 0.450 | 0.396 | +2.63 | — | — | **−2.13** | — |
| mean of the 7 v1 languages | | | | −0.36 | +0.18 | **−1.15** | |

³ v1 = signer-dependent VAL (random 12% of U01+U02); v2 = held-out signer U03 (split v2a). These are
different quantities.
⁴ v1 BISINDO exists only at pseed 0 (5 fine-tune seeds; `results/provenance/bisindo_v1_lawpoint_2026-10-22.txt`).

* Relative to the matched v1 pseeds 0–2, v2 moves the mean gain over the seven common languages from −0.36 to
  −1.15 pp. Per-language changes range from −2.86 to +3.43 pp. These are the same order as the across-pseed SD
  of G (1.1–3.3 pp in v2, 1.7–4.2 in v1), and pseed p in v1 and in v2 are unrelated draws.
* **Not identifiable.** v2 changes, all at once:
  1. the WLASL skeleton in the pool (scramble fixed);
  2. the k-means target features (`pose_motion` → `pose_motion_masked`, set by the pre-registered dropout
     rule);
  3. the KArSL protocol;
  4. the language set (BISINDO added).

  Any v1 → v2 change in G, per language or on average, cannot be attributed to an individual fix.
* The distance decomposition (v1 → each fix → v2) is in `step2/distances_v2_pooledbw.md` and in the paper's
  App. "Distance decomposition".

## Deviations from pre-registration and other disclosures

1. **Distance-tool source-weighting bug (fixed before Addendum A).** The first full distance run (job J37)
   drew the v1-replica (`oldsrc_*`) sources at the loaded-pool shares (66/34) instead of the v1 tool's 50/50.
   This affected only the sensitivity and decomposition sources, not the primary `pool_v2` source. It was
   fixed and everything was recomputed once (job J38) before Addendum A was written. The first-run outputs
   are kept as `step2/run1_srcweight_bug_*` and are not used (LOG 05:37, 05:48).
2. **Confounded fixes.** The v2 backbones differ from v1 in both the WLASL skeleton and the k-means target
   features. The target switch was decided by the pre-set dropout rule (26/64 clusters and 10.9% of frames
   dropout-dominated; `step5/kmeans_dropout_diag.json`) before any training, and is stated in PREREG §1. Their
   separate effects are not identifiable from this rerun (LOG 05:07).
3. **PREREG header time.** The file's header says "written ~05:30 EDT". Its mtime and hash record say 05:17:19.
   Both precede every v2 artefact.
4. **S7 implementation.** PREREG §7.3 asks for variance components for pseed *and* language × pseed. With one
   cell mean per (ℓ, p), the language × pseed component is not identifiable separately from the residual, so
   `refit_v2.py` fits a pseed variance component. The residual absorbs w + ε, and the known s² is not used.
   S7 is secondary; the primary model is as specified.
5. **Old-vs-new decomposition of G (§7.4).** This is pre-specified (descriptive). It is not computed by `refit_v2.py`; it was computed afterwards
   by `tools/paper/make_figs.py` from the v1 refit's `pooled_gains.json`, which holds the same
   `results_final.csv` runs the PREREG names. BISINDO has no v1 pseed 1–2 runs.
6. **Not pre-registered (descriptive only):**
   * per-pseed OLS slopes;
   * pooled-G OLS and Spearman;
   * the "all LOLO predictions negative" observation;
   * SD_ps;
   * the t-intervals over pretraining seeds and the Holm correction;
   * the paper figure (`fig_v2law.pdf`);
   * the interpretation above.
7. **Operational.** Pretraining was packed into one 2-GPU job (3 runs per GPU), and fine-tunes into one 2-GPU
   job per pretraining seed, because of the shared 100-job submit cap. The v2 pretraining configs checkpoint
   every epoch (stated in PREREG §1). Each job was requeued once at the wall-time limit and resumed from
   checkpoints. No hyper-parameter, seed, epoch count or data change was made.
8. **Interrupted evaluations.** "Best epoch" means the best of the 20 VAL evaluations, one every 5 epochs. The
   fine-tune requeue at 2026-10-23 06:32 interrupted an evaluation in 10 of the 240 runs (`results_p1fix_v2.csv`
   has 230 runs with 20 rows, 6 with 21 and 4 with 19).
   * The 6 runs with 21 rows re-evaluated one epoch after resuming, and the repeat did not change their best
     epoch.
   * The 4 runs with 19 rows each lost one of their 20 evaluations:
     * `lsfb_si_f10_v2_xling_ps2_seed4`, epoch 59;
     * `karsl_si_f10_v2_xling_ps2_seed0`, epoch 49;
     * `karsl_si_f10_v2_xling_ps2_seed1`, epoch 39;
     * `bdsl_block_gcn_si_f10_v2_xling_ps0_seed4`, the final epoch 99.

     Their best epoch is the best of the 19 remaining evaluations.
9. **Pre-registration timing.** The time stamps are file-system times and hashes recorded by the authors; there
   is no public registry. `PREREG.md`, Addendum A and `PREREG.sha256` were committed to version control in
   commit 2e745eb (2026-10-22 15:53 EDT). That is before any v2 fine-tune result existed: the first row of
   `results_p1fix_v2.csv` is from 22:53 that day, and the first fine-tune finished at 00:22 on 2026-10-23.

No other deviations. The analysis used the same languages (8), pretraining seeds (0–2), fine-tune seeds (0–4),
frozen distances (Addendum A), exclusion rules, model, priors and decision rule as pre-registered. The verdict
was computed once, after all v2 SLURM jobs had ended.
