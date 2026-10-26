# P1 distance-gating law, v2 (fixed inputs): pre-registration

**Written 2026-10-22 ~05:30 EDT, before any v2 result exists.** At writing: the v2 k-means targets
were not yet computed, no v2 backbone had been pretrained, no v2 fine-tune had been launched, and no
v2 distance (the `v2` descriptor on the real pool) had been computed on full samples (only a
throw-away smoke run on 40-60 clips to test the code, whose numbers are not used). The file
`results_p1fix_v2.csv` does not exist. sha256 + mtime are in `PREREG.sha256`; the file is then made
read-only. Any deviation is listed in `RESULTS.md` under "Deviations from pre-registration".

**Already seen (disclosed).** All v1 results (P1 paper; pretraining seeds 0–2 of the v1 law,
`docs/improvement_runs_2026-10/item6_p1_law_pretrain_seeds.md`), the v1 distances, the audit
findings, and in this program: the KArSL U03 re-scoring of v1 checkpoints (Step 3) and the geometry /
fps / k-means diagnostics (Steps 1, 2, 5a). None of these involve a v2 backbone or a v2 distance value.

## 1. What is rerun (fixed)

* **Pool (both arms):** `data/ssl_pool_manifest_bdsl_asl_v2.json` = the v1 pool (BdSLW401-front 51,098 +
  WLASL 21,083 + BdSLW102a 3,408) with WLASL replaced by the geometry-checked `data/wlasl_pose_cache_v2`.
* **k-means targets:** `data/pretrain_kmeans_targets_bdsl_asl_v2.npz`, K=64, fit-sample 0.1, seed 0,
  per-source cap 200,000, feature mode `pose_motion_masked` (decided by the pre-set rule in
  `tools/p1_fix/kmeans_dropout_diag.py` before any training).
* **Backbones:** pool = `config/p1fix/bdsl_shubert_pretrain_bdsl_asl_v2.yaml`, xling =
  `config/p1fix/bdsl_shubert_pretrain_xlingual_v2.yaml` (identical to v1 except manifest, targets and
  checkpoint frequency); pretraining seeds p ∈ {0, 1, 2}; epoch-30 backbone.
* **Fine-tune (identical P1 @10% law recipe):** `config/p1fix/<stem>_f10_v2.yaml`: 10% labels
  (subset_seed 42), lr 0.01, 100 epochs, SGD, no clipping, fc re-initialised; fine-tune seeds
  f ∈ {0,1,2,3,4} for **every** language; both arms; all 3 pretraining seeds. 8 × 2 × 5 × 3 = 240 runs.
  Names `<stem>_f10_v2_<arm>_ps<p>_seed<f>`; results in `results_p1fix_v2.csv`.
* **Quantity:** best-epoch **VAL** Top-1 (%) of each run = max Top1_Acc over its rows in
  `results_p1fix_v2.csv`. G_{l,p,f} := xling − pool. Cell gain G_{l,p} = mean_f xling − mean_f pool
  (arms unpaired).

## 2. Languages (fixed: all 8)

BdSL (`bdsl_block_gcn_si`), LSA64, AUTSL, GSL, LSFB, LIBRAS, **BISINDO** (no longer excluded — it was a
planned prospective target dropped post hoc in v1) and **KArSL on the signer-disjoint split
`data/karsl_si_v2a`** (train U01+U02, val = held-out signer U03). Step 3 showed the v1 KArSL val is
signer-dependent (U03 accuracy 6–12% vs 55–73% on the SD val), so the v1 KArSL protocol is not used.
No language is added or dropped after seeing v2 results.

## 3. Distance (x-axis), fixed definition

d_l := mean over **20 independent draws** of the unbiased squared MMD (U-statistic, RBF kernel) between
a proportional sample of **800 clips of the v2 pool manifest** and **400 clips of language l's SI train
bundle**, using the **`v2` descriptor** of `tools/compute_lang_distance_v2.py` (27 joints × {mean, std} of
per-second speed; coordinates divided by the clip's median shoulder width over frames with both
shoulders detected; each joint trajectory resampled to 25 Hz by linear interpolation inside detected
runs only; speeds only between grid samples inside one detected run; joints never tracked → 0) with the
**pooled** median-heuristic bandwidth (median squared distance over the union of the source sample and
all 9 target samples of that draw), native fps from `docs/p1_fix_2026-10/step2/fps_registry.json`,
`--seed 0`, descriptor pools 6,000 source / 2,400 target clips. Run as
`python tools/compute_lang_distance_v2.py --out docs/p1_fix_2026-10/step2/distances_v2.json --draws 20`.
The values (label-free, backbone-free) are recorded in an addendum to this file with its own timestamp
before any v2 fine-tune result exists, and are never re-tuned.

## 4. Exclusion rules

A run is excluded if: (1) **incomplete** (no `_model-99.pt` or no `_model-best.pt`, or no CSV row);
(2) **collapsed**: best VAL ≤ 3 × chance (chance = 100/num_class: BdSL 60, LSA64 64, AUTSL 226,
KArSL 70, GSL 310, LSFB 60, LIBRAS 20, BISINDO 32 classes); (3) **gross failure**: best VAL < 0.5 × the
median best VAL of all runs of that language (both arms, all pseeds/seeds). A (language, pseed) cell
enters only if each arm keeps ≥ 2 runs. A pretraining seed whose backbone fails permanently is dropped
for both arms. The analysis runs once all v2 SLURM jobs have ended; the verdict is computed only if ≥ 2
pretraining seeds have all 8 languages complete, else "INCONCLUSIVE (insufficient data)".
Failed fine-tunes are resubmitted with resume before the analysis (not an exclusion).

## 5. Model and estimator (primary)

The requested model G_{l,p,f} = a + b·d_l + u_p + w_{l,p} + ε, fitted on cell means with the
fine-tune term as known variance:

```
G_{l,p} ~ Normal(a + b·d_l + u_p + w_{l,p},  s²_{l,p})
s²_{l,p} = S²_{x,l}/n_{x,l,p} + S²_{p,l}/n_{p,l,p}   (S² = within-cell fine-tune variance, pooled over pseeds, per language×arm)
u_p ~ N(0, σ_u²)    pretraining-seed effect shared across languages (one backbone pair per seed)
w_{l,p} ~ N(0, σ_w²) language × pretraining-seed interaction
```

Bayesian estimator (code shared with `tools/p1_refit/refit.py`'s grid machinery): a ~ N(0, 20²),
b ~ N(0, (50/Δd)²) where Δd = max_l d_l − min_l d_l (scales the slope prior to the v2 distance
units: a ±50 pp change across the observed distance range is 1 SD), σ_u, σ_w ~ Half-Normal(5 pp);
(a, b, u, w) marginalised analytically; posterior over (σ_u, σ_w) on a 61×61 grid on [0, 20] pp.
"95% CI of b" = central 95% posterior credible interval. Implementation: `tools/p1_fix/refit_v2.py`.

## 6. Decision rule (binding)

**Law SUPPORTED** iff BOTH:
* **(C1)** the 95% CI of b lies entirely below 0 (upper limit < 0), and
* **(C2)** leave-one-language-out sign accuracy ≥ **6/8**: for each language l, refit the primary model
  on the other 7 languages (all pseeds), predict ĝ_l = posterior mean of a + b·d_l, and count correct if
  sign(ĝ_l) = sign(Ḡ_l), where Ḡ_l = mean over included pseeds of G_{l,p} (seed-pooled observed gain).
  (If a language has no included cell it is skipped and the threshold scales to ⌈0.75·n⌉.)

Otherwise **NOT SUPPORTED**. No other statistic can rescue the law (R² on pooled means, single pseeds,
subsets of languages, the frozen v1 d* gate, the calibration/prospective split, test-split gains).

## 7. Secondary (pre-specified, descriptive; do not change the verdict)

1. **Pretraining-seed ICC** = (σ_u² + σ_w²)/(σ_u² + σ_w² + s̄²) (share of single-run cell variance due to
   the pretraining run) and σ_u²/(…) (shared part); posterior median + 95% CrI; plus method-of-moments.
2. **P(G < 0 | d)** posterior predictive at the 8 observed d_l and at min/mid/max of the observed range:
   (i) a new single pretraining run with 5 fine-tune seeds (adds u*, w*, ε* with median s²);
   (ii) pretraining-seed-averaged (line + w*); (iii) the line alone. Plus a pseed-level bootstrap
   (resample pretraining seeds, then fine-tune seeds within cells) of P(Ḡ < 0) per language.
3. **Sensitivity fits** (each reports the C1/C2 outcome; none decides):
   S1 no exclusions except incomplete; S2 add a language random effect v_l (guards against pseudo-
   replication of the 8 x-values); S3 x = v1 `old` descriptor on the v2 pool; S4 x = `v2_interp`;
   S5 x = `v2` with the v1 tool's source (oldsrc_wlasl_v2); S6 drop the two near-floor languages
   (KArSL-v2a, BISINDO); S7 frequentist MixedLM (REML, fixed a + b·d, variance components for pseed and
   language×pseed), Wald CI of b; S8 held-out TEST-signer G where a separate test split exists
   (BdSL, LSA64, AUTSL, GSL, LSFB, LIBRAS, BISINDO); S9 fine-tune seeds 0–2 only.
4. **Old-vs-new decomposition** (descriptive): for each language, v1 G (pseeds 0–2, `results_final.csv`,
   v1 KArSL protocol) vs v2 G; v1 distance vs each step of the distance fix (WLASL fix only; + real pool
   source; + dropout mask; + per-second; + 25 Hz/shoulder = v2), all from `distances_v2.json`.

---
Footer: sha256 and mtime of this file in `PREREG.sha256`; then chmod a-w.
