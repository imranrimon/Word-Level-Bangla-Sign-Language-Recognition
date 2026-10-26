# P1 distance-gating law: pre-registered 7-pretraining-seed refit

**Written 2026-10-22 04:50 EDT.** At the time of writing, NO result for pretraining seeds 3–6 had been
read. No `results_final.csv` row with a `_ps3` / `_ps4` / `_ps5` / `_ps6` experiment name was opened,
printed or parsed. Status at writing: pretrains J23 (pool ps3), J25 (pool ps5), J27 (xling ps5),
J29 (pool ps6) and J31 (xling ps6) were running. Fine-tunes J20 (xling ps3), J21 (pool ps4)
and J22 (xling ps4) were running. Fine-tunes J24, J26, J28, J30 and J32 were pending
on their dependencies.

**Disclosure of what was already seen.** Pretraining seeds 0, 1 and 2 have been analysed and are public
in-repo:

* pseed 0 is the original P1 single run, which **generated** the law hypothesis.
* pseeds 1 and 2 are analysed in `docs/improvement_runs_2026-10/item6_p1_law_pretrain_seeds.md`.
  Pooled over 0–2: slope(7) = −1.6, R²(7) = 0.02, paper-protocol sign 1/7.

This pre-registration therefore covers four genuinely unseen pretraining seeds (3–6). It is analysed
jointly with three seen ones. A replication-only sensitivity analysis (below) uses pseeds 1–6 and drops
the hypothesis-generating seed.

## 1. Quantity

* G = xling − pool. Each arm is measured as best-epoch **VAL** Top-1 (%) at **10% labels**.
* Recipe: `config/<lang>_si_lr01_f10.yaml`, with subset_seed 42, lr 0.01, 100 epochs, fc re-initialised
  and no gradient clipping. This is the identical recipe for every pseed. pseed 0 used the original runs.
* Backbones:
  * pool = `bdsl_shubert_pretrain_bdsl_asl[_pseed<p>]` (BdSL+ASL pool, no alignment losses)
  * xling = `bdsl_shubert_pretrain_xlingual[_pseed<p>]` (same pool, GRL-adv 0.1 + contrastive 0.1)
  * Both use epoch 30.
* Experiment names: `<stem>_lr01_f10_<arm>[_ps<p>]_seed<f>`. pseed 0 has no `_ps` suffix.
* Stems:
  * BdSL `bdsl_block_gcn_si`
  * LSA64 `lsa64_si`
  * KArSL `karsl_si`
  * AUTSL `autsl_si`
  * GSL `gsl_si`
  * LSFB `lsfb_si`
  * LIBRAS `libras_si`
* Parsing: identical to `tools/p1_law_pretrain_seed_analysis.py::load`. VAL is the max over rows of an
  experiment name. TEST is the last `_test` row.

## 2. Included data

* **Languages (7, fixed):** BdSL, LSA64, KArSL, AUTSL, GSL, LSFB, LIBRAS.
  * BISINDO is excluded, as in P1: it is below the learnability floor at 34% even with 100% labels.
  * No language is added or dropped after seeing data.
* **Pretraining seeds:** 0–6, for both arms.
* **Fine-tune seeds:** f ∈ {0,1,2} for all languages, plus f ∈ {3,4} for GSL, LSFB and LIBRAS. This
  matches the P1 protocol and the packed fine-tune script.
* **x-axis d_l is FIXED** at the canonical one-pass pool→target pose-motion RBF-MMD values in
  `tools/gating_law_fit.py`:
  * GSL 0.1206
  * BdSL 0.1429
  * LSA64 0.1763
  * LIBRAS 0.3151
  * KArSL 0.3445
  * AUTSL 0.4115
  * LSFB 0.4798

  These values are label-free and backbone-free, and they are not recomputed or tuned.

## 3. Exclusion rules (fixed now)

**A run is excluded from the primary analysis if any of the following holds:**

1. **Incomplete:** there is no `_model-best.pt`, or the run did not reach epoch 99 (no `_model-99.pt`).
   A run still in progress is treated as missing.
2. **Collapsed:** best VAL Top-1 ≤ 3 × chance (chance = 100 / num_class):
   * BdSL 5.0
   * LSA64 4.69
   * KArSL 4.29
   * AUTSL 1.33
   * GSL 0.97
   * LSFB 5.0
   * LIBRAS 15.0
3. **Gross failure:** best VAL Top-1 < 0.5 × the median best VAL Top-1 of all runs of the same language
   (both arms, all pseeds, all fine-tune seeds). This rule is symmetric across arms.

**Rules for cells, pseeds and the analysis as a whole:**

* **Minimum cell size:** a (language, pseed) cell enters only if each arm keeps at least 2 runs.
  Otherwise the cell is dropped; the model below handles unbalanced or missing cells.
* **Failed pretraining seed:** if a pretraining seed has no usable backbone for an arm (pretrain failed
  permanently), that pseed is dropped for the affected arm. Cells need both arms, so it is in effect
  dropped entirely.
* **Minimum to proceed:** the analysis proceeds with whatever exists once the SLURM jobs have ended,
  completed or failed permanently. The verdict is computed only if at least 5 pretraining seeds have
  all 7 languages complete. With fewer, the result is reported as "INCONCLUSIVE (insufficient data)",
  together with the same statistics.

**Sensitivity analyses:**

* (S1) No exclusions except incomplete runs.
* (S2) Fine-tune seeds 0–2 only, for every language.

## 4. Model and estimator (primary)

Cell gain and its known fine-tune sampling variance:

```
G_{l,p}  = mean_f(xling_{l,p,f}) − mean_f(pool_{l,p,f})          (arms unpaired)
s²_{l,p} = S²_{x,l}/n_{x,l,p} + S²_{p,l}/n_{p,l,p}
```

Here S²_{arm,l} is the within-cell fine-tune variance for (language, arm), pooled over pseeds. It is
the fine-tune-seed term ε, plugged in as known.

Hierarchical model:

```
G_{l,p} ~ Normal(a + b·d_l + u_p + v_l + w_{l,p},  s²_{l,p})
u_p     ~ Normal(0, σ_u²)   pretraining-seed effect SHARED across languages (shared backbone pair)
v_l     ~ Normal(0, σ_v²)   language-level lack of fit (d is constant within language; without v_l the
                            49 cells would pseudo-replicate 7 x-values)
w_{l,p} ~ Normal(0, σ_w²)   language × pretraining-seed interaction
```

This is the requested G_{l,p,f} = a + b·d_l + u_p + w_{l,p} + ε, plus the language term v_l. v_l is
included because b is identified only from between-language variation. With σ_v = 0 it reduces exactly
to the requested model, which is also reported as sensitivity (S3).

**Estimator: Bayesian.**

* Priors:
  * a ~ N(0, 20²) pp
  * b ~ N(0, 50²) pp per unit MMD (the P1 seed-0 slope was −24.8)
  * σ_u, σ_v, σ_w ~ Half-Normal(scale = 5 pp), independent
* Computation:
  * (a, b, u, v, w) are marginalised analytically. The model is Gaussian given the σ's, with
    Cov(G) = σ_u² Z_u Z_uᵀ + σ_v² Z_v Z_vᵀ + (σ_w² I + diag s²) + X Σ_β Xᵀ.
  * The posterior over (σ_u, σ_v, σ_w) is evaluated by deterministic grid quadrature: 61 points per
    σ, uniform on [0, 20] pp.
  * The posterior of (a, b) given the σ's is the conjugate Gaussian, mixed over the grid weights.
  * There is no MCMC, so there are no convergence diagnostics to tune.
* Implementation: `tools/p1_refit/refit.py`.
* Prior sensitivity (S4), reported but not decision-relevant:
  * σ priors Half-Normal(10)
  * σ priors Half-Cauchy(2.5)
  * b ~ N(0, 100²)

**Frequentist cross-check (secondary, not decision-relevant).** statsmodels MixedLM, REML, on the cell
gains G_{l,p}. It has fixed effects a + b·d_l and crossed random intercepts for pseed and language,
fitted as variance components in one all-encompassing group. The residual is w + ε. Report b with its
Wald 95% CI.

## 5. Decision rule (pre-registered, binding)

**KEEP-LAW** only if BOTH of the following hold. Otherwise the verdict is **REFRAME**.

* **(D1)** Under the primary model fitted to all included pseeds 0–6, posterior **P(b < 0) > 0.95**.
  The 95% central credible interval of b is also reported, and "excludes 0" is noted. D1 is the
  one-sided criterion, because the law's claim is directional.
* **(D2)** Leave-one-language-out sign accuracy is **≥ 6/7**:
  * For each language l, refit the primary model on the other 6 languages (all their pseeds).
  * Predict ĝ_l = posterior mean of a + b·d_l.
  * The prediction is correct if sign(ĝ_l) = sign(Ḡ_l), where Ḡ_l = mean over included pseeds of
    G_{l,p}. This is the pseed-pooled observed gain.

**Rules that apply to every outcome:**

* No other statistic can rescue the law. In particular this includes R² on pooled means, any single
  pseed, a subset of languages, the calibration-5/prospective-2 split, and the frozen d* = 0.193 gate.
* The replication-only analysis (pseeds 1–6), S1–S4 and the MixedLM fit are reported, but they do not
  change the verdict. If they disagree with the primary verdict, that disagreement is reported
  prominently.

## 6. Secondary quantities (pre-specified, descriptive)

1. **Pretraining-seed ICC.** Posterior median and 95% CrI of:
   * ICC_shared = σ_u² / (σ_u² + σ_v² + σ_w² + s̄²)
   * ICC_pseed = (σ_u² + σ_w²) / (σ_u² + σ_v² + σ_w² + s̄²)

   s̄² is the mean of s²_{l,p} over cells. ICC_pseed is the share of single-run variance in G that comes
   from the pretraining run. Also reported: the posterior of σ_u, σ_v and σ_w, and the method-of-moments
   variance components that `p1_law_pretrain_seed_analysis.py` already prints.
2. **Posterior predictive P(G < 0 | d) at d ∈ {0.15, 0.30, 0.45}** for a new language. Three versions:
   * (i) Single new pretraining run with the standard 3 fine-tune seeds:
     G_new = a + b·d + u* + v* + w* + ε*, with Var(ε*) = median s²_{l,p}.
   * (ii) Pretraining-seed-averaged expected gain: a + b·d + v*.
   * (iii) The regression line alone: a + b·d.
3. **Pooled BdSL xling − mono gain** (P1 `tab:sig` / `tab:test` headline), VAL and TEST, computed from:
   * xling over pseeds 0–6;
   * mono over pseeds 0–2, from item 5. Mono pretrains for pseeds 3–6 do not exist and are not run
     for this analysis.

   CI method: the two-level hierarchical bootstrap of item 5. It resamples pseeds within an arm, then
   fine-tune seeds within each drawn pseed, with arms unpaired and 20,000 draws. The method is
   implemented in `p1_law_pretrain_seed_analysis.boot_lang`. Also reported:
   * the matched pseeds 0–2 version, which must reproduce item 5's +0.78 [−3.27, +4.91] VAL;
   * pooled BdSL pool − mono (G_data) and mono − scratch.
4. **Per-language pooled G** with hierarchical 95% CI and P(G > 0). Also the per-pseed OLS slopes, the
   pooled-mean OLS (for comparability with P1) and the paper-protocol sign count. All of these are
   descriptive only.
5. **Refits of P1 `tab:transfer` and `tab:robust` against pooled G** (pseeds 0–6), using
   `tools/p1_law_baseline_refit.py` with the cached estimator distances
   (`docs/improvement_runs_2026-10/raw/item6_estimator_distances.json`). Descriptive only.
6. **Held-out TEST G** for BdSL, LSA64 and AUTSL (P1 `tab:test`), pooled over pseeds 0–6, with the same
   bootstrap.

## 7. Commitments

* Code is written and debugged on pseeds 0–2 only, which is already-seen data. It is then run once on
  pseeds 0–6.
* Any deviation from this document is listed in `RESULTS.md` under "Deviations from pre-registration",
  with reasons.

---
Footer: the sha256 of this entire file (it cannot contain its own hash) and its mtime are recorded in
`PREREG.sha256` at write time; the file is then made read-only (chmod a-w).
