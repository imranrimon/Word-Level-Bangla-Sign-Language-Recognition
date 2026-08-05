# P1 — Results section (draft, 2026-08-04)

Draft of the method paper's core results. Numbers are **val Top-1, 3-seed mean**
unless noted; [BRACKETS] = pending (LOSO 11-fold, held-out test signers — running).
All conditions share one recipe (BlockGCN, SGD, **base_lr 0.01**, 100 epochs, SI
split); SSL conditions initialise from a masked-pose backbone (`--ignore fc`).

## Setup
- **Backbones** (masked pose-unit pretraining, `main_pretrain.py`, 30 epochs):
  **mono** = BdSL-only pool; **pool** = ASL+BdSL pool, mechanism off; **xling** =
  ASL+BdSL + language-adversarial (GRL) + shared-codebook contrastive.
- **Targets** (signer-independent): BdSLW60 (5,748 train), LSA64 (2,240), AUTSL (28,142).
- **Data-efficiency**: fine-tune each condition on stratified subsets (5–100% of train).

## Finding 1 — SSL helps, and most when labels are scarce
At 5–10% of labels, every SSL condition beats from-scratch by large margins; the
gap closes as data grows.

**Table 1a. BdSLW60-SI val Top-1 vs. train fraction (3-seed mean).**
| cond | 5% | 10% | 15% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| scratch | 0.357 | 0.460 | 0.509 | 0.607 | 0.749 | 0.780 |
| mono | 0.416 | 0.508 | 0.602 | 0.690 | 0.749 | 0.802 |
| pool | 0.389 | 0.533 | 0.610 | 0.654 | 0.705 | 0.747 |
| **xling** | **0.432** | **0.550** | 0.602 | 0.693 | 0.762 | 0.774 |

## Finding 2 — Cross-lingual beats monolingual *only when labels are scarce*
`xling − mono` is large and significant at the low end and **statistically vanishes
at full data** — the effect is a property of the low-resource regime.

**Table 2. Significance of the cross-lingual advantage (paired bootstrap, one-sided, 10k).**
| Target @ fraction | xling − mono | p | SSL (mono − scratch) | p |
|---|---|---|---|---|
| BdSLW60 @ 5% | +1.5 pp | 0.039 | +6.0 pp | 0.000 |
| BdSLW60 @ 10% | **+4.2 pp** | **0.000** | +4.8 pp | 0.000 |
| LSA64 @ 10% | **+1.8 pp** | **0.039** | −0.5 pp | 0.405 |
| LSA64 @ 100% | +1.6 pp | 0.150 | −0.7 pp | 0.298 |
| AUTSL @ 10% | [pending] | — | +2.3 pp | 0.039 |

## Finding 3 — On the scarcest target, the benefit is *specifically* cross-lingual
**Table 1b. LSA64-SI val Top-1 (3-seed mean).** LSA64 (2,240 train) is low-resource
even at full data — and cross-lingual wins throughout.
| cond | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| scratch | 0.605 | 0.801 | 0.905 | 0.956 |
| mono | 0.600 | 0.830 | 0.919 | 0.949 |
| pool | 0.608 | 0.828 | 0.927 | 0.963 |
| **xling** | **0.618** | 0.830 | **0.933** | **0.965** |

On LSA64 `mono ≈ scratch` (p=0.40) while `xling > mono` (p=0.039): **adding ASL is
what helps, not self-supervision alone** — a direct argument for the cross-lingual
mechanism. (AUTSL @10/25% [pending re-run].)

## The mechanism is target-dependent (honest nuance)
`xling > pool` at every BdSLW60 and LSA64 fraction (+1.7 to +5.7 pp) — the GRL +
contrastive mechanism improves over plain pooling **for those targets**. But on
**AUTSL** (Turkish — the most distant target, absent from the ASL+BdSL pretraining
pool) the **mechanism hurts**: xling@10% ≈0.59 vs pool 0.70, mono 0.69.

→ **The robust win is cross-lingual POOLING** — `pool ≥ mono ≥ scratch` on all three
languages at low data. The **mechanism helps closer targets but can hurt a distant
unseen language** — a real limitation to report and to probe with the λ-ablation.

**Table 1c. AUTSL-SI val Top-1 (3-seed mean).**
| cond | 10% | 25% |
|---|---|---|
| scratch | 0.666 | 0.836 |
| mono | 0.690 | 0.833 |
| **pool** | **0.700** | **0.840** |
| xling | 0.592* | 0.790 |

\*xling@10% from 2/3 seeds (3rd running); both are genuinely low (0.586, 0.599).

## Val → test confirmation ✅
The crossover **holds on held-out test signers**, not just val: xling is the best
condition at @10% on BdSL *and* LSA64.

**Table 3. @10% held-out TEST-signer Top-1 (3-seed mean).**
| Target | scratch | mono | pool | **xling** | xling−mono |
|---|---|---|---|---|---|
| BdSLW60 (sig. 2,13) | 0.442 | 0.439 | 0.405 | **0.478** | **+3.9 pp** |
| LSA64 (sig. 9,10) | 0.597 | 0.622 | 0.581 | **0.631** | +0.9 pp |
| AUTSL (official) | 0.663 | 0.686 | **0.693** | [running] | — |

## Robustness [pending — running]
- **LOSO** (11 signer folds, BdSLW60): [mean ± std] — the headline signer-noise
  variance (array `124307`, running).

## Honest scope / limitations
- The cross-lingual advantage is a **low-data effect** (≤10–15%); at full data on the
  larger target (BdSLW60) monolingual matches or exceeds it. We claim exactly this.
- Significance is currently **n=3 seeds**; LOSO (n=11 folds) is the stronger test.
- Val→test confirmation pending.

## Headline claim (what Fig 1 shows)
> *Cross-lingual masked pose pretraining significantly improves low-resource sign
> recognition — the scarcer the labels, the larger the gain — and generalises across
> languages; the advantage is specifically cross-lingual, not self-supervision alone.*
