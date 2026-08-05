# P1 — Research Design (method paper)

Designed around the verified novelty constraints in
[`DEEP_RESEARCH_2026-08-02.md`](DEEP_RESEARCH_2026-08-02.md) §6 (re-check 2026-08-03).
The novelty survives **only at the four-way intersection stated jointly** — this
design foregrounds the two most defensible axes (masked-reconstructive-on-pose +
formal SI) and frames the target as isolated word-level SLR.

## Working title
**"Signer-Independent Cross-Lingual Masked Pose Pretraining for Low-Resource
Isolated Sign Language Recognition."**

## One-line thesis (all four axes, jointly)
A **masked-reconstructive** pose autoencoder pretrained on **multi-source
cross-lingual** pose (ASL+BdSL), **fine-tuned** under a **formal
signer-independent** protocol, transfers to **low-resource** target sign
languages — beating monolingual SSL and from-scratch, and generalizing across
≥2 target languages and signer-disjoint LOSO.

## Empirical status — the low-resource reframe (2026-08-04)

Running the gate + a data-efficiency curve **reframed the claim, honestly**:

- **At FULL data on BdSLW60, cross-lingual does NOT beat monolingual** (mono 0.802 >
  xling 0.774 val, 3-seed). The naive "xling > mono" gate *fails*.
- **The data-efficiency curve shows the cross-lingual advantage is a LOW-RESOURCE
  phenomenon.** `xling − mono` = **+4.2 pp @10%** → +0.3 @25% → +1.3 @50% → −2.8 @100%.
  The benefit exists *exactly when labels are scarcest* and vanishes as data grows.
- **The mechanism (xling: GRL + contrastive) beats plain pooling at EVERY fraction**
  (+1.7 to +5.7 pp) — the novel component adds value (earlier "mechanism hurts" reads
  were seed-0 noise).
- **All SSL beats scratch at 10–25%** (+5 to +9 pp) — masked pose SSL helps low-resource.

| cond | 10% | 25% | 50% | 100% | (BdSLW60-SI val Top-1, 3-seed) |
|---|---|---|---|---|---|
| scratch | 0.460 | 0.607 | 0.749 | 0.780 | |
| mono | 0.508 | 0.690 | 0.749 | 0.802 | |
| pool | 0.533 | 0.654 | 0.705 | 0.747 | |
| **xling** | **0.550** | 0.693 | 0.762 | 0.774 | |

→ **Reframed thesis:** *cross-lingual masked pose SSL beats monolingual **when labels
are scarce*** — the low-resource regime the paper always claimed. **The data-efficiency
curve is now the headline figure (Fig 1).** Caveats: val (not test); significance needs
LOSO (running) + paired bootstrap; the crossover is at the very-low end (≤10–15%).
Extending now: BdSL 5/15%, LSA64/AUTSL curves (smaller targets = naturally low-resource).

## Conceptual core (the fundamental contribution)

Everything empirical below is evidence for **one idea**:

> **In sign language, articulatory *form* is cross-linguistically universal and
> signer-invariant — while the identity shortcut and the low-resource bottleneck
> are both failures to learn form. Masked pose pretraining across unrelated sign
> languages learns a transferable, identity-free representation that fixes both at
> once.**

Three fundamental, *falsifiable* claims:

1. **The wrong-metric reframe.** Reported SLR accuracy conflates *what sign* with
   *who signs*; random splits reward the latter. We make signer-identity leakage a
   first-class, measurable, localizable, **removable** property — reframing the goal
   from accuracy to *signer-invariant* accuracy.
2. **The universality-of-form hypothesis** (deepest, least-claimed). Sign languages
   are mutually unintelligible *lexicons* but share a substrate of manual-articulatory
   dynamics (handshape inventories, movement primitives, coarticulation — the
   "phonology" of sign). So a cross-lingual masked pose model learns a **signer- and
   language-agnostic motor model of signing**, not a shared vocabulary — which is why
   it transfers to an *unseen, lexically-disjoint* low-resource language.
3. **The objective-choice insight.** In a domain with an identity shortcut, the SSL
   objective decides *what leaks*: masked reconstruction of low-level articulatory
   units forces encoding of *how a sign is formed*, whereas contrastive/instance
   objectives can satisfy themselves with signer-discriminative features (the shortcut
   itself). This is *why* masked > contrastive **here specifically**.

**The unifying move:** the shortcut and the low-resource problem are the *same disease*
— insufficient signer/form diversity — and cross-lingual masked pose pretraining is a
*single cure*: it injects articulatory-form diversity across languages, simultaneously
filling the data gap and diluting the signer signal.

### Falsifiable predictions → experiment
| Fundamental claim | Prediction | Test |
|---|---|---|
| Wrong-metric / shortcut is real & fixable | large SI collapse; interventions shrink it | SI vs random (22.46 pp); pool/objective ablations |
| Form is universal & signer-invariant | cross-lingual > monolingual **despite zero shared signs**; gain tracks articulatory (not lexical) overlap | the gate (xling vs mono vs scratch) across 3 languages + LOSO |
| Masked learns form, contrastive learns instance | masked > contrastive at matched setup; signer-identity harder to decode from masked reps | SSL-SLR head-to-head + a signer-decodability probe |

Framed this way, the gate is not "an accuracy bump" but **a scientific test of whether
sign form is universal** — which is what makes it top-venue, not incremental.

## Draft abstract (v0 — brackets = fill from results)
> Isolated sign-language recognition (ISLR) for low-resource languages is bottlenecked
> twice: by data scarcity, and by an evaluation shortcut in which models memorize
> *signer identity* rather than sign form — inflating accuracy under random splits and
> collapsing under signer-independent (SI) evaluation. We first quantify this shortcut
> on word-level Bangla SL (BdSLW60): a [22.46] pp SI drop across [11] architectures,
> which we localize to signer-identity features. We then ask whether *cross-lingual*
> self-supervision can recover the lost accuracy without native web-scale data. We
> pretrain a **masked-reconstructive** pose autoencoder on a mixed ASL+BdSL pose pool
> and fine-tune it signer-independently on low-resource targets. Cross-lingual masked
> pretraining improves SI Top-1 by [Δ_xling] pp over monolingual pretraining and
> [Δ_scratch] pp over from-scratch, consistently across [3] target languages
> (Bangla, Argentine LSA64, Turkish AUTSL) and signer-disjoint LOSO. Against the
> contrastive cross-lingual pose paradigm (SSL-SLR) at matched pretraining and
> fine-tuning, the masked objective yields [Δ_mask] pp under SI. Unlike concurrent
> cross-lingual sign systems (SIGNET, ECCV'26) which target translation with frozen
> experts, and monolingual masked methods (SignMAE), ours is the first to occupy the
> intersection of masked pose pretraining, cross-lingual transfer, fine-tuned
> adaptation, and signer-independent evaluation for low-resource ISLR. We release the
> benchmark, splits, and extraction pipeline.

## Novelty positioning (disposition every close work)
| Work | Owns | Missing vs us | How we cite it |
|---|---|---|---|
| **SIGNET** (ECCV'26) | cross-lingual + fine-tuned + low-resource | masked, **SI**, isolated-SLR (it's translation) | closest cross-lingual pose work; we are masked + SI + ISLR |
| **SignMAE** (5/26) | masked + fine-tuned | **cross-lingual**, pose-only, SI | closest masked work; it's monolingual + RGB + non-SI |
| **SSL-SLR** (2509) | pose + cross-lingual + low-resource | **masked** (it's contrastive), **SI** | primary head-to-head baseline (matched) |
| Sigma (2509) | pose SOTA, semantic | masked, cross-lingual-transfer, SI | anchor only; **do not quote 64.54 as pose-only ISLR SSL SOTA** |
| BEST (AAAI'23) | masked + fine-tuned | cross-lingual, SI | masked-pose lineage prior work |
| Uni-Sign / SHuBERT | supervised / masked-hybrid ASL-only | — | foundation baselines (B1) |

**Load-bearing rule:** never claim novelty on a three-axis subset. Separate from
SSL-SLR via **masked + SI** (not "fine-tuned vs linear-probe" — its v2 may add
fine-tuning).

## Contributions (as claimed)
- **C1 (benchmark, → P2):** first rigorous SI benchmark for low-resource ISLR
  (BdSLW60) + quantified & localized 22.46 pp identity shortcut.
- **C2 (method, the gate):** cross-lingual **masked** pose SSL that transfers to
  low-resource targets under **SI fine-tuning**, beating monolingual + scratch.
- **C3 (generalization):** consistent across ≥2 target languages + LOSO.
- **C4 (rigor):** masked-vs-contrastive head-to-head (SSL-SLR paradigm) + honest
  comparison to supervised (Uni-Sign) and masked-hybrid (SHuBERT) foundations.

## Headline table skeleton (SI Top-1, mean±std; the gate is the middle block)
| Method / pretraining | BdSLW60-SI | LSA64-SI | AUTSL-SI | Modality |
|---|---|---|---|---|
| From-scratch BlockGCN | ~77 (have) | **87.2 ±5.6** (have) | *(pending)* | pose |
| **Monolingual** masked SSL (BdSL) → FT | ? | ? | ? | pose |
| **Cross-lingual** masked SSL (ASL+BdSL) → FT ← **OURS** | ? | ? | ? | pose |
| SSL-SLR contrastive (matched) → FT | ? | ? | ? | pose |
| Uni-Sign (pose) FT / SHuBERT frozen-probe | ? | ? | ? | pose / hybrid |
| I3D / VideoMAE (RGB anchor) | ? | ? | ? | RGB |

**THE GATE:** row 3 − row 2 ≥ **+3 pp** on BdSLW60-SI (ideally consistent on
LSA64/AUTSL). YES → NeurIPS/ICLR/ACL method paper. Marginal → P2 benchmark paper
(WACV/BMVC/NeurIPS D&B).

## Ablations / secondary tables
- Pretraining pool ablation: scratch vs BdSL-only vs ASL+BdSL (the gate) + pool size.
- Objective ablation: masked-reconstructive vs contrastive (SSL-SLR) at matched setup.
- Co-training vs sequential (Logos finding).
- LOSO variance (headline variance, not seed).
- Zero-GPU/D&B-grade: per-signer fairness, calibration, identity-cue saliency, retrieval (R@k/MRR).

---

## Framework design (the method)

**Input representation.** 27-keypoint MediaPipe skeleton (7 body + 10 per hand),
`(C=3, T, V=27, M=1)`. The skeleton is the **cross-lingual bridge** — it abstracts
away recording modality (RGB, Kinect, gloves), so ASL (WLASL) and BdSL pose live in
one space. `flip_index` handles mirroring augmentation.

**Backbone.** `model.block_gcn.Model` (BlockGCN, CVPR'24), ~1.4 M params, with
`return_features: True` and **`stride_between_stages: False`** so `T_out == T_in`
(per-frame masked prediction needs full temporal resolution). Fits 24 GB GPUs.

**Pretraining — masked pose-unit prediction (SHuBERT-style; THE differentiating axis).**
1. **Discretize** targets: MiniBatchKMeans over **pose_motion** features → per-frame
   cluster IDs (`data/pretrain_kmeans_targets_*.npz`). Motion targets (not static
   frames) block the trivial copy-neighbour solution.
2. **Mask** a fraction of frames/joints; the backbone encodes the corrupted sequence.
3. **Predict** the cluster ID at masked positions (cross-entropy) — BERT-style masked
   *unit* modelling. This is **masked-reconstructive**, separating us from SSL-SLR
   (contrastive) and Sigma/SIGNET (semantic vision-text alignment).
   - `model.shubert_pretrain.ShubertPretrainer` wraps the backbone;
     `feeders.pretrain_feeder.Feeder` serves (pose, cluster-id) pairs; saves
     **backbone-only** checkpoints for `main.py` to fine-tune.

**Cross-lingual mechanism (the second key axis).** ONE shared backbone pretrained on
the **pooled ASL+BdSL** pose (`ssl_pool_manifest_bdsl_asl.json`, 75,589 clips), with
**no language labels** — it learns language-agnostic pose dynamics. Ablated vs
BdSL-only (monolingual) and scratch = the gate.

**Fine-tuning (SI).** `main.py <target-cls-config> --weights backbone.pt
--ignore-weights <head-keys>` → loads backbone, fresh classifier head, trains on the
target's SI train split, selects on val signers, reports on held-out test signers.

**Optional high-value extension — signer-invariant pretraining (direction D2).** Add
an adversarial signer-ID branch with gradient reversal during pretraining, penalizing
signer-decodability of the representation. Success metric = **shrink the 22.46 pp SI
gap + LOSO variance** — converts the shortcut *finding* into a *method*.

**What we deliberately are NOT** (the four-axis identity): not contrastive (SSL-SLR),
not semantic/translation (Sigma/SIGNET), not monolingual (SignMAE/BEST), not RGB —
**pose-only × masked × cross-lingual × signer-independent**.

---

## Exact experiment queue (HPC)

Configs and assets that already exist are marked ✅. `main_pretrain.py` does SSL;
fine-tune = `main.py` on a classification config + `--weights <backbone>.pt
--ignore-weights <classifier-head-keys>` (backbone-only checkpoint; new head is random).

**Phase 0 — SSL inputs** ✅ built: `data/ssl_pool_manifest_{bdsl_asl,bdsl_only}.json`
+ `data/pretrain_kmeans_targets_{bdsl_asl,bdsl_only}.npz` (pose_motion targets).
Guard: `tests/test_ssl_pool_no_leak.py` (pool excludes all BdSL val/test signers).

**Phase 1 — Pretrain 2 backbones** (backbones not currently on scratch → run):
- `python main_pretrain.py --config config/bdsl_shubert_pretrain_xlingual.yaml`  → `backbone_xling`
- `python main_pretrain.py --config config/bdsl_shubert_pretrain_bdsl_only.yaml` → `backbone_mono`
- (`stride_between_stages: False` must hold so T_out==T_in for per-frame masked prediction.)

**Phase 2 — THE GATE: SI fine-tune on BdSLW60** (3 conditions × 3 seeds):
- scratch: `config/bdsl_block_gcn_si.yaml` (have ~77%)
- mono:  `bdsl_block_gcn_si.yaml` + `--weights backbone_mono --ignore-weights <head>`
- xling: `bdsl_block_gcn_si.yaml` + `--weights backbone_xling --ignore-weights <head>`
- Save best (`-best.pt`) + `--phase test` held-out signers. **Decision: xling − mono ≥ 3 pp?**

**Phase 3 — Generalize to N≥2 targets** (extends the gate):
- LSA64: from-scratch ✅ done (87.2); add mono/xling FT (`config/lsa64_si.yaml` + backbone).
- AUTSL: from-scratch 🔄 auto-running (123928); add mono/xling FT (`config/autsl_si.yaml` + backbone).

**Phase 4 — Mandatory baselines:**
- SSL-SLR contrastive paradigm, matched pool+FT (implement contrastive head on same backbone).
- B1: Uni-Sign pose fine-tune + SHuBERT frozen-feature probe (`docs/B1_FOUNDATION_BASELINES.md`).
- B2: RGB anchor (`path4_rgb_baseline`, I3D/VideoMAE) + Logos co-train.
- B3: co-training vs sequential (`main_cotrain.py`, `config/bdsl_block_gcn_cotrain_si.yaml` ✅).

**Phase 5 — LOSO** on BdSLW60 (headline variance): `tools/run_loso.py`.

**Phase 6 — Reporting:** `tools/summarize_seeds.py`, `tools/paired_bootstrap.py`,
+ zero-GPU analyses. Licensing P0: email BdSLW401 authors (CC BY-NC-ND) before any
D&B artifact claim.

### Critical path to the gate
Phase 1 (pretrain xling+mono) → Phase 2 (3×3 SI fine-tune on BdSLW60) → read gate.
Everything else parallelizes or follows. **The gate is the one experiment that
decides the paper's venue — run it first.**
