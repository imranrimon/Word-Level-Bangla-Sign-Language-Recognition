# Paper 2 — Cross-Domain Bangla Sign Language Recognition

**Working title**: *"Cross-Domain Bangla Sign Language Recognition:
Vocabulary Drift, Recording-Condition Robustness, and the Case for
BdSLW401-Pretraining."*

**Target venue**: BMVC, WACV, NeurIPS D&B, or EMNLP Findings.
**Estimated time-to-submission**: ~4 weeks (1 week scaffolding + 1 week
training + 2 weeks writing).

---

## 1. Claim (one paragraph)

Existing Bangla SLR papers train and evaluate within a single dataset,
producing 99 %-tier numbers that don't survive evaluation under any
different distribution. We construct the **first cross-domain Bangla
SLR benchmark** over three publicly-available word-level corpora —
BdSLW60 (60 words, 18 signers, multi-trial), BdSLW401 (401 words, 25
signers, large-scale), and BdSLW60-SingleTrial (same 60-word vocabulary
as BdSLW60 but a different recording protocol with one trial per signer).
For every (model, source-training-set, target-eval-set) cell we report
Top-1 on the vocabulary-aligned subset of the target. We further propose
**BdSLW401-Pretrain Transfer (BPT)** — a simple recipe of supervised
pretraining on BdSLW401 followed by target-dataset fine-tuning — and
show it outperforms scratch training on small Bangla SLR datasets by
[X] pp Top-1, establishing BdSLW401 as the de-facto pretraining base
for Bangla SLR.

## 2. Three contributions

1. **First cross-domain Bangla SLR benchmark.** A vocabulary-aligned
   transfer matrix isolating two distinct domain-shift axes — *vocabulary
   drift* (BdSLW60 ↔ BdSLW401) and *recording-condition shift*
   (BdSLW60 ↔ BdSLW60-SingleTrial). Three backbone architectures
   (ST-GCN, CTR-GCN, BlockGCN) span 2018→2024 evolution.
2. **Vocabulary alignment table.** First documented mapping between
   BdSLW60 (60 words) and BdSLW401 (401 words). Released as JSON.
3. **BPT recipe.** Supervised pretraining on BdSLW401 + classifier-head
   replacement + low-LR fine-tune on the target. Validated to outperform
   scratch training by [TBD] pp Top-1 on the small target.

## 3. Datasets

| Dataset | Role | Words | Clips | Signers | Splits | Status |
|---|---|---:|---:|---:|---|---|
| **BdSLW60** | source + target | 60 | 9,307 | 18 | SI (this work) | ✅ NPY bundles ready |
| **BdSLW401** | source + target | 401 | 51,098 | 25 | authors' train/val/test | ✅ NPY bundles ready (`data/bdslw401_si/`) |
| **BdSLW60-SingleTrial** | **target only** | 60 (same as BdSLW60) | 774 | 18 | eval-only bundle | ✅ ready (`data/bdsl60_singletrial_eval/`) |

Why a 3-way framing with 2 train + 1 eval-only:

| Pair | Variation isolated |
|---|---|
| BdSLW60 ↔ BdSLW401 | Cross-dataset (different vocab, different collection team) |
| BdSLW60 ↔ BdSLW60-SingleTrial | **Cross-recording** (same vocab) — clean isolation |
| BdSLW401 ↔ BdSLW60-SingleTrial | Combined shift (vocab + recording + signers) |

BdSLW60-SingleTrial is too small (774 clips) to train robustly, so it is
eval-only. This is a feature: it isolates the cross-recording shift
without confounding it with training-set quirks.

## 4. Methodology

### 4.1 Backbone architectures

Three architectures span 2018→2024 skeleton-SLR evolution. All use the
27-keypoint graph from SLGTFormer (Song 2022, audit fix #9) and consume
the same `(N, C, T, V, M)` pose tensor.

| Architecture | Year/venue | Why |
|---|---|---|
| ST-GCN | Yan AAAI 2018 | Classical baseline. |
| CTR-GCN | Chen ICCV 2021 | Mid-era SOTA. |
| BlockGCN | Zhou CVPR 2024 | Recent SOTA + our headline. |

If time-constrained, run only BlockGCN for v1 (Plan C in the seed/arch
trade-off discussion). The three-architecture sweep is recommended but
not required.

### 4.2 Cross-domain evaluation protocol

For each (source-training-set, architecture) pair (= 2 sources × 3 archs
= 6 models):
1. Train on source's training split.
2. Evaluate on the source's val/test (within-source reference — T1).
3. For each target dataset, evaluate on the **vocabulary-aligned subset**
   of the target's val/test. Drop target clips whose label has no match
   in the source vocabulary.

A clip is in the "vocabulary-aligned subset" iff its target-vocabulary
label has a confirmed match (lexical + Bangla-native curator approval)
in the source vocabulary. For BdSLW60 ↔ BdSLW60-SingleTrial the subset
is the full 60-class vocabulary (trivially equal). For BdSLW60 ↔
BdSLW401 the subset is the curator-confirmed intersection (estimate:
30–55 words out of 60).

Important caveat (audit fix #7): when evaluating ON BdSLW60-SingleTrial,
filter clips to **the 295-clip held-out subset** (signers NOT in SI
train) for the headline number. The full-bundle 774-clip number goes in
the appendix.

### 4.3 Proposed: BdSLW401-Pretrain Transfer (BPT)

1. **Pretrain**: train BlockGCN on the full BdSLW401 train split with
   the 401-way classifier head. Default schedule (100 epochs, base LR
   0.1, step 60/80). Save backbone state_dict after best Top-1 epoch.
2. **Head swap**: replace the 401-way `fc` with a `num_classes_target`-
   way `fc` (60 for BdSLW60). Initialize the new head with
   `nn.init.trunc_normal_(std=0.02)`.
3. **Fine-tune**: train on the target dataset for 40 epochs with:
   * base LR = 1e-2 (10× lower than scratch)
   * step at [25, 35]
   * everything else identical to scratch config.

The contribution is showing this simple recipe works for low-resource
Bangla SLR — there's no published Bangla SLR transfer-learning result
to compare against.

## 5. Experiments

### Compute estimate (Plan B = 1 seed × 3 architectures × 2 train sources)

| Phase | Runs | GPU-hours (RTX 8000) |
|---|---:|---:|
| Within-source scratch (T1) | 6 (3 arch × 2 src) | ~55 |
| Cross-dataset eval (T2) | 0 training | < 5 (inference only) |
| BPT pretrain (BdSLW401 backbone — shared across BPT runs) | already in T1 | 0 extra |
| BPT finetune (T3) on BdSLW60 | 3 (1 per arch) | ~7 |
| Vocabulary-overlap analysis (T4) | 0 training | minutes |
| **Total** | **9 runs** | **~70 GPU-h ≈ 3 GPU-days local, or ~10 h on 8-GPU HPC** |

(BlockGCN-only Plan C: 3 runs ≈ 23 GPU-h ≈ ~1 day local.)

### T1 — Within-source Top-1 (reference)

| Source | ST-GCN | CTR-GCN | BlockGCN |
|---|---:|---:|---:|
| BdSLW60 | | | |
| BdSLW401 | | | |

### T2 — Cross-domain transfer matrix

| Trained on \ Eval on | BdSLW60 | BdSLW401 | BdSLW60-SingleTrial (held-out 295) |
|---|---:|---:|---:|
| BlockGCN / BdSLW60 | (diag, T1) | aligned subset | full 60 classes |
| BlockGCN / BdSLW401 | aligned subset | (diag, T1) | aligned subset |
| (repeat for ST-GCN, CTR-GCN) | | | |

Six off-diagonal cells × 3 architectures = 18 transfer numbers.

### T3 — BPT effect on BdSLW60

| Architecture | BdSLW60 scratch (T1) | BdSLW60 BPT | Δ |
|---|---:|---:|---:|
| ST-GCN | | | |
| CTR-GCN | | | |
| BlockGCN | | | |

If average Δ ≥ 3 pp Top-1, you have a top-tier headline.

### T4 — Vocabulary overlap vs accuracy correlation

Scatter plot: x-axis = Jaccard overlap between source and target
vocabulary (per source-target pair), y-axis = transfer Top-1. A positive
correlation supports the "vocabulary drift dominates transfer" hypothesis.

## 6. Reviewer-defense

| Anticipated critique | Defense |
|---|---|
| "Only 1 seed" | §3 disclosure: signer/source axes dominate variance, init noise ≤ 0.6 pp from main paper. |
| "Only 3 datasets" | They are *all* the publicly distributed Bangla word-level SLR datasets we are aware of. Acknowledge in §7. |
| "BdSLW60-SingleTrial isn't really a separate dataset" | Acknowledge: it's a same-vocabulary controlled cross-recording setting — that's *why* it isolates the recording-shift axis cleanly. |
| "Vocabulary alignment is subjective" | Release the JSON; document the lexical + Bangla-native curator review protocol. |
| "BPT is just transfer learning" | Yes — and that's the point. No one has done it for Bangla SLR. We measure the gap and publish the recipe. |
| "Why not include BdSL-MNIST etc." | Image-only handshape datasets; see companion sister paper (Path 3). |
| "Why not BdSLW102_A" | Sentence-level only; not word-aligned. See **Paper 3** companion sentence-recognition paper. |

## 7. Time-to-submission budget

| Week | Milestone |
|---|---|
| 1 | Vocabulary alignment table built + curator-reviewed (BdSLW60 ↔ BdSLW401); training configs ready |
| 2 | BdSLW401 training (16 GPU-h) + BdSLW60 already trained (main paper); cross-dataset eval runs |
| 3 | BPT finetune; T4 analysis; draft §3, §4, §5 |
| 4 | Full draft (§1-§7), reviewer-defense polishing, submit |

## 8. What still needs human input

1. **Vocabulary alignment review.** Script produces lexical matches; you
   (Bangla-native) confirm semantic equivalence for the edit-distance
   candidates. ~2 hours of review.
2. **BdSLW401 class-name list.** The pose-cache filenames give numeric
   IDs (W001..W401). Extract the actual Bangla word names from
   `data/bdslw401_raw/bdsl words-complete.pdf` (or copy from the
   BdSLW401 paper appendix). Otherwise alignment will work only at the
   numeric-id level.
3. **Whether to run all 3 architectures or just BlockGCN for v1.**
   Plan B = all 3 (~3 GPU-days); bare minimum = BlockGCN only
   (~1 GPU-day).

## 9. Cross-reference to main paper

This paper depends on main-paper findings to motivate its protocol:

- *§1*: cites main paper's "22.46 pp identity-shortcut gap on BdSLW60-SI"
  to motivate why within-dataset numbers are misleading.
- *§4.2*: SI protocol from main paper extends to the cross-dataset
  setting; the SingleTrial caveat (audit fix #7) is reused.
- *§6*: per-signer identity-shortcut analysis applied to the cross-
  domain matrix.

Either paper stands alone if the other is delayed.

## 10. Cross-reference to Paper 3 (sentence-level companion)

Paper 3 (BdSLW102_A sentence-level Bangla SLR) is the natural companion
covering **word-to-sentence granularity transfer**: we show BPT (this
paper's recipe) lifts BdSLW102_A sentence-level Top-1 by [Y] pp, even
though the pretraining was at word granularity. This extends the BPT
recipe's scope from "Bangla word→word" to "Bangla word→sentence" — the
same backbone pretraining recipe transfers across granularities.

If Paper 3 lands first, this paper cites it as "extended to sentence
classification in [Paper 3]." If this lands first, Paper 3 cites it as
"the BPT recipe of [Paper 2]." Either ordering works.
