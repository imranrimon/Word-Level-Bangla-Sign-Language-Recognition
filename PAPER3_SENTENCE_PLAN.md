# Paper 3 — Sentence-Level Bangla Sign Language Recognition

**Working title**: *"From Words to Sentences: BdSLW401-Pretraining
Transfers to Bangla Sign Language Sentence Recognition."*

**Target venue**: workshop (LREC sign-language workshop, ACL sign-
language workshop, sign-language WACV / NeurIPS workshops). One step
below Paper 2 in venue tier.

**Estimated time-to-submission**: ~2 weeks after Paper 2's BdSLW401-
pretrain checkpoint is available. Total wall-time ~3 weeks.

---

## 1. Claim (one paragraph)

Bangla sign-language datasets are split between word-level corpora
(BdSLW60, BdSLW401) and sentence-level corpora (BdSLW102_A). No
published Bangla SLR paper bridges the two granularities. We establish
the first baseline on BdSLW102_A's sentence track (19 sentence classes,
1,351 clips) using skeleton-only BlockGCN, then demonstrate that the
**BdSLW401-Pretrain Transfer (BPT)** recipe — supervised pretraining on
the BdSLW401 word-level corpus followed by classifier-head replacement
and low-LR fine-tune — improves sentence-level Top-1 by [X] pp over
scratch training, even though pretraining was at word granularity. This
suggests that word-level pose representations carry compositional
structure that transfers to sentence-level classification, motivating
larger-scale Bangla word-level corpus collection as a foundation for
sentence-level SLR.

## 2. Two contributions

1. **First published sentence-level baseline on BdSLW102_A.** A clean
   train/val/test split (deterministic random, seed 0, 70/15/15) +
   BlockGCN scratch Top-1 + the bundled pose data
   (`data/bdslw102a_sentence/`) released alongside the paper.
2. **Word-to-sentence transfer via BPT.** Shows that pretraining on
   BdSLW401 (word-level, 102k clips) + low-LR sentence finetune lifts
   sentence Top-1 vs scratch.

This is a 4-page workshop paper, not a full venue submission. The
contribution is the *transfer demonstration*, not a new method.

## 3. Dataset

| Property | Value |
|---|---|
| Source | BdSLW102_A (the "Word Label" version — actually sentence-level despite the file naming) |
| Pose cache | `data/bdslw102_a_pose_cache/Sentence/` |
| Bundled NPY | `data/bdslw102a_sentence/` |
| Sentence classes | 19 (sentence ID 13 missing from disk; documented in §6) |
| Clips (withBg variant) | 1,351 |
| Split | random 70/15/15 (seed 0): 947 train / 202 val / 202 test |
| Signer folders | 19 (signers 0..19 except 13) |

Sentence text examples (from `BdSLW102_A/Sentence Label.xlsx`):
- "আপনি কেমন আছেন?" (How are you?)
- "আমি ভাল আছি" (I am well.)
- "আপনার নাম কি?" (What is your name?)

Listed as 20 sentences in the label file but sentence 13 is absent in
the pose cache — paper acknowledges this in §6 limitations.

## 4. Methodology

### 4.1 Backbone

Same BlockGCN (`model.block_gcn.Model`) used in Paper 2 and the main
paper. Same 27-keypoint skeleton graph. Same pose feeder
(`feeders.feeder.Feeder`).

### 4.2 Two training conditions

| Condition | Initialization | LR schedule |
|---|---|---|
| **Scratch** | Random init | base_lr=0.1, step=[25,35], 40 epochs |
| **BPT** (proposed) | BdSLW401 backbone checkpoint (from Paper 2) | base_lr=0.01, step=[25,35], 40 epochs |

Both keep `--ignore-weights fc.weight fc.bias` so the 401-way → 19-way
head swap is automatic.

### 4.3 Sentence-classification protocol

Single-label classification: each sentence-mp4 is one of 19 classes.
This is straightforward classification, NOT continuous sentence parsing
or sequence-to-sequence — pose `(C=3, T, V=27, M=1)` → 19-way logits.

## 5. Experiments

### Compute estimate

| Run | GPU-h on RTX 8000 |
|---|---:|
| Scratch BlockGCN | ~1 |
| BPT-finetuned BlockGCN | ~1 |
| Optional: full ablation (3 architectures × 2 conditions × 1 seed = 6 runs) | ~6 |
| **Total** | **2–6 GPU-h** — overnight on one GPU |

The BdSLW401 pretrain backbone is *reused* from Paper 2 — zero extra
pretraining cost.

### S1 — Sentence-level Top-1

| Architecture | Scratch | BPT | Δ |
|---|---:|---:|---:|
| BlockGCN | | | |
| (optional: ST-GCN, CTR-GCN) | | | |

### S2 — Sentence-confusion analysis

Confusion matrix (19 × 19) — which sentence pairs are easily confused?
Linguistic interpretation: sentences sharing common words ("আমি ভাল
আছি" vs "আপনি ভাল থাকবেন") likely confused more than unrelated ones.

### S3 — Word-content overlap vs sentence-class accuracy

Each sentence is a Bangla word sequence. Map each sentence to its
word-bag, compute overlap of word-bag with BdSLW401's 401-word vocab,
and plot vs per-class Top-1. If correlation is positive: BPT helps
sentences whose constituent words appeared in the pretraining corpus.
(One-figure analysis.)

## 6. Limitations / honest disclosures (§6 of the paper)

- Sentence class 13 is absent from the pose cache; bundled NPY uses 19
  classes not 20. Documented in §3.
- 1,351 clips and 19 classes is small; results may have high variance.
  Use 1 seed (=0); the BdSLW401-pretrain backbone variance dominates.
- We use the `withBg` (masked) variant only; the `raw` variant is
  reserved for future appendix work.
- No comparison to non-skeleton baselines (no RGB CNN, no audio) — out
  of scope for this skeleton-only workshop paper.
- The random train/val/test split is not signer-disjoint within the 19
  signer folders. A follow-up SI split is sketched in §7.

## 7. Time-to-submission

| Phase | Time |
|---|---|
| Wait for BdSLW401 backbone from Paper 2 | (parallel with Paper 2's Week 2) |
| Run scratch + BPT on BdSLW102_A | ~1 day (overnight) |
| Analysis (S2, S3) + figures | ~3 days |
| Draft + polish | ~1 week |
| **Total wall-time** | **~2 weeks after Paper 2 backbone available** |

## 8. Cross-reference

- Paper 2 (cross-dataset word-level) is the parent — Paper 3 reuses
  Paper 2's BPT checkpoint, citing it as the source.
- The main paper (identity-shortcut + 3 interventions) is the cousin
  — Paper 3 doesn't directly depend on it.

## 9. What still needs human input

1. **Whether to even submit Paper 3**. If Paper 2's BPT result is
   strong, Paper 3 is bonus. If Paper 2's BPT result is weak, Paper 3's
   cross-granularity claim is also weakened — consider folding the
   sentence-level finding into a Paper 2 appendix table instead.
2. **Signer-disjoint sentence split**. The 19 signer folders permit a
   leave-one-signer-out-style split. This is a 1-day code task; the
   random split in §3 is the v1 baseline.
3. **Bangla-native review of the per-sentence confusion patterns** for
   §S2's qualitative analysis paragraph.
