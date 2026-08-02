# Option B — Cross-lingual Pose SSL paper (scope sketch)

This is the *separate-paper* scope for the cross-lingual SSL line of work,
in case the main-paper ablation (Option A — `T5` with BdSL-only vs BdSL+ASL
variants) returns a strong "ASL helps" result that you want to amplify into
a standalone methodological paper.

**Tentative title**: *PoseHuBERT: Cross-Lingual Self-Supervised Pose
Pretraining Transfers Across Sign Language Families*

**This document is a planning artifact, not a commitment.** Decide
"yes go" only after Option A's T5 result is in hand and the gap exceeds
3 pp Top-1.

---

## 1. The claim a standalone paper would need to make

Option A in the main paper claims:
> "Cross-lingual SSL on ASL pose improves BdSL Top-1 by X pp."

A standalone Option B paper needs the **stronger, more general** claim:
> "Cross-lingual SSL on ASL pose improves SLR Top-1 across **multiple
> target sign languages**, demonstrating that pose dynamics generalize
> across SL families."

The plural is load-bearing. With only BdSL as target, reviewers will say:
"That's a single data point. Could be Bangla-specific."

## 2. What's needed beyond what we already have

| Need | What we have today | What's missing |
|---|---|---|
| Pretraining corpus | WLASL (21k pose clips) + ASL-Citizen (83k videos, pose extraction needed) | None — pretraining side is done |
| Target language 1: BdSL | BdSLW60-SI, full pipeline | None |
| **Target language 2** | nothing | AUTSL (Turkish SL, ~38k clips) — need download + pose extract + SI split + fine-tune |
| **Target language 3** | nothing | One of: ChineseSL (LSA-MSR), KETI (Korean SL), or LSA64 (Argentinian SL) |
| Ablations | BdSL+ASL vs BdSL-only | × 3 target languages each |
| Comparison to non-cross-lingual SSL | SignBERT, SHuBERT (single-language) | Re-implement or skip |

## 3. Experimental matrix that justifies the paper

| Pretrain pool \ Target | BdSL | TSL | LSA64 (or CSL) |
|---|:---:|:---:|:---:|
| None (supervised baseline) | ✓ | ✓ | ✓ |
| In-language only (BdSL pool / TSL pool / LSA64 pool) | ✓ | ✓ | ✓ |
| ASL only (WLASL + ASL-Citizen) | ✓ | ✓ | ✓ |
| Pooled (all 4 corpora) | ✓ | ✓ | ✓ |

→ 12 cells × 3 seeds = 36 fine-tune runs + 4 SSL pretrains. ≈ 80–120 GPU-h
total, ~2 GPU-weeks on 1× RTX 8000 or ~3 wall-days on 8-GPU HPC.

## 4. Compute and timeline (single researcher)

| Week | Milestone |
|---|---|
| 1 | Download AUTSL, ChineseSL/LSA64. Pose-extract both with MediaPipe (~24 GPU-h) |
| 2 | SI splits for both target languages; fine-tune harness ports |
| 3 | All 4 SSL pretrains complete |
| 4 | 36 fine-tunes (3 seeds × 4 pool variants × 3 targets) |
| 5 | Aggregation, significance tests, ablations on mask-ratio / cluster-K |
| 6–7 | Draft writing, related-work positioning vs SignBERT/SHuBERT/SignVQ |
| 8 | Reviewer-defense appendix, polish, submission |

**Realistic: ~2 months extra after main paper submits.**

## 5. Venue targeting

| Venue | Fit | Why |
|---|---|---|
| **CVPR / ICCV** | Best fit | Methods venue; "first cross-lingual pose SSL for SLR" is a clear contribution |
| NeurIPS | Good | If cluster-balance + statistical-rigor is heavy |
| ACL / EMNLP | Less ideal | Mostly text venues; sign language sometimes accepted under MM track |
| WACV | Backup | Lower bar, faster turnaround |
| AAAI | Backup | Lower bar than CVPR but applied-ML-friendly |

## 6. Risks

1. **Single-language gain doesn't generalize.** If BdSL gains 3 pp but TSL gains 0 pp, the standalone paper's headline is broken. **Mitigation**: prepare a backup framing ("language-family-specific cross-lingual transfer") that's still defensible but less strong.
2. **SHuBERT scoop.** Gueuwou et al. (Findings 2024) already did SHuBERT for ASL. A direct extension to cross-lingual might be in their pipeline. **Mitigation**: cite carefully, frame as "first to demonstrate cross-lingual *transfer*" rather than "first to do pose-SSL for SL."
3. **AUTSL data licensing.** AUTSL is freely available but requires registration. If we want zero-friction reproducibility, may need to mention this in limitations.
4. **Two-paper compute cost.** Even on HPC, an extra 2 weeks of SSL pretrains is a real time investment. If the main paper's Option A result is borderline, this paper isn't worth chasing.

## 7. Recommendation

**Decide YES on Option B paper if** the Option A T5 row shows BdSL+ASL > BdSL-only by **≥ 3 pp Top-1** at p < 0.05.

**Decide NO if** the gap is < 1 pp or non-significant — fold the result into the main paper's appendix as a "cross-lingual is comparable" footnote and move on.

**Decide MAYBE if** the gap is 1–3 pp — discuss with co-authors; depends on how much you want to invest in this thread vs the main paper's other interventions (Path 1, Path 2).

## 8. What changes in the main paper if Option B happens

| Main paper section | Without Option B | With Option B (later) |
|---|---|---|
| §1 Intro | Same identity-shortcut narrative | Adds 1 sentence: *"We further explore cross-lingual SSL in a companion paper [cite Option B]"* |
| §4.2 PoseHuBERT method | Full description of cross-lingual approach | Defer method details to companion; main paper just uses results |
| §5 T5 table | Both variants reported, headline is whichever wins | Both variants reported, with citation to Option B for the cross-lingual analysis |
| §6 Conclusion | "Cross-lingual SSL is a promising direction" | "Cross-lingual SSL is a promising direction; see companion paper" |

→ Main paper is mostly unchanged; the citation to Option B replaces an "open question" line.

## 9. Decision matrix (after Option A T5 lands)

```
                  ASL > BdSL-only          ASL ≈ BdSL-only          ASL < BdSL-only
                  by ≥3pp                  (within CI)              by ≥1pp
Option B paper?   YES, GO                  NO                        NO (negative result
                                                                      goes in main paper
                                                                      footnote)
Main paper        Lead with cross-         Report both, frame as     Lead with BdSL-only,
framing           lingual contribution     "comparable, supports     mention ASL pool
                                            pool expansion"           briefly
```
