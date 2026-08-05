# P2 — Draft prose (abstract + intro)

Companion to [`PAPER_P2_BENCHMARK.md`](PAPER_P2_BENCHMARK.md) (plan/status/tables).
[BRACKETS] = fill from final results.

## Abstract (draft)
Word-level sign language recognition (SLR) for low-resource languages is routinely
reported at near-ceiling accuracy — but under random train/test splits that let models
memorise *signer identity* rather than sign form. We present the first rigorous
**signer-independent (SI) benchmark** for word-level Bangla SLR (BdSLW60): a fixed
signer-disjoint split evaluated across [8] architectures. Accuracy collapses from
~99% (random split) to **~77% (SI)** for a strong skeleton backbone — a **22.46 pp
identity shortcut** consistent across models. We localise the shortcut with a
feature-isolation study (MediaPipe pose vs DINOv2 hand/face crops through one temporal
head), and show a **label-free intervention** — monolingual masked pose self-supervised
pretraining — significantly reduces the gap **where labels are scarce** (+[4.8] pp at
10% of labels, p<0.001). We report signer-fold (LOSO) variance as the headline noise,
and release the canonical splits, pose-extraction pipeline, evaluation harness, and
Croissant metadata. Bangla — a top-10 spoken language — has been absent from top-venue
SLR; our benchmark provides a reusable signer-independent protocol and a concrete
instance of the community's call for out-of-distribution evaluation.

## 1. Introduction (draft)
**The problem.** Published word-level SLR accuracies are often near-perfect, and
Bangla SLR is no exception (README-lineage claims of ~99%). But these use *random*
clip-level splits, which place different repetitions of the same word **by the same
signer** in both train and test. A model can then exploit **signer identity** — a
shortcut (Geirhos et al.) — inflating accuracy without learning sign form.

**The gap.** Signer-independent evaluation is established for ASL (ASL Citizen, NeurIPS
D&B 2023) and Turkish (AUTSL, 2020), but **no rigorous SI benchmark exists for Bangla**,
and Bangla SLR has never appeared at a top venue despite Bangla being a top-10 spoken
language with a large Deaf community and no trustworthy recognition benchmark.

**Our approach** follows the accepted shortcut-paper arc — **quantify → localise →
intervene → release**: (i) a fixed signer-disjoint split; (ii) an 8-architecture SI
table exposing a 22.46 pp collapse; (iii) feature isolation localising *where* identity
leaks; (iv) a label-free intervention (monolingual masked pose SSL) that shrinks the gap
in the low-resource regime; (v) released artifacts + protocol.

**Contributions.**
1. The first rigorous **signer-independent benchmark** for word-level Bangla SLR.
2. **Quantification** of a 22.46 pp identity shortcut, consistent across architectures,
   with **LOSO** signer-fold variance as the headline metric.
3. **Localisation** of the shortcut via pose-vs-DINOv2 feature isolation.
4. A **label-free intervention** (masked pose SSL) that reduces the gap where labels
   are scarce, plus a recipe-control finding (fine-tune LR is worth ±13 pp).
5. **Released artifacts**: splits, pipeline, harness, hashes, Croissant metadata.

## Related work (stub — to expand)
SI/OOD eval: AUTSL, ASL Citizen, MSLR; shortcut learning: Geirhos, DeGrave, RESOUND;
pose SLR & SSL: SignBERT+/BEST/MASA/SHuBERT, BlockGCN/CTR-GCN; Bangla: BdSLW60/BdSLW401.
