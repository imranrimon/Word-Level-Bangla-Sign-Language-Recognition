# P2 — Benchmark Paper (the de-risked paper)

The **safe-harbor** paper: publishable on rigor alone, independent of whether the
cross-lingual method (P1) survives. It is the reason the whole program exists —
a trustworthy signer-independent (SI) evaluation for a low-resource sign language,
with the identity shortcut quantified, localized, and partially removed.

## Working title
**"How Much of Bangla Sign Recognition Is Just Signer Recognition? A Rigorous
Signer-Independent Benchmark, Shortcut Diagnosis, and Interventions for
Word-Level BdSL."**

## Thesis (one line)
Reported word-level BdSL accuracy is inflated by a **signer-identity shortcut**;
under a rigorous SI protocol accuracy collapses by **22.46 pp**, and we quantify,
localize, and partially close that gap — delivering the field's first trustworthy
Bangla SLR benchmark + reusable protocol.

## Narrative arc (the accepted shortcut-paper template)
**Quantify → Localize → Intervene → Release** (RESOUND @ ECCV, DeGrave @ Nature MI,
ASL Citizen @ NeurIPS D&B).

## Contributions
- **C1 — SI benchmark.** First signer-independent benchmark for word-level BdSL:
  11 architectures × 3 seeds on the canonical signer-disjoint split (train {1,4,5,6,
  8,9,11,12} / val {15} / test {2,13}). Headline variance is **LOSO** (signer folds),
  not seed noise.
- **C2 — Shortcut quantification.** Random-split ~99% collapses to **~77% Top-1 SI**
  for BlockGCN → a **22.46 pp identity shortcut**, consistent across architectures.
- **C3 — Shortcut localization.** Feature isolation (MediaPipe pose vs DINOv2 hand/
  face crops through one temporal head) shows *where* identity leaks (Option B).
- **C4 — Interventions that shrink the gap.** (a) **Monolingual masked pose SSL**
  pretraining helps under SI (mono 0.80 > scratch 0.78 val, BdSLW60 @ lr0.01) — a
  cheap, label-free intervention; (b) recipe control (fine-tune LR is worth ~13 pp
  on pretrained inits — itself a reproducibility lesson); (c) [pending] signer-
  invariant / augmentation interventions.
- **C5 — Released artifacts.** Canonical splits, extraction pipeline, eval harness,
  and result hashes (pose arrays gated by BdSLW401 CC BY-NC-ND — see licensing).

## Tables / figures
1. **Table 1 — SI benchmark**: 11 archs × {random split, SI split}, Top-1/Top-5,
   mean±std → the 22.46 pp gap column is the hook.
2. **Fig 1 — the gap** across architectures (bar chart, random vs SI).
3. **Table 2 — feature isolation**: pose vs DINOv2, SI vs SD gap per representation.
4. **Table 3 — interventions**: scratch vs monolingual-SSL vs [signer-invariant],
   SI Top-1 + LOSO.
5. **Fig 2 — LOSO per-signer breakdown** (fairness/variance).
6. **[D&B extras]** per-signer fairness, calibration, identity-cue saliency,
   dictionary-retrieval metrics (R@k, MRR).

## Honest positioning vs prior SI work
- AUTSL quantified a random-vs-SI gap (Turkish, 2020, IEEE Access) and ASL Citizen
  institutionalized signer-disjoint splits (NeurIPS D&B 2023) — **cite as precedent**.
  Our novelty: **first for Bangla** (absent from top venues), **11-architecture**
  breadth, **localization** (feature isolation), and **label-free interventions**.

## Results status (2026-08-05) — data mostly complete

**Table 1 draft — SI benchmark, Top-1 mean ± std (val, policy-aware `summarize_seeds`):**
| Architecture | N | Top-1 SI |
|---|---|---|
| Adaptive GNN | 3 | **88.80 ± 1.16** |
| SLGTFormer | 2 | 86.64 ± 0.33 |
| Attention GNN | 3 | 85.19 ± 1.47 |
| ST-GCN vanilla | 3 | 84.73 ± 2.17 |
| GNN + Bi-LSTM | 3 | 83.05 ± 2.46 |
| Pose-LSTM | 3 | 74.20 ± 0.95 |
| BlockGCN | pilot | ~76.95 (clean; aggregate row polluted — re-run) |
| GNN + Transformer | 3 | 30.89 ± 18.97 (did not converge — re-run) |

- ✅ **Shortcut**: random ~99% → **~77% SI** (BlockGCN) = **22.46 pp gap**; SD
  counterparts on disk for the per-arch gap column.
- ✅ **Intervention — monolingual masked pose SSL helps under SI**: `mono > scratch`
  significant where labels are scarce (BdSL@10% +4.8 pp p<0.001; @5% +6.0 pp) — a
  label-free intervention that shrinks the gap most in the low-data regime.
- ✅ **Recipe control**: fine-tune LR worth ±13 pp on pretrained inits.
- ✅ **Cross-dataset SI**: LSA64 from-scratch 86.7% test (3-seed) + AUTSL official split.
- 🔄 **LOSO** (headline signer-fold variance) — array `124307`, 11 folds running.
- ⬜ **Feature isolation** (Option B, pose vs DINOv2) — Stage B assets exist.
- ⬜ **Clean re-runs**: BlockGCN + GNN-Transformer SI (polluted / non-converged).

## What's needed to submit
1. **Complete the 11-arch SI table** (multi-seed sweep) + random-split counterparts.
2. **LOSO** on the headline archs (signer-fold variance + significance).
3. **Feature isolation** run (pose vs DINOv2).
4. **Licensing (P0)**: email BdSLW401 authors (CC BY-NC-ND) → enables pose/weight
   release for the D&B route; release pipeline+splits+hashes regardless.
5. **D&B extras**: retrieval task framing, per-signer fairness, calibration, saliency.

## Venue
- **NeurIPS D&B / "Evaluations & Datasets"** (audits/stress-tests welcome, no SOTA
  required) — needs hosted artifacts + Croissant metadata (hard requirement).
- **WACV / BMVC** — historical home of ISLR benchmarks.

## Why this is the safe harbor
It needs **no positive method result** — the shortcut *is* the finding, the
monolingual-SSL intervention is a bonus, and the benchmark + protocol are the
lasting artifact. It de-risks the entire program: even if P1's cross-lingual claim
dies, P2 stands on rigor.
