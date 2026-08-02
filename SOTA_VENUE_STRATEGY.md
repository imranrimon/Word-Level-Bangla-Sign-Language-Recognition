# SOTA & Venue Strategy — Deep-Research Synthesis (2026-07-15)

Deep-research report answering: **how to upgrade this program to current SOTA
standards, where to submit, and what exactly is novel/unique/impactful.**
Produced by a fan-out web-research pass over five angles (SOTA landscape,
venue precedents, shortcut-paper framing, Bangla/cross-lingual prior work,
licensing/artifacts), with primary-source quote extraction and adversarial
verification. Verification legend:

- ✅ = claim survived a 2–3-vote adversarial verification panel against the primary source
- 📄 = direct quote extracted from the primary source (panel incomplete — session limit), treat as high-confidence but re-check before camera-ready

---

## 0. Executive summary

1. **The one top-tier methodological claim still open to us is: *cross-lingual
   masked pose SSL transfers to a low-resource sign language*.** SHuBERT's
   authors explicitly scope their work to ASL and state they "do not know how
   well our model would generalize to other languages" 📄; SignCLIP pretrains
   multilingually but defers all non-ASL evaluation to future work 📄.
   Meanwhile *supervised* cross-lingual transfer is already published
   (Uni-Sign CSL→ASL at ICLR 2025 📄, Logos RSL→WLASL/AUTSL at EMNLP 2025 📄,
   SlovoExt→WLASL 📄, cross-lingual few-shot SLR in Pattern Recognition 2024 📄).
   So the claim must be scoped precisely: novel **mechanism** (self-supervised
   masked pose prediction, not supervised classification) × novel **setting**
   (low-resource target, Bangla) — and it gets dramatically stronger with a
   second target language.
2. **The identity-shortcut finding is motivation, not a paper.** AUTSL
   quantified the same gap in 2020 (95.95% random-split → 62.02%
   signer-independent, ~34 pp) and it landed in IEEE Access, folded into a
   dataset release 📄. ASL Citizen made signer-disjoint splits the built-in
   protocol of a NeurIPS D&B 2023 benchmark 📄. Top-venue shortcut papers all
   pair diagnosis with **quantification metric + intervention + released
   benchmark** (RESOUND @ ECCV 2018 📄, scene-bias mitigation @ NeurIPS 2019 📄,
   COVID shortcut localization @ Nature MI 2021 📄). Our main paper already has
   this shape — keep the benchmark + three interventions as the headline, use
   the 22.46 pp gap as the hook.
3. **Two of our "SOTA baselines" claims need repair before any submission**:
   NLA-SLR's headline numbers are RGB+pose fusion (its keypoint-only ablation
   is 49.10% on WLASL-2000, ~12 pp below headline) ✅ — compare pose-only
   against pose-only; and RQE originates with the BdSLW60/BdSLW401 authors 📄 —
   our `block_gcn_rqe` is a re-implementation to cite, not a contribution.
4. **Factual corrections found**: BdSLW401 is officially **401 signs,
   102,176 samples, 18 signers, front+lateral views** 📄 (verified locally:
   our `data/bdslw401_si` bundle = 51,098 **front-view** clips, signers 1–18,
   test signers {4, 8}). The repo docs' "25 signers" is wrong. The 99.41%
   figure we debunk is from follow-up/README-lineage work — the original
   BdSLW60 paper reports only 67.6% (SVM) / 75.1% (Bi-LSTM) 📄, so our 76.95%
   SI result *already exceeds the dataset authors' signer-dependent baselines*
   — a much stronger framing.
5. **Licensing is a real constraint on the SSL pool**: CC BY-NC-ND on
   BdSLW401 means extracted pose data is a non-shareable derivative, and
   model weights are a grey area to treat conservatively 📄. Metrics are
   always publishable 📄. Fix: request written permission from the authors
   now, keep an ND-free "releasable" pool variant, release pipeline +
   manifests + splits instead of arrays (YouTube-ASL precedent at NeurIPS
   D&B 📄).
6. **Venue reality (as of 2026-07-15)**: NeurIPS 2026 "Evaluations & Datasets"
   deadline passed (May 4/6, 2026) ✅→📄 — that track explicitly welcomes
   audits/stress-tests of prior evaluations and does *not* require SOTA 📄,
   making NeurIPS 2027 E&D the natural home for the benchmark paper if
   licensing is resolved. Near-term: WACV 2027 round 1 (~now/July),
   AAAI 2027 (~Aug), ICLR 2027 (~Sep), CVPR 2027 (~Nov), ARR→ACL/NAACL 2027
   cycles (verify each CFP — dates below are from historical patterns).

---

## 1. The 2023–2026 ISLR SOTA landscape

### 1.1 WLASL-2000 per-instance Top-1 anchor table

| Method | Input | Venue | WLASL-2000 P-I Top-1 | Status |
|---|---|---|---|---|
| SignBERT+ | pose-only SSL | TPAMI 2023 | 48.85 | 📄 (Uni-Sign Table 3) |
| BEST | pose-only SSL | AAAI 2023 | 46.25 (54.59 w/ RGB fusion) | 📄 |
| MASA | pose-only SSL | TCSVT 2024 | 49.06 | 📄 |
| NLA-SLR keypoint-only ablation | pose-only | CVPR 2023 | **49.10** | ✅ |
| SAM-SLR-v2 | multi-modal ensemble | — | 59.39 | ✅ |
| NLA-SLR (headline) | **RGB+63-kp heatmaps**, S3D, K400-pretrained | CVPR 2023 | 61.05 / 61.26 (3-crop) | ✅ |
| SHuBERT | 4-stream (DINOv2 crops + pose) SSL, 1,000 h ASL | ACL 2025 oral | 60.90 | 📄 |
| MViTv2-S + recipe + SlovoExt pretrain | RGB-only | arXiv 2412.11553 | 62.89 | 📄 |
| Uni-Sign (pose-only) | 69-kp GCN + mT5, CSL-News 1,985 h | ICLR 2025 | **63.13** | 📄 |
| Uni-Sign (RGB+pose) | fusion | ICLR 2025 | 63.52 | 📄 |
| Logos (RGB single-stream, RSL pretrain) | RGB | EMNLP 2025 | ~66.8 (reported) | 📄 unverified |

Key reading of this table for us:

- **Pose-only SOTA ≈ 63% (Uni-Sign), reached only via 1,985 h of pretraining**
  📄. Without web-scale data, the pose-SSL family (SignBERT+/BEST/MASA) sits
  at 46–49%. Our SSL pool (~54–76 k clips) is *comparable to or larger than
  MASA's pool* — MASA pretrained only on the downstream benchmarks' train
  sets 📄 — so "small-pool pose SSL" has a published journal-tier precedent;
  the cross-lingual composition is what's new.
- **Never compare our pose-only numbers against fusion headlines.** NLA-SLR
  keypoint-only = 49.10 ✅ is the honest CVPR-2023 pose bar.
- **Training recipe is a confound worth ±4–10 pp** 📄 (arXiv 2412.11553:
  +6.54 WLASL / +3.93 AUTSL / +10.12 Slovo from recipe alone, backbone
  fixed). Our 11-architecture SI benchmark must state the recipe-control
  policy explicitly (shared schedule/augs + per-model LR sweep) or reviewers
  can dismiss the ranking.
- Scale bar for SL pretraining corpora in 2025–26: YouTube-ASL 984 h,
  CSL-News 1,985 h (~751 k clips) ✅, YouTube-SL-25 3,207 h 📄. Position our
  pool as **low-resource + pose-only + cross-lingual**, never as scale.

### 1.2 Mandatory-citation set for any 2026 pose-based ISLR paper

NLA-SLR (CVPR 23), SAM-SLR-v2, SignBERT (ICCV 21) / SignBERT+ (TPAMI 23),
BEST (AAAI 23), MASA (TCSVT 24), SHuBERT (ACL 25), Uni-Sign (ICLR 25),
SignCLIP (EMNLP 24), Logos (EMNLP 25), ASL Citizen (NeurIPS D&B 23),
MM-WLAuslan (NeurIPS D&B 24), AUTSL (IEEE Access 20), WLASL (WACV 20),
MS-ASL (BMVC 19), training-strategies (arXiv 2412.11553), plus framing
citations: Geirhos et al. shortcut learning (Nature MI 20), RESOUND
(ECCV 18), scene-bias mitigation (NeurIPS 19), DeGrave et al. COVID
shortcuts (Nature MI 21), Desai et al. Deaf-led critique (LREC-COLING 24
sign-language workshop), Koller deep-hand (CVPR 16), BlockGCN (CVPR 24),
CTR-GCN (ICCV 21), ST-GCN (AAAI 18), and the BdSLW60 (MTAP 2025) /
BdSLW401 (arXiv 2503.02360) dataset papers.

### 1.3 Mandatory experimental baselines on BdSLW60-SI (gap to close)

| # | Baseline | Why reviewers demand it | Est. cost |
|---|---|---|---|
| B1 | **A released sign foundation model adapted to BdSLW60-SI** — Uni-Sign checkpoint (pose; code+weights public 📄) fine-tuned, and/or SHuBERT frozen features + linear head (SHuBERT shows frozen ≈ fine-tuned, flagged by its authors as "promising for low-resource sign languages" 📄) | Positions us against 2025 SOTA directly; doubles as *the* cross-lingual transfer comparison (CSL→BdSL, ASL→BdSL) | 1–3 GPU-days |
| B2 | **One strong RGB baseline** (I3D minimum; VideoMAE-S/MViTv2-S better) | ASL Citizen's NeurIPS D&B baselines were exactly I3D (63.10) vs ST-GCN (59.52) 📄 — an SI benchmark without an RGB row looks incomplete, and RGB-only SOTA papers explicitly attack pose-only pipelines 📄 | 2–4 GPU-days |
| B3 | **BPT vs joint co-training ablation** — Logos found multi-head co-training beats sequential pretrain→finetune for low-resource targets 📄 | Preempts the obvious "did you try co-training?" review; if co-training wins, adopt it and cite Logos | 1–2 GPU-days |
| B4 | **Pose-SSL family comparison** — at minimum discuss SignBERT+/BEST/MASA numbers and ablate our SSL vs supervised-only at matched recipe; ideally run one public implementation on BdSLW60-SI | Our SSL contribution is read against this trio 📄 | 2–5 GPU-days (or 0 if positioned by ablation only) |

---

## 2. Novelty map — what's taken vs open

### Taken (do not claim; cite and differentiate)

- Signer-independent protocol per se — AUTSL 2020 📄, ASL Citizen 2023 📄,
  MSLR/SignEval 2025 challenge institutionalizes unseen-signer eval 📄.
- Gap quantification per se — AUTSL published a ~34 pp random-vs-SI gap in
  2020 📄.
- Supervised cross-lingual transfer — Uni-Sign (CSL→ASL) 📄, Logos
  (RSL→WLASL/AUTSL "universal encoder") 📄, SlovoExt→WLASL 📄, few-shot
  cross-lingual SLR (Pattern Recognition 2024, ASL/DGS/TID) 📄.
- Multilingual sign pretraining — SignCLIP (44 languages, contrastive) 📄.
- Masked pose SSL per se — SignBERT+/BEST/MASA/SHuBERT lineage.
- RQE — introduced by the BdSLW60/BdSLW401 authors 📄; theirs to cite. Note
  they report WER, not Top-1 📄 — make our comparison metric-compatible. They
  also concede RQE fails to scale to WLASL-2000 📄.

### Open and defensible (our claims, sharpest first)

1. **Cross-lingual masked pose SSL → low-resource target.** SHuBERT: "the
   current scope of our work is limited to American Sign Language … we do
   not know how well our model would generalize" 📄. SignCLIP: non-ASL eval
   deferred to future work 📄. Nobody has published masked-pose-SSL
   cross-lingual transfer, and nobody has any SSL result on Bangla. The
   BdSL-only vs BdSL+ASL pool ablation (T5) is exactly the right experiment;
   with a second target language (AUTSL and/or LSA64/INCLUDE) it becomes an
   ICLR/ACL-tier claim (`OPTION_B_CROSS_LINGUAL_PAPER_SCOPE.md`'s 12-cell
   matrix is the full version).
2. **First rigorous SI benchmark + identity-shortcut quantification for
   Bangla SLR.** The BdSLW401 authors themselves name signer variability as
   the core challenge yet publish no SI protocol or shortcut number 📄. All
   published Bangla SLR sits at mid-tier venues (BdSLW60 @ Multimedia Tools
   & Applications 2025 📄; BdSLW401 arXiv-only 📄) and Bangla is absent from
   both major curated SLR literature lists 📄 — the field is genuinely
   unclaimed at top venues.
3. **Feature-level shortcut localization** (MediaPipe vs DINOv2 vs
   Bangla-DINOv2 under one architecture). No SLR analog exists; the template
   is DeGrave et al.'s localize-the-shortcut XAI analysis (Nature MI 2021) 📄.
   DeGrave also warns a held-out split alone can't prove shortcut removal 📄
   — our feature-isolation experiment is precisely the extra evidence
   reviewers of shortcut claims now expect. Bonus: SHuBERT itself uses
   DINOv2 hand/face-crop features + tiny body-pose vector 📄, independently
   validating our Option-B input design.
4. **First Bangla handshape foundation encoder (195 k images, LoRA-DINOv2) +
   deep-hand-style KD into a skeleton model.** Koller's deep-hand is CVPR
   2016; the modernization (foundation teacher → GCN student, low-resource
   language) has no published analog.
5. **Cross-domain Bangla transfer matrix + BPT.** ASL Citizen's headline
   analysis was exactly a cross-dataset transfer comparison 📄, so the matrix
   is a recognized D&B-grade analysis. BPT itself is standard transfer
   learning — its value is the *measured recipe on released checkpoints*,
   strengthened by the B3 co-training comparison.
6. **First sentence-level BdSLW102_A baseline + word→sentence transfer.**
   Workshop-tier as planned; MSLR/SignEval's pose-only continuous track
   (86-kp, 18 signers, ~14 k clips 📄) shows this scale is challenge-grade.

### Impact framing (why the community should care)

- **Equity/reach**: Bangla is a top-10 spoken language; its sign community
  has no trustworthy recognition benchmark — published numbers are
  signer-memorization artifacts (our central measurement).
- **Field norms**: Geirhos et al. call for o.o.d. tests to "become the rule
  rather than the exception" 📄; MSLR 2025's CFP explicitly solicits
  signer-independent systems, SSL, and cross-lingual benchmarks 📄; Desai et
  al.'s Deaf-led review documents systemic evaluation flaws across 101
  sign-AI papers 📄. Our program is the concrete remedy instance for one
  language, packaged as reusable protocol + artifacts.
- **Method transfer**: the cross-lingual pose-SSL recipe, if it works, is
  *the* practical recipe for the ~150+ other low-resource sign languages 📄
  (annotation is expert-dependent; pose is cheap and privacy-friendlier).
- **Deployment honesty**: Koller (MSLR 2025 keynote): zero-shot, no-finetune
  evaluation is "the true benchmark for real-world deployment" 📄 — our
  eval-only BdSL60-SingleTrial set is exactly that and should be promoted in
  the writing, not buried.

---

## 3. Venue strategy

### 3.1 Where SLR work actually lands (2023–2026 precedents)

- **CVPR/ICCV main**: method papers with multi-benchmark SOTA (NLA-SLR
  CVPR 23 ✅, VSNet CVPR 25 📄, cross-view ISLR ICCV 25 📄). Bar: SOTA on
  WLASL/MSASL-class benchmarks — not reachable with BdSL-only experiments.
- **ICLR/ACL/EMNLP**: pretraining/representation papers — Uni-Sign ICLR 25 📄,
  SHuBERT ACL 25 oral 📄, SignCLIP EMNLP 24 📄, Logos EMNLP 25 📄. **This is
  where the cross-lingual SSL claim belongs** (NLP venues are notably
  receptive to sign-language work framed as language technology for a
  low-resource language).
- **NeurIPS D&B / "Evaluations & Datasets"**: dataset+benchmark releases —
  ASL Citizen 23 📄, PopSign 23 📄, MM-WLAuslan 24 📄. The 2026 CFP explicitly
  welcomes "rigorous reproduction, auditing, and stress-testing of prior
  evaluations" and does not require beating SOTA 📄. Hard requirements:
  hosted accessible artifacts + Croissant metadata; non-compliance =
  desk-reject 📄. NeurIPS 2026 deadline passed (May 4/6, 2026) 📄 → target
  the 2027 cycle (~May 2027).
- **WACV/BMVC**: the historical home of ISLR datasets/benchmarks (WLASL @
  WACV 20, MS-ASL @ BMVC 19, background-robustness benchmark @ BMVC 22) 📄.
  Solid Tier-1.5 fit for the cross-domain paper.
- **Workshops**: MSLR @ ICCV 25 (archival ICCV proceedings, double-blind,
  topics = exactly this program 📄; watch for a 2026/2027 edition), SLRTP @
  CVPR, LREC/ACL sign-language workshops (where critique/position papers
  land 📄).
- **Journals**: TCSVT (MASA), Pattern Recognition (cross-lingual few-shot),
  TPAMI (SignBERT+) — realistic for extended versions; PR-tier work gets
  modest visibility (the 2024 cross-lingual paper had single-digit citations
  by mid-2026 📄).

### 3.2 Recommended mapping (Tier-1.5 default, conditional upgrades)

| Paper | Default target | Upgrade condition → upgraded target |
|---|---|---|
| **Main** (SI benchmark + shortcut + 3 interventions) | WACV 2027 (R1 ~July, R2 ~Aug/Sep 2026 — **verify CFP now**) or BMVC 2027 | If T5 cross-lingual SSL gain ≥ 3 pp **and** a 2nd target language added → split the SSL result into its own ICLR 2027 (~Sep 2026) / ARR→ACL 2027 paper; benchmark half → NeurIPS 2027 E&D (needs licensing resolved + hosting + Croissant) |
| **Paper 2** (cross-domain + BPT) | WACV 2027 / AAAI 2027 (abstract ~Aug 2026) | If transfer matrix + released harness are strong and artifacts releasable → NeurIPS 2027 E&D |
| **Paper 3** (sentence) | MSLR/SLRTP-class workshop (archival) | — |

Deadline dates other than NeurIPS 2026 are historical-pattern estimates —
verify every CFP; my knowledge cutoff is Jan 2026.

### 3.3 Framing rules distilled from precedents

1. Lead with the **benchmark + interventions**, hook with the gap number
   (RESOUND/Choi pattern: quantify → intervene → release) 📄.
2. Express the shortcut DeGrave-style: internal vs o.o.d. numbers, then
   **localize** it (feature isolation), then show interventions shrink it 📄.
3. Frame vs the original authors: "we exceed the dataset authors' published
   baselines (75.1%) under a *stricter* protocol (76.95% SI)" — not merely
   "we debunk 99.41%".
4. Adopt ASL Citizen's extras where cheap: a use-case task framing
   (dictionary retrieval for BdSL), retrieval metrics (R@1/5/10, MRR, DCG)
   alongside Top-1/Top-5 📄.
5. State the recipe-control policy for the 11-architecture table 📄.

---

## 4. Licensing (BdSLW401, CC BY-NC-ND 4.0) — action items

Findings (CC official guidance + practitioner/legal analyses 📄):

- BY/SA/ND conditions trigger **only on public sharing** — private training
  is not itself a violation under the mainstream reading 📄 (CC's strictest
  reading would bar even training 📄; jurisdiction TDM exceptions may apply 📄).
- **Extracted pose data = derivative → cannot be released** without
  permission 📄. **Weights = unsettled grey area → assume derivative**,
  keep research-only/permission-gated 📄.
- **Metrics/numbers/analysis are always publishable** with attribution 📄.
- **Do not mix ND-derived pretraining into checkpoints intended for
  release** — taint concern; keep pools segregated 📄.
- NC clause: all uses must stay non-commercial — fine for this program 📄.

Actions:

1. **Email the BdSLW401 authors now** requesting written permission to
   release (a) extracted pose arrays, (b) pretrained backbone weights, for
   non-commercial research. This single email potentially unlocks the
   NeurIPS E&D route.
2. Maintain **two SSL pool variants**: (i) full pool (max accuracy;
   weights research-only unless permission granted), (ii) ND-free pool
   (BdSLW60-derived + ASL corpora — verify each ASL corpus license has no ND
   clause) whose checkpoint can be released.
3. Regardless of permission, release: pose-extraction pipeline, manifests,
   canonical split files, eval harness, and result hashes — the
   YouTube-ASL "pointers + tooling" precedent was accepted at NeurIPS D&B 📄.
4. Add a licensing/ethics paragraph citing CC's official AI-training
   guidance 📄 — reviewers increasingly check this.

---

## 5. Prioritized upgrade path (reviewer impact per GPU-day)

| Priority | Upgrade | Cost | Why |
|---|---|---|---|
| P0 | Citation/factual hygiene: BdSLW401 = 18 signers/102,176 (front+lateral; our 51,098 = front-view), RQE attribution, 99.41% provenance, NLA-SLR pose-only ablation for comparisons | 0 GPU | Any one of these caught by a reviewer sinks trust in everything else |
| P0 | License-permission email + pool segregation plan | 0 GPU | Gates NeurIPS E&D; takes a day |
| P1 | B1: Uni-Sign checkpoint finetune + SHuBERT frozen-feature probe on BdSLW60-SI | 1–3 d | Single highest-impact addition: SOTA anchoring + cross-lingual baseline in one |
| P1 | B2: RGB baseline (I3D / VideoMAE-S) on BdSLW60-SI | 2–4 d | ASL-Citizen norm; justifies pose-only choice with a number |
| P1 | B3: co-training vs BPT ablation | 1–2 d | Logos finding directly challenges BPT; cheap to preempt |
| P2 | Recipe control statement + per-model LR sweep for the 11-arch table | 2–5 d | ±6.5 pp recipe confound is a known review attack 📄 |
| P2 | Zero-GPU analyses: per-signer fairness breakdown, calibration, per-class heatmap, saliency localization of identity cues, retrieval metrics | ~0 GPU | D&B-grade analysis depth; DeGrave-style evidence |
| P3 | Second SSL target language (AUTSL, +LSA64/INCLUDE) | 1–2 wk | Converts the SSL result from "Bangla case study" to ICLR/ACL-tier general claim; gate on T5 ≥ 3 pp per existing decision |
| P3 | Dictionary-retrieval task framing for the benchmark release | ~0 GPU | ASL-Citizen-style use-case grounding for E&D |

---

## 6. Source list (primary sources consulted)

- NLA-SLR — https://arxiv.org/abs/2303.12080 (CVPR 2023) ✅
- Uni-Sign — https://arxiv.org/abs/2501.15187 (ICLR 2025); code github.com/ZechengLi19/Uni-Sign 📄
- SHuBERT — https://arxiv.org/abs/2411.16765 / https://aclanthology.org/2025.acl-long.1397/ (ACL 2025 oral) 📄
- SignCLIP — https://aclanthology.org/2024.emnlp-main.518/ (EMNLP 2024) 📄
- MASA — https://arxiv.org/pdf/2405.20666 (TCSVT 2024) 📄
- Training strategies for ISLR — https://arxiv.org/html/2412.11553v1 📄
- Logos — https://arxiv.org/abs/2505.10481 (EMNLP 2025) 📄
- Cross-lingual few-shot SLR — Pattern Recognition 151 (2024), DOI 10.1016/j.patcog.2024.110374 📄
- ASL Citizen — NeurIPS D&B 2023, https://openreview.net/forum?id=zbEYTg2F1U 📄
- YouTube-ASL — NeurIPS D&B 2023 proceedings 📄
- AUTSL — IEEE Access 8 (2020), https://ieeexplore.ieee.org/document/9210578/ 📄
- MSLR 2025 workshop @ ICCV — https://multimodal-sign-language-recognition.github.io/ICCV-2025/ 📄
- NeurIPS 2026 Evaluations & Datasets CFP — https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets 📄
- Geirhos et al., Shortcut learning — Nature MI 2 (2020) 📄
- DeGrave et al., COVID shortcuts — Nature MI 3 (2021) 📄
- RESOUND / Diving48 — ECCV 2018 📄
- Choi et al., scene-bias mitigation — NeurIPS 2019 📄
- Desai et al., Deaf-led critique — https://aclanthology.org/2024.signlang-1.6/ 📄
- BdSLW60 — https://arxiv.org/abs/2402.08635 (MTAP 2025) 📄
- BdSLW401 — https://arxiv.org/abs/2503.02360 (arXiv-only) 📄
- CC AI-training guidance — https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/ 📄
- Curated lists: github.com/ZechengLi19/Awesome-Sign-Language, github.com/VIPL-SLP/awesome-sign-language-processing 📄

**Caveat**: the adversarial-verification phase was cut short by a session
usage limit — 5 verification votes completed (all confirmed: NLA-SLR numbers
and architecture, SAM-SLR-v2 number, CSL-News scale); the remaining claims
are direct primary-source quote extractions (📄). Re-verify any 📄 number you
put in a camera-ready table against the cited PDF.
