# Deep-Research Synthesis — 2026-08-02

Fresh fan-out web-research pass (2024–2026 sources, adversarial 3-vote
verification) updating [`../SOTA_VENUE_STRATEGY.md`](../SOTA_VENUE_STRATEGY.md)
(2026-07-15) and [`TOPTIER_NEURIPS_ICLR_PLAN.md`](TOPTIER_NEURIPS_ICLR_PLAN.md).
Run IDs: `wf_237d827e-456` (SOTA + thesis validation) and `wf_0436ed18-4be`
(focused follow-up on Bangla novelty / handshape-KD / generative — **pending at
time of writing**). Verification legend:

- ✅ survived a 3-vote adversarial verification panel against the primary source
- ⟐ author synthesis / proposal (not a verified external claim)
- ⚠️ flagged unverified or anomalous — re-check before citing

---

## 0. Executive summary

1. **The thesis is still open, but the moat narrowed sharply in late 2025.**
   "Cross-lingual **masked** pose SSL → low-resource target, **fine-tuned** &
   **signer-independent**" is unclaimed **only at a four-way intersection**:
   `masked/reconstructive (not contrastive)` × `cross-lingual multi-source
   pretrain` × `fine-tuned (not linear-probe)` × `formal SI target eval`.
   Weaken any one axis and a 2025 competitor exists. ✅
2. **The single most dangerous prior work is `SSL-SLR` (arXiv 2509.05188, Sept
   2025)** — a MediaPipe pose SSL framework that *explicitly* demonstrates
   cross-lingual transfer to low-resource targets (LSFB→GSL 54.78%, ASL→LSA
   43.84%). It is **contrastive** (free-negative-pairs, not masked) and uses
   **frozen linear evaluation** (not fine-tuned SI). Those two gaps are now the
   program's *entire* differentiation — it must be cited and beaten head-to-head. ✅
3. **Pose-only SOTA moved to `Sigma` (arXiv 2509.21223, Sept 2025), 64.54 P-I on
   WLASL2000** (semantic vision-text alignment, not masked), above Uni-Sign
   (63.13). ✅ These are the correct pose-only-vs-pose-only anchors.
4. **Pose-vs-fusion hygiene is load-bearing.** Do **not** quote as pose-only:
   NLA-SLR 61.05 (VKNet RGB+keypoint fusion) ✅, Logos 66.82 (RGB co-training) ✅,
   SHuBERT 60.90 (pose+DINOv2 hybrid) ✅. Uni-Sign's exact *pretraining modality*
   claim was **refuted** in verification — cite only its pose-only numbers. ⚠️

---

## 1. Current pose-only SOTA landscape (WLASL2000, P-I Top-1)

| Method | Modality | Venue | P-I | Notes |
|---|---|---|---|---|
| **Sigma** ✅ | pose-only | arXiv 9/2025 | **64.54** | semantic vision-text align (mT5); not masked; no SI/cross-lingual claim |
| **Uni-Sign** ✅ | pose-only | ICLR 2025 | 63.13 | strongest supervised anchor; fusion adds only +0.39 |
| SHuBERT ✅ | pose+DINOv2 **hybrid** | ACL 2025 oral | 60.90 | masked SSL but **ASL-only**, four-stream (not pure pose) |
| MSLU ✅ | pose SSL | — | 56.29 | |
| SignBERT+ ✅ | pose SSL | TPAMI 2023 | 48.85 | |
| BEST ✅ | pose SSL | AAAI 2023 | 46.25 | |
| MASA ✅ | pose SSL (masked motion-residual) | 2024/IEEE | — | closest *masked* pose method; monolingual, not SI |
| SSL-SLR ✅ | pose SSL (contrastive) | arXiv 9/2025 | — | **cross-lingual, but contrastive + linear-probe**; WLASL-100 77.95 / WLASL-300 71.21 |
| SignBart ⚠️ | pose-only | arXiv 6/2025 | 68.0 | 749k params; anomalously high — **verify before citing** |
| *NLA-SLR* | *RGB+kp fusion* | *CVPR 2023* | *61.05* | ❌ never quote as pose-only; keypoint-only ablation is the honest bar |
| *Logos* | *RGB co-train* | *EMNLP 2025* | *66.82* | ❌ RGB route; use as **B2 baseline** (co-train 66.82 vs separate 60.88, +5.94) |

## 2. Thesis validation — the defensible niche

**Closest competitors, each missing on one axis** (✅):

| Work | Masked? | Cross-lingual? | Fine-tuned SI? | Pure pose? |
|---|---|---|---|---|
| SHuBERT (ACL25) | ✅ | ✗ (ASL-only) | ✗ | ✗ (hybrid) |
| Uni-Sign (ICLR25) | ✗ (supervised) | partial | ✓ | ✅ |
| SSL-SLR (9/25) | ✗ (contrastive) | ✅ | ✗ (linear) | ✅ |
| MASA (2024) | ✅ | ✗ | ✗ | ✅ |
| **This program** | ✅ | ✅ | ✅ | ✅ |

**Mandatory baselines a reviewer will now demand:** Uni-Sign pose fine-tune ·
SHuBERT frozen-feature probe · **SSL-SLR's contrastive paradigm re-run
head-to-head vs. masked, matched pretrain+SI** (newly non-negotiable) · RGB
anchor (I3D/VideoMAE + Logos co-train) · co-training vs sequential.

**Time-sensitivity ✅:** SSL-SLR, Sigma, Logos all appeared Sept–Nov 2025 — a
preprint occupying the exact niche could surface before submission. **Re-run the
novelty check immediately before submitting.**

## 3. First-to-solve directions ⟐ (ranked; grounded in program assets)

| # | Direction | Risk | Why unclaimed | Program edge |
|---|---|---|---|---|
| **D1** | Masked-vs-contrastive cross-lingual pose SSL, fine-tuned + SI (head-to-head vs SSL-SLR) | Low | SSL-SLR is contrastive+linear-probe | SHuBERT-style masked pipeline already built |
| **D2** | **Signer-invariant pose pretraining** — adversarial/gradient-reversal on signer-ID *as an SSL objective*; success = shrink the 22.46 pp gap + LOSO variance | Med | Shortcut-removal never tied to a *pretraining objective* in sign | Measured shortcut + LOSO + signer metadata |
| **D3** | Handshape-foundation KD → skeleton (LoRA-DINOv2 Bangla teacher → pose GCN; Koller Deep-Hand modernized) | Med | No handshape-foundation→skeleton KD for a low-resource SL | 195k-crop handshape set + DINOv2 branch |
| **D4** | Pose-token sign-LM for cross-lingual retrieval / zero-shot (VQ/RQE pose units → mT5; dictionary-retrieval framing) | High | Sigma aligns but isn't cross-lingual/zero-shot/SI | **RQE already provides a pose quantizer** |
| **D5** | Diffusion pose-infilling as SSL + cross-signer augmentation engine | Moonshot | Generative-as-SSL + shortcut-attack unclaimed | Ties generative to the shortcut hook |

**D1 = safe headline. D2 = highest-value** — it converts the identity-shortcut
finding from *motivation* into a *method* (addresses the internal note that the
shortcut finding "is not a paper" on its own).

## 4. Caveats & open items (from the verified pass)

- **Refuted:** Uni-Sign as "monolingual CSL generative pretraining" (vote 1-2) —
  do not assert its exact pretraining modality; only its pose-only numbers.
- **Unverified:** SignBart 68.0 pose-only WLASL2000 (anomalous vs peers).
- **Not yet covered** (deferred to follow-up pass `wf_0436ed18-4be`): (a) whether
  Bangla is genuinely unclaimed at top venues + strongest existing BdSL result &
  protocol; (b) handshape-KD prior art (Koller lineage → 2026); (c)
  generative/diffusion/retrieval/pose-tokenization framings. **This section to be
  appended when that pass returns.**

## 5. Primary sources (verified pass)

- Sigma — https://arxiv.org/html/2509.21223 ✅
- SSL-SLR — https://arxiv.org/pdf/2509.05188 ✅
- Uni-Sign — https://arxiv.org/abs/2501.15187 / ICLR 2025 proceedings ✅
- SHuBERT — https://arxiv.org/abs/2411.16765 (ACL 2025 oral) ✅
- MASA — https://arxiv.org/abs/2405.20666 ✅
- NLA-SLR — https://arxiv.org/abs/2303.12080 (CVPR 2023) ✅
- Logos — https://arxiv.org/abs/2505.10481 (EMNLP 2025) ✅
- SignBart — https://arxiv.org/pdf/2506.21592 ⚠️
- BlockGCN — https://arxiv.org/abs/2305.11468 (CVPR 2024) ✅
- BdSLW60 — https://arxiv.org/abs/2402.08635 · BdSLW401 — https://arxiv.org/abs/2503.02360
- MSLR 2026 workshop — https://m-slrt.github.io/MSLR2026/

**Stats:** 5 angles · 22 sources fetched · 103 claims → 25 verified (24 confirmed,
1 killed) · 104 agents. Re-verify any single number against its primary PDF
before camera-ready.
