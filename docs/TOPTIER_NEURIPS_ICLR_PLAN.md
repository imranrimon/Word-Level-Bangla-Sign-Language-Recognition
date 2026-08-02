# Top-tier (NeurIPS / ICLR) execution plan

Target chosen 2026-07-29: **NeurIPS / ICLR** (representation-learning framing).
This plan maps every top-tier requirement to a concrete run/task and states the
one decision gate the whole framing hinges on. Companion docs:
[`SOTA_VENUE_STRATEGY.md`](../SOTA_VENUE_STRATEGY.md) (venue map, novelty),
[`RECIPE_CONTROL.md`](RECIPE_CONTROL.md), [`B1_WSL2_RUNBOOK.md`](B1_WSL2_RUNBOOK.md).

## Thesis
**Cross-lingual masked pose SSL** — pretrain a pose backbone on ASL+BdSL pose,
transfer to low-resource target sign languages — is the open, unclaimed
contribution. The **signer-independent (SI) benchmark** (and the 22.46 pp
identity-shortcut finding) is the rigorous evaluation foundation beneath it.
Headline model: SHuBERT-style masked-pose-pretrained BlockGCN, fine-tuned SI.

## THE decision gate (everything branches here)
> **Does the BdSL+ASL (cross-lingual) SSL backbone beat the BdSL-only
> (monolingual) backbone by ≥ 3 pp on SI test?**  (Option C, running now — both
> backbones already pretrained; only the 6 fine-tunes remain.)
> - **YES →** commit NeurIPS/ICLR; the method is the story; B1 + LOSO + 2nd
>   language make it bullet-proof.
> - **NO / marginal →** the benchmark + interventions are a clean WACV/BMVC
>   paper; do not force top-tier on a null method result.

## Requirements → concrete runs (status as of 2026-07-29)

| # | Requirement (why top-tier needs it) | Concrete run / artifact | Status |
|---|---|---|---|
| R1 | **SI baseline table** (foundation; the shortcut finding) | `experiments_si_main.yaml` 11 archs ×3 seeds | ▶ running (Phase 1) |
| R2 | **Feature-isolation** pose vs DINOv2 (localizes the shortcut) | `bdsl_pose_temporal_si` + `bdsl_dino_temporal_si` (A11/A12); Stage B.1 done | ▶ A11 running; A12 queued |
| R3 | **SLGTFormer ablations** (component credit) | `experiments_si_ablations.yaml` ×3 | queued (Phase 2) |
| R4 | **RGB baseline** B2 (modality anchor) | `path4_rgb_baseline` S3D + I3D | queued (Phase 4) |
| R5 | **Co-training vs sequential** B3 (design ablation) | `main_cotrain.py` ×3 | queued (Phase 5) |
| R6 | **THE method:** cross-lingual mechanism (impl. 2026-08-01) vs pool-only vs monolingual | pretrain `config/bdsl_shubert_pretrain_xlingual.yaml` → finetune ×3; ablations = `bdsl_asl` (pool-only) + `bdsl_only` (monolingual) | ✅ mechanism coded; re-pretrain pending (HPC) |
| R7 | **Foundation-SOTA baselines** B1 (mandatory) | Uni-Sign + SHuBERT via WSL2 | ⬜ needs user — [runbook](B1_WSL2_RUNBOOK.md) |
| R8 | **Headline variance = LOSO** (signer noise, not seed) | `SLGT_LOSO`: 11-fold BlockGCN | ⏸ parked → auto after Option C |
| R9 | **2nd target language** (N=1→2 cross-lingual) | **LSA64** (fast win) + **AUTSL** (rigorous SI); KArSL optional 3rd | ✅ scoped — see §R9 |
| R10 | **Significance** (paired bootstrap) | `tools/paired_bootstrap.py` on final rows | at write-up |
| R11 | **Benchmark artifacts / licensing** | Croissant metadata; BdSLW401 ND license blocks pose/weight release — resolve before D&B claim | ⬜ P0 (zero-GPU) |

R1–R6 + R8 run unattended on the RTX 8000 in sequence (kill-immune detached
tasks). **R7 (B1) and R9 (2nd language) are the two things that need you.**

## Sequencing (single GPU, auto-chained)
```
SI program (R1→R2→R3→R4→R5)  ──full_si_done──▶  Option C SSL (R6)  ──optionc_done──▶  LOSO (R8)
Stage B.1 (R2 features) ✅ done          B1 (R7, WSL2) — parallel on user's schedule, after GPU frees
2nd language (R9) — add as 3rd target once selected (extends R6 + R8)
```

## What we claim (contribution list for the paper)
1. First rigorous **SI benchmark** for word-level BdSL; quantifies + localizes a
   22.46 pp identity shortcut across 11 architectures.
2. **Cross-lingual masked pose SSL**: ASL+BdSL pretraining transfers to
   low-resource targets, beating monolingual SSL and from-scratch by ≥3 pp (gate).
3. Generalization across **≥2 target languages** (R9) and **signer-disjoint LOSO**
   variance (R8) — robustness, not a single lucky split.
4. Honest comparison to supervised (Uni-Sign) and SSL (SHuBERT) foundation
   transfer (R7).

## R9 — second target language (scoped 2026-07-29)
Add a 2nd (ideally 3rd) low-resource sign language as a fine-tuning TARGET to lift
the cross-lingual claim from N=1 to N=2. All are MediaPipe-extractable (our 27-kpt
skeleton abstracts away gloves/Kinect-RGB), and re-extraction is required anyway.

| Rank | Dataset (language) | Signs / signers / clips | SI split | License | Why |
|---|---|---|---|---|---|
| 1 | **LSA64** (Argentine) | 64 / 10 / 3,200 | build from signer IDs | CC BY-NC-SA 4.0 | **Fast win** — 1.5 GB, pure RGB, drops straight in |
| 2 | **AUTSL** (Turkish) | 226 / 43 / ~36k | **official SI split** (7 test signers) | research/non-commercial | **Credible** — reviewer-trusted SI protocol, 43 signers |
| 3 | **KArSL** (Arabic) | 502 / 3 / 75k | canonical train-2/test-1 | email-gated | **Max contrast** — most distant language; 3-signer SI is thin |

**Plan:** pair **LSA64 + AUTSL** for N=2 (a genuinely low-resource target + a trusted
SI benchmark); hold **KArSL** as a "maximally-distant-language" robustness add.
**Avoid INCLUDE (Indian)** as primary — weak signer metadata undercuts the SI story,
and ISL shares regional influence with BdSL (least cross-lingual contrast).
Each target extends R6 (fine-tune the same ASL+BdSL backbone) and R8 (its own SI/LOSO).

## Top risks
- **Gate fails** (SSL gain marginal): fall back to WACV/BMVC benchmark paper. — *mitigate: read Option C result before committing write-up.*
- **B1 environment** (deepspeed/WSL2 friction): — *mitigate: [runbook](B1_WSL2_RUNBOOK.md); SHuBERT probe is the lighter half if Uni-Sign stalls.*
- **BdSLW401 ND license** blocks releasing pretrained pose/weights → weakens a
  D&B artifact claim. — *mitigate: email authors for permission (P0); ICLR/NeurIPS main-track method claim does not require weight release, only the benchmark-artifact framing does.*
- **LOSO cost** (~11 GPU-days at 1 seed): — *mitigate: wired at 1 seed/fold; expand to 3 seeds only if a reviewer asks.*

## Open user decisions
- Confirm 2nd-language target once scoping returns (R9).
- Schedule B1/WSL2 (R7) — after the GPU sweep, or on a 2nd GPU if available.
