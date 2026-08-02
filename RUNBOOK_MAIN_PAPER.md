# RUNBOOK — Main Paper

**Working title**
*Identity-Shortcut Narrowing for Word-Level Bangla Sign Language Recognition:
A Signer-Independent Benchmark and Three Orthogonal Interventions.*

This runbook is the primary-author's execution guide for producing every
number, table, and figure in the main paper. It is scoped narrowly to what
the paper claims — the general-purpose operational runbook is `RUNBOOK.md`,
the Path-3 sister paper is `RUNBOOK_SISTER_PAPER.md`.

---

## 1. Paper claim (in one paragraph)

BdSLW60 accuracy numbers reported in the existing literature are
signer-memorization: the 99.41 % Top-1 on the repo's legacy random split
collapses to **76.95 %** under a canonical signer-disjoint split (single-seed
pilot), a **22.46 pp identity-shortcut gap**. We propose and measure three
*orthogonal* interventions that each narrow the gap:

1. **Input-representation change** — swap MediaPipe-27 pose for DINOv2 hand /
   face-crop features (Option B).
2. **Domain-adapted encoder** — LoRA-fine-tune DINOv2 on ~195 k aggregated
   Bangla handshape images (Path 1), the first Bangla-specific handshape
   foundation encoder.
3. **Handshape knowledge distillation into the skeleton model** — distil the
   Path-1 encoder's representation into BlockGCN during training (Path 2).

An additional track (Option C — SHuBERT-style masked SSL on ~56 k unlabeled
Bangla pose clips) is reported as a complementary intervention. Stage D
quantifies cross-recording-condition robustness on BdSL60-SingleTrial.

## 2. Positioning (what's genuinely new)

| Claim | Precedent | Delta |
|---|---|---|
| Signer-independent BdSLW60 benchmark at 3 seeds | none published | first |
| Identity-shortcut measured as SD-SI gap on Bangla SLR | general claim exists; no Bangla number | first |
| Bangla-specific DINOv2 adaptation | none | first |
| BlockGCN applied to sign language recognition | BlockGCN is from skeleton-action recognition (Zhou CVPR 2024) | first in SLR |
| SHuBERT-style SSL on Bangla pose | SHuBERT is ASL (Gueuwou 2024) | first in Bangla |
| Handshape KD into a skeleton SLR backbone | Koller "deep-hand" (CVPR 2016) used ResNet→CNN; we use DINOv2→GCN | modern teacher, modern student |

## 3. Target tables and figures

| # | Caption (provisional) | Source stage |
|---|---|---|
| **T1** | Signer-independent Top-1 / Top-5 on BdSLW60 for every model in the repo, 3 seeds, mean ± std with bootstrap 95 % CI. | Stage A |
| **T2** | Identity-shortcut gap per input representation: `Top1_SD − Top1_SI`. | Stage B + B.3 |
| **T3** | Effect of encoder domain: generic DINOv2 vs Bangla-DINOv2 under the same FlatTemporal architecture. | Path 1 |
| **T4** | Effect of handshape KD: BlockGCN vs BlockGCN+KD(generic) vs BlockGCN+KD(Bangla-DINOv2). | Path 2 |
| **T5** | Effect of SSL pretraining: BlockGCN from scratch vs SHuBERT-pretrained. | Option C |
| **T6** | Cross-recording robustness: Top-1 gap between BdSLW60-SI val and BdSL60-SingleTrial. | Stage D |
| **F1** | Learning curve of the BlockGCN pilot on signer-independent train/val. | Stage A, seed 0 (already exists in log file) |
| **F2** | Per-class Top-1 heatmap on the SI test set (60 × 60). | Stage A, a chosen checkpoint |

Total experimental budget for **all** tables: ~4–7 GPU-days on one RTX 8000.

## 4. Smoke verification (run before committing GPU days)

Each long command below is a separate workflow that takes hours to days.
Before launching the real run, execute the matching **smoke command** — each
takes 2–5 minutes and verifies imports, configs, paths, data alignment,
single-batch forward, gradient flow, and CSV writing. If smoke passes, the
full run is statistically very likely to succeed.

> **Convention**: every smoke command writes to a `*_smoke` work_dir so it
> never clobbers real results.

### 4.0.1 Phase 0 — repo + data sanity (10 s)

```bash
python -m pytest tests/ -q                         # expect 57 passed
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -c "
import numpy as np, os
for f in ['data/bdsl_si/train_data.npy','data/bdsl_si/val_data.npy',
          'data/ssl_pool_manifest.json','data/pretrain_kmeans_targets.npz',
          'data/bdsl60_singletrial_eval/eval_data.npy']:
    print(f, 'OK' if os.path.exists(f) else 'MISSING')
"
```

### 4.0.2 Stage A smoke — classification harness on real data (~2 min)

```bash
python main.py --config config/bdsl_block_gcn_si_smoke.yaml \
    --seed 0 -Experiment_name bdsl_block_gcn_si_smoke
```

Expects: training log shows non-NaN loss decreasing across the 1 epoch,
eval prints Top-1 (random ≈ 1.7 %, ≥ random is fine for the smoke).

### 4.0.3 Stage B.1 smoke — DINOv2 extractor pipeline (~2 min)

```bash
python preprocessing/extract_dinov2_features.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir /tmp/_dino_smoke --cache-dir /tmp/_dino_smoke_cache \
    --splits val --device cuda \
    --model vit_small_patch14_dinov2.lvd142m --batch-size 16 --dry-run
```

`--dry-run` confirms dataset scan, MediaPipe import, DINOv2 model download
path, and disk plan without running extraction.

### 4.0.4 Stage B.2 smoke — FlatTemporal harness (~2 min)

```bash
python main.py --config config/bdsl_pose_temporal_si.yaml \
    --seed 0 -Experiment_name bdsl_pose_temporal_si_smoke \
    --num_epoch 1
# (no smoke YAML needed — main.py honors --num_epoch override)
```

### 4.0.5 Path 1 smoke — Bangla-DINOv2 LoRA harness (~3-5 min)

```bash
python -m path1_bangla_dinov2.train \
    --config path1_bangla_dinov2/configs/train_lora_smoke.yaml --seed 0
```

Expects: source inventory printed, LoRA replacements counted (>0), 1 epoch
of training over two small sources, per-source val Top-1 printed,
checkpoint saved at `work_dir/bdino_lora_smoke/encoder_epoch1.pt`.

### 4.0.6 Path 2 smoke — handshape KD harness (~2 min)

Requires either `data/bdsl_si_dino/` (Option B output) or
`data/bdsl_si_bdino/` (Path 1 output) to exist for the teacher. To test
without that pre-requisite, comment out the kd_proj head temporarily and
run with `--num_epoch 1`. Otherwise:

```bash
# Quick path: edit configs/train_kd.yaml: set num_epoch=1, batch_size=4,
# and add `debug: True` under both train_feeder_args and test_feeder_args.
python -m path2_handshape_kd.train_kd \
    --config path2_handshape_kd/configs/train_kd.yaml --seed 0
# Or: copy-then-edit configs/train_kd.yaml -> configs/train_kd_smoke.yaml first.
```

### 4.0.7 Option C smoke — SHuBERT pretraining harness (~3 min)

```bash
python main_pretrain.py --config config/bdsl_shubert_pretrain_smoke.yaml --seed 0
```

Uses `max_clips: 64` and `num_epoch: 1`. Verifies: SSL pool manifest +
k-means targets load aligned, BlockGCN with `stride_between_stages=False`
forwards `(N, 120, 256)` features, mask + cross-entropy yields a finite
loss, optimizer step runs, backbone-only checkpoint saved.

### 4.0.8 Stage D smoke — cross-recording eval pipeline (~30 s)

After Stage A smoke produced a checkpoint:

```bash
python main.py --config config/bdsl_block_gcn_si.yaml --phase test \
    --weights work_dir/bdsl_block_gcn_si_smoke/bdsl_block_gcn_si_smoke_model_best.pt \
    --test-feeder-args "{'data_path':'data/bdsl60_singletrial_eval/eval_data.npy','label_path':'data/bdsl60_singletrial_eval/eval_label.pkl','window_size':120,'random_choose':False,'normalization':True}" \
    -Experiment_name bdsl_smoke_evalST
```

Confirms `--phase test`, `--weights` loading, `--test-feeder-args` JSON
parsing, and BdSL60-SingleTrial eval bundle alignment with the model's
60-class head.

---

## 5. Reproduction sequence (strictly ordered)

> Each numbered block is a single copy-paste. All outputs write to `./work_dir/`
> and `results_final.csv`. Run `python tools/summarize_seeds.py` at any point to
> see partial results. **All commands run from repo root with `bdsl_graph`
> activated.**

### 4.1 Prerequisites

```bash
conda activate bdsl_graph
cd F:\SLGTformer
python -m pytest tests/ -q                 # must report 57 passed
```

Verify data artefacts exist (see `RUNBOOK.md §3.2` for details). If they
don't, regenerate them before proceeding:

```bash
# BdSLW60 signer-independent NPYs (CPU, ~2.5-3 h)  — probably already done
python preprocessing/generate_signer_split_npy.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si --cache-dir data/bdsl_cache \
    --splits train val test pretrain
```

### 4.2 Stage A — the headline benchmark (T1)

```bash
python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 --skip-existing
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/T1_stage_A.md
```

**Duration**: ~2–4 GPU-days. Produces one row per (config, seed) in
`results_final.csv`; aggregator collapses to one row per config with
mean ± std + bootstrap 95 % CI. Feeds **T1**.

### 4.3 Option B.1 / B.2 — DINOv2 vs MediaPipe feature isolation (rows of T2)

```bash
# B.1: extract DINOv2 features on BdSLW60 SI (GPU, ~3-5 h)
python preprocessing/extract_dinov2_features.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si_dino --cache-dir data/bdsl_dino_cache \
    --splits train val test --device cuda \
    --model vit_small_patch14_dinov2.lvd142m --batch-size 64

# B.2: flat-temporal pose-baseline x 3 seeds
python tools/run_multiseed.py --single config/bdsl_pose_temporal_si.yaml --seeds 0 1 2

# B.2: flat-temporal DINOv2 x 3 seeds
python tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2
```

### 4.4 Option B.3 — signer-dependent counterpart of B.2 (rows of T2)

```bash
# Create _sd.yaml duplicates pointing at data/bdsl/  (the legacy random split)
cp config/bdsl_pose_temporal_si.yaml config/bdsl_pose_temporal_sd.yaml
cp config/bdsl_dino_temporal_si.yaml config/bdsl_dino_temporal_sd.yaml
# Hand-edit each _sd.yaml: data/bdsl_si/ -> data/bdsl/ ;
#                           data/bdsl_si_dino/ -> data/bdsl_dino/ ;
#                           Experiment_name suffix -> _sd
# For the DINOv2 SD track, also run extract_dinov2_features.py against
# data/bdsl/ (or generate generic-split DINOv2 features via any equivalent path).

python tools/run_multiseed.py --single config/bdsl_pose_temporal_sd.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_sd.yaml --seeds 0 1 2
```

**Compute T2**: for each representation R,
`gap(R) = mean(Top1_SD) − mean(Top1_SI)`. Hypothesis: `gap(DINOv2) < gap(pose)`.

### 4.5 Path 1 — Bangla-DINOv2 encoder (T3 rows)

```bash
# P1.1: LoRA-tune DINOv2 on aggregated Bangla handshape corpus (~3-6 h GPU)
python -m path1_bangla_dinov2.train \
    --config path1_bangla_dinov2/configs/train_lora.yaml --seed 0

# P1.2: apply the adapted encoder to BdSLW60 hand crops (~3-5 h GPU)
python -m path1_bangla_dinov2.extract_features \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si_bdino --cache-dir data/bdsl_bdino_cache \
    --encoder-checkpoint work_dir/bdino_lora/encoder_epoch10.pt \
    --splits train val test --device cuda --batch-size 64

# P1.3: FlatTemporal on Bangla-DINOv2 features x 3 seeds
cp config/bdsl_dino_temporal_si.yaml config/bdsl_bdino_temporal_si.yaml
# Edit: data/bdsl_si_dino/ -> data/bdsl_si_bdino/ ; Experiment_name -> _bdino_
python tools/run_multiseed.py --single config/bdsl_bdino_temporal_si.yaml --seeds 0 1 2
```

**T3 rows**: `bdsl_pose_temporal_si`, `bdsl_dino_temporal_si`,
`bdsl_bdino_temporal_si`.

### 4.6 Path 2 — handshape KD into BlockGCN (T4 rows)

```bash
# Teacher features are the P1.2 output (or fall back to data/bdsl_si_dino/ if
# you want the generic-DINOv2 KD ablation row).
for seed in 0 1 2; do
  python -m path2_handshape_kd.train_kd \
      --config path2_handshape_kd/configs/train_kd.yaml --seed $seed
done
```

For the generic-DINOv2 KD ablation, edit `teacher_data_path` in the config
to `data/bdsl_si_dino/...` and rerun with a different Experiment_name suffix.

### 4.7 Option C — SSL pretrain + fine-tune (T5 rows)

```bash
# C.1: (re)build SSL pool manifest
python preprocessing/build_ssl_pool_manifest.py \
    --cache-dirs data/bdsl_cache data/bdslw102_a_pose_cache data/bdslw401_pose_cache_front \
    --output data/ssl_pool_manifest.json

# C.2: k-means cluster targets (~30-60 min CPU)
# Default --feature-mode is now `pose_motion` (162-dim = pose + first-derivative)
# instead of bare per-frame pose (81-dim, audit fix #1). The previous bare-pose
# clustering let SHuBERT solve masked prediction trivially by copying the
# neighbour's cluster ID — pose at t and t+1 were nearly identical.
# Use --feature-mode frame to reproduce the legacy behaviour for ablation.
python preprocessing/compute_pretrain_targets.py \
    --manifest data/ssl_pool_manifest.json \
    --output data/pretrain_kmeans_targets.npz \
    --num-clusters 64 --fit-sample 0.10 --seed 0 \
    --feature-mode pose_motion

# C.3: masked SSL pretraining (~10-30 h GPU)
python main_pretrain.py --config config/bdsl_shubert_pretrain.yaml --seed 0

# C.4: fine-tune on BdSLW60-SI x 3 seeds
for seed in 0 1 2; do
  python main.py --config config/bdsl_block_gcn_si.yaml \
      --seed $seed -Experiment_name bdsl_block_gcn_shubert_seed$seed \
      --weights work_dir/bdsl_shubert_pretrain/pretrained_epoch30.pt \
      --ignore-weights fc.weight fc.bias
done
```

### 4.8 Stage D — cross-recording robustness (T6)

BdSL60-SingleTrial eval NPY is already produced at
`data/bdsl60_singletrial_eval/eval_data.npy` (774 clips). Evaluate every
Stage-A checkpoint on it:

```bash
# Example for BlockGCN seed 0:
python main.py --config config/bdsl_block_gcn_si.yaml --phase test \
    --weights work_dir/bdsl_block_gcn_si_seed0/bdsl_block_gcn_si_seed0_model_best.pt \
    --test-feeder-args "{'data_path':'data/bdsl60_singletrial_eval/eval_data.npy','label_path':'data/bdsl60_singletrial_eval/eval_label.pkl','window_size':120,'random_choose':False,'normalization':True}" \
    -Experiment_name bdsl_block_gcn_si_seed0_evalST
# Repeat for every model x seed in Stage A.
```

For **T6**, compute `drop_ST(model) = Top1_SI_val(model) − Top1_BdSL60ST(model)`.

> **Caveat (audit fix #7)**: The full SingleTrial bundle (774 clips) spans all
> 18 signers — 479 clips (62 %) share a signer with the SI training set. So
> a Top-1 on the full bundle conflates **cross-recording robustness** with
> **seen-signer-but-different-take** generalization. For the headline T6
> number, filter the eval bundle to the **295-clip held-out subset** (signers
> NOT in `SIGNER_SPLIT['train']`), which still covers all 60 classes. Report
> the full-bundle number too in the appendix for comparison.
> Verified by `tests/test_singletrial_signer_overlap.py`.

### 4.9 Compile the paper results directory

```bash
mkdir -p results
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/master_table.md
```

## 5. Reviewer-defense notes

| Anticipated criticism | Defensive evidence |
|---|---|
| "99 %+ numbers in other BdSLW papers" | §4.2 comparison of our SD and SI numbers shows the 22.46 pp gap; cite EMNLP-Findings 2025 "Impact of Signer Dependence" as external support. |
| "Only 3 seeds" | We report bootstrap 95 % CIs. 3 seeds is the community norm for SLR; more seeds increases cost without changing conclusions since our effect sizes (CIs) are resolvable. |
| "Gains may be from hyperparameter noise" | `tools/summarize_seeds.py` emits per-seed values; gains exceed per-model seed variance. Include the raw per-seed table in the appendix. |
| "Teacher quality for Path 2 is not measured" | Path 1's P1.3 FlatTemporal-Bangla-DINOv2 row *is* the teacher-quality measurement. |
| "Option C pretraining corpus includes BdSLW60-pretrain signers" | Those 7 signers are labelled only for 38 of 60 classes; using them *unlabeled* in SSL does not leak SI val/test labels (confirm with `splits/signer_independent.json`). |
| "Cross-recording drop might just be frame-rate difference" | Measure this: BdSL60-SingleTrial clip mean length vs BdSLW60-SI. Report in appendix. |
| "Why BlockGCN specifically" | Zhou CVPR 2024 shows it's topology-aware + param-efficient; we show it's the best skeleton model on BdSLW60-SI in T1. |

## 6. Time-to-paper budget

Optimistic single-researcher timeline (1 GPU):

| Week | Milestone |
|---|---|
| 1 | Stage A complete (T1); write §3 protocol. |
| 2 | Stage B.1–B.3 complete (T2); write §4 feature isolation. |
| 3 | Path 1 complete (T3); draft §5 domain adaptation. |
| 4 | Path 2 complete (T4); draft §6 knowledge distillation. |
| 5 | Option C complete (T5); Stage D (T6). |
| 6 | Full draft, appendices, reviewer defense section. |

Tight but workable. The two risks are BdSLW401 pose-extraction completion
(~14 h pending) and SHuBERT pretraining convergence (unknown until we try).

## 7. Appendix — where every artefact lives

* Pilot training log: `logs/train_block_gcn_si_seed0.log`
* Raw per-seed CSV: `results_final.csv`
* Aggregated tables: `results/T*.md`
* Model checkpoints: `work_dir/<exp>_seed<N>/`
* SSL pool manifest: `data/ssl_pool_manifest.json`
* Canonical signer split: `splits/signer_independent.json`
* Memory summary for future re-runs: `C:\Users\rimon\.claude\projects\F--SLGTformer\memory\project_first_si_baseline.md`

---

## 8. Methodology audit & remediation

This section tracks methodology bugs found during pre-submission audit and
the fixes applied. Each entry names the audit ID, what was wrong, what
changed, and the smoke test that locks the fix in place.

### 8.1 Audit fix #1 — temporal cluster targets for SSL

**Was**: `compute_pretrain_targets.py` clustered each frame's bare 81-dim
pose vector. Two adjacent frames in 25 fps pose video are nearly identical,
so cluster IDs at $t$ were trivially predictable from $t \pm 1$. SHuBERT's
masked-prediction task degenerated into "copy the neighbour."

**Now**: default `--feature-mode pose_motion` clusters on 162-dim
(pose + first-derivative). Optional `--feature-mode window<K>` stacks $K$
consecutive frames (e.g. 324-dim for window4) for closer parity with
SHuBERT-for-audio's MFCC-window paradigm. Legacy `--feature-mode frame`
preserved for ablation.

**Tests**: `tests/test_option_c_smoke.py::test_pretrain_target_feature_modes_*`
verify (a) correct dim per mode, (b) pose_motion neighbour distance >
pose-only neighbour distance on random-walk pose, (c) window-mode boundary
zero-padding.

**Action required**: re-run `preprocessing/compute_pretrain_targets.py`
once and re-launch SHuBERT pretraining.

### 8.2 Audit fix #3 — sister paper S2 transfer matrix protocol

**Was**: §6.4 of `RUNBOOK_SISTER_PAPER.md` listed cross-dataset transfer
as a TODO. Per-source label spaces are disjoint, so "train on A, classify B"
is not a well-defined operation.

**Now**: `path3_handshape_benchmark/eval_cross_dataset.py` implements the
canonical protocol: freeze encoder trained on A → extract features on B's
train+val sets → fit `sklearn.linear_model.LogisticRegression` on B's train
features → eval B's val Top-1. Writes an N×N markdown matrix + row/col
marginals.

**Run**: after Path 3 H1 has produced LoRA encoder checkpoints,

```bash
python -m path3_handshape_benchmark.eval_cross_dataset \
    --encoder-dir work_dir/bhc_lora \
    --epoch 10 --seed 0 \
    --output results/S2_transfer_matrix.md
```

### 8.3 Audit fix #4 — SSL pool data-leak smoke test

**Was**: `build_ssl_pool_manifest.py` had `--exclude-si-val-test` (default on)
but no automated test asserting it works on real data. A regex bug would
have silently leaked val/test signers into SSL pretraining → inflated SI
Top-1.

**Now**: `tests/test_ssl_pool_no_leak.py` (3 tests) asserts no clip path
contains a signer ID from `SIGNER_SPLIT['val'] ∪ SIGNER_SPLIT['test']`,
that per-source counts match a recount, and that the `excluded_signers`
metadata field is consistent.

### 8.4 Audit fix #7 — BdSL60-SingleTrial signer breakdown

**Was**: T6 was reported on the full 774-clip SingleTrial bundle without
distinguishing which clips share signers with the training set.

**Finding** (from `tests/test_singletrial_signer_overlap.py`):

| Bucket | Signers | Clips |
|---|---|---:|
| SI train signers {1,4,5,6,8,9,11,12} | all 8 | 479 |
| SI val/test signers {2,13,15} | all 3 | 179 |
| SI pretrain-pool signers {3,7,10,14,16,17,18} | all 7 | 116 |

**Now**: report T6 on the **295-clip held-out subset** (signers NOT in SI
train) as the headline number. Full-bundle number goes in the appendix as
"recording-condition-only" comparison. See caveat box in §4.8.

### 8.5 Audit fix #2 — LOSO (leave-one-signer-out) Stage A sweep

**Critique**: SI test set is only 2 signers. Three init seeds estimate
init/shuffle noise, not split-choice noise — CIs are tight around the
wrong distribution.

**Scaffolded** (awaiting HPC launch):
* `preprocessing/generate_loso_split_bundle.py` — one fold's data bundle.
  Re-uses the pose cache (no MediaPipe re-extraction), ~30-60 s per fold.
* `tools/run_loso.py` — sweep driver. Generates each fold once, then
  trains every (config × fold × seed) combination via main.py.
* `tests/test_loso_assignment.py` — 3 unit tests on the assignment helper.

**Run commands**:

```bash
# Full 11-fold LOSO at 3 seeds for ONE headline model (e.g. BlockGCN):
python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
    --test-signers 1 4 5 6 8 9 11 12 2 13 15 --seeds 0 1 2

# 3-fold LOSO at 3 seeds for the 8 baseline models (cheap variance estimate):
python tools/run_loso.py --config experiments_si_baselines.yaml \
    --test-signers 2 13 15 --seeds 0 1 2

# Resume after crash:
python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
    --test-signers 1 4 5 6 --seeds 0 1 2 --skip-existing
```

Per-fold data lands under `data/bdsl_si_loso/test_U<XX>_val_U<YY>/`.
Each run writes a row to `results_final.csv` with
`Experiment = <config_stem>_loso_test<XX>_seed<N>`; the aggregator
collapses across both seed and LOSO axes when reporting mean ± std.

**Cost**:
* Smart sweep (4 headline × 11 folds + 8 baselines × 3 folds = 68 train
  runs at ~3 GPU-h each) ≈ **~9 GPU-days** on one RTX 8000, ≈ **~1.2 wall-
  clock days** on 8-GPU HPC.
* Full sweep (12 models × 11 folds × 3 seeds = 396 runs) ≈ **~50 GPU-days**
  on one RTX 8000, ≈ **~6 wall-clock days** on 8-GPU HPC.

**Impact**: largest single credibility lift toward CVPR/ICCV-tier. Without
LOSO, the paper headline rests on a single (val,test) signer pair and the
3-seed CI captures init noise only, not signer-choice noise.

### 8.6 Audit fix #9 — 27-keypoint skeleton justification

**Critique anticipated**: *"Why 27 nodes (7 body + 10 per hand)?
MediaPipe Holistic gives 75. Which joints did you drop? Did you ablate?"*

**Response**: full justification in `docs/AUDIT_FIX_9_KEYPOINT_JUSTIFICATION.md`.

Headline points:
* The 27-node graph topology is **inherited directly from SLGTFormer**
  (Song 2022, arXiv:2212.10746) — the keypoint-only WLASL2000 baseline.
  Same node set, same adjacency; we substitute MediaPipe Holistic for
  MMPose HRNet as the keypoint source.
* Per-hand reduction (21 → 10) keeps the wrist + MCP knuckles + finger
  tips + thumb tip; drops PIP/DIP joints. Linguistically motivated:
  handshape = finger selection × aperture (Brentari 1998); MCP/tip carry
  this; intermediate finger joints add redundancy.
* Comparison table vs SignBERT (42), SignBERT+ (50), SHuBERT (75),
  SPOTER (54), ASL-Citizen baseline (71) — we sit at the parsimonious end.
* Reviewer-defense paragraph is paste-ready in §5 of the cited doc.

**Backup plan if reviewers escalate**: ~1 GPU-day to re-extract pose
with full 21-per-hand layout and run a single ablation row.

### 8.7 Audit fix #10 — per-signer hand-detection rates

**Critique**: *"DINOv2 hand crops depend on MediaPipe detection success.
If signer A's hand is detected 95% of frames and signer B's only 60%,
ANY downstream feature already leaks signer identity before learning."*

**Diagnostic** (`tools/diagnose_hand_crop_quality.py`): computes per-
signer hand/face detection rates from a pose or DINOv2 cache.

**Finding on `data/bdsl_cache` (pose layout)**: train-cohort detection
rates have **substantial cross-signer variance** — see
`results/hand_detection_by_signer_pose.md` for the full table.

| Train signer | L-hand % | R-hand % | Flag |
|---|---:|---:|---|
| U01 | 78.5 | 48.4 | **L +2.4 SD** (likely left-handed signer) |
| U04 | 36.6 | 51.1 | both below mean |
| U05 | 34.6 | 82.0 | low L, high R |
| U08 | 40.7 | 90.7 | high R |
| U12 | 21.6 | 38.1 | **R -1.7 SD** (both low) |
| Train cohort mean ± SD | 42.5 ± 15.3 | 67.2 ± 17.5 | |

Even in the pretrain pool (used for SSL):
| Signer | L-hand % | R-hand % | Flag |
|---|---:|---:|---|
| U16 | 85.0 | 59.8 | **L +2.8 SD** |
| U17 | 95.3 | 60.6 | **L +3.5 SD** |

**Implication**: this finding actually *strengthens* the paper's
identity-shortcut framing. Per-signer hand-presence patterns are a real
signal that a SD-split model could exploit; SI evaluation correctly
factors this out. Add a one-sentence reference in §1 (intro) when
motivating the SI protocol, and the full table in the appendix.

**Remaining work**: re-run the diagnostic on `data/bdsl_dino_cache/`
once Stage B extraction lands, to check whether the per-signer DINOv2
crop quality follows the same pattern.

## 9. B-upgrades — reviewer-mandated baselines (added 2026-07-16)

Three additions from the deep-research gap analysis (`SOTA_VENUE_STRATEGY.md`
§1.3), plus the recipe-control policy. Infrastructure is implemented and
smoke-tested; commands:

| Row | Command | Status |
|---|---|---|
| **B3** co-training vs BPT | `python main_cotrain.py --config config/bdsl_block_gcn_cotrain_si.yaml --seed 0` (smoke: `..._smoke.yaml`) | ready to run (~1–2 GPU-days incl. 3 seeds) |
| **B2** RGB baseline (S3D, K400-pretrained) | `python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml --seed 0` | ready to run (~0.5–1 GPU-day/seed) |
| **B1** Uni-Sign pose-only finetune + SHuBERT probe | see `docs/B1_FOUNDATION_BASELINES.md` (labels generated; needs Linux/HPC for deepspeed + user-downloaded checkpoints) | scaffolded |
| Recipe control | policy + paper statement in `docs/RECIPE_CONTROL.md`; LR-grid pre-pass per model | policy written |

Table placement: B2 and B1 rows go in a separate "pretrained / RGB" block
under T1 (they are context anchors, not recipe-controlled comparisons);
B3 extends T5's pretraining comparison (from-scratch vs BPT vs co-training
vs SSL-init).
