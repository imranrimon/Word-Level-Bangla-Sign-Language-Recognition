# RUNBOOK — Word-Level Bangla Sign Language Recognition

A professional operational runbook for executing the full experimental program:
data preparation, three novel contribution tracks (Options A / B / C), cross-dataset
robustness checks, and final reporting. Intended to be executable top-to-bottom by
the primary author without re-consulting chat history.

All commands assume the repo root `F:\SLGTformer` is the current working directory
and the `bdsl_graph` conda environment is active:

```bash
conda activate bdsl_graph
cd F:\SLGTformer
```

---

## Table of contents

0. [Overview and glossary](#0-overview-and-glossary)
1. [Dataset catalog](#1-dataset-catalog)
2. [Quick-Run Playbook — end-to-end command sequence](#2-quick-run-playbook--end-to-end-command-sequence)
3. [Stage 0 — Prerequisites and verification](#3-stage-0--prerequisites-and-verification)
4. [Stage A — Rigorous signer-independent benchmark](#4-stage-a--rigorous-signer-independent-benchmark)
5. [Stage B — Feature-representation comparison (DINOv2 vs MediaPipe)](#5-stage-b--feature-representation-comparison-dinov2-vs-mediapipe)
6. [Stage C — Self-supervised pretraining + fine-tuning](#6-stage-c--self-supervised-pretraining--fine-tuning)
7. [Stage D — Cross-dataset robustness](#7-stage-d--cross-dataset-robustness)
8. [Final reporting](#8-final-reporting)
9. [Troubleshooting](#9-troubleshooting)
10. [Time and compute budget summary](#10-time-and-compute-budget-summary)

---

## 0. Overview and glossary

**Primary target**: BdSLW60 — 60 word-level Bangla signs, 18 signers, 9,307 clips,
single-view frontal. Evaluated under a **canonical signer-disjoint split**
(`splits/signer_independent.json`).

**Contribution tracks**:

| Track | One-sentence framing |
|---|---|
| **A** | Rigorous signer-independent benchmark of every model in the repo with 3 seeds and proper statistics. |
| **B** | Swap MediaPipe-27 pose for DINOv2 hand/face-crop features, and measure the signer-dependent → signer-independent accuracy drop per representation. A smaller drop with DINOv2 would quantify the "MediaPipe identity shortcut". |
| **C** | BlockGCN backbone + Relative Quantization Encoding + SHuBERT-style masked SSL pretraining on BdSLW401 + BdSLW102_A, fine-tune on BdSLW60-SI. |
| **D** | Cross-recording-condition robustness — train on BdSLW60-SI, evaluate on BdSL60-SingleTrial (different recording day / production conditions). |

**Abbreviations used below**:

* **SI** = signer-independent split (target protocol).
* **SD** = signer-dependent (legacy random split in `data/bdsl/`, kept only for B.3 delta measurement).
* **Top-1@best** = Top-1 accuracy at the epoch that achieved the best Top-1 on the
  held-out val set (this is what the training loop logs to `results_final.csv`).

**Pilot result already on record** (seed 0, signer-independent, BlockGCN, 1.4 M params):
**Top-1 = 76.95 %**, Top-5 = 96.34 % (epoch 70). README's signer-dependent claim was
99.41 %; the **22.46 pp gap** is the central motivating datapoint.

---

## 1. Dataset catalog

All 8 downloaded datasets, classified by what role each plays in the pipeline.
Only 3 of 8 feed the *skeleton* pipeline; another 4 are retained on disk as a
future-work CNN handshape-pretraining pool.

### 1.1 In-pipeline datasets (video, parseable signer IDs)

| Name | Disk location | Size | Classes | Clips | Role |
|---|---|---:|---:|---:|---|
| **BdSLW60** | `Word_level_Bangla_Sign_Language_Dataset/BdSLW30/` | 2.6 GB | 60 | 9,307 | **target** (Options A, B, C, D) |
| **BdSLW401** | `data/bdslw401_raw/` (and `data/BdSLW401/`) | 49 GB | 401 | 102,176 | **SSL pretraining corpus** (Option C); also a secondary 401-class eval |
| **BdSL60-SingleTrial** | `data/bdsl60_singletrial/` (and `data/BdSL60/`) | 24 GB | 60 | 777 | **cross-recording test set** (Stage D) |

### 1.2 SSL-pool-only dataset (sentence-level, used unlabeled)

| Name | Disk location | Size | Content | Role |
|---|---|---:|---|---|
| **BdSLW102_A (sentences)** | `data/bdslw102_a_videos/` (pose cache at `data/bdslw102_a_pose_cache/`) | 9 GB | 3,408 sentence clips, with/without background masking | additional unlabeled pool for SHuBERT-style SSL |

### 1.3 Out of scope for the skeleton pipeline (image / alphabet / detection)

Retained on disk as reference data; could be repurposed for a CNN handshape
pretraining branch in future work. **Not preprocessed into the pose pipeline.**

| Name | Location | Modality | Why out of scope |
|---|---|---|---|
| BDSL 49 | `data/bdsl49_extracted/` + original folder | **still images** (49-class JPG + Detection annotations) | images, not video — would need a new CNN branch |
| BSLD_45 | `data/BSLD_45/` | **still images** (94 k JPGs, 45 classes, `Data/{Train,Augmented,Test,Validation}/<class>/*.jpg`) | images, not video |
| BdSL47 | `data/BdSL47/Bangla Sign Language Dataset - Sign {Digits,Letters}/` | **images, alphabet-level** (letters + digits, by signer+sign) | alphabet task, not word-level |
| BdSL-MNIST | `data/BdSL-MNIST/` | **images, alphabet-level** (PNG) | alphabet task, not word-level |

### 1.4 The canonical BdSLW60 signer-disjoint split

Fixed by `preprocessing/bdsl_signer_split.py` and committed at
`splits/signer_independent.json`. Measured counts:

| Split | Signers | Clips | Classes |
|---|---|---:|---:|
| train | U01, U04, U05, U06, U08, U09, U11, U12 | 5,748 | 60 |
| val   | U15 | 655 | 60 |
| test  | U02, U13 | 1,365 | 60 |
| pretrain-only | U03, U07, U10, U14, U16, U17, U18 | 1,539 | 38 (incomplete) |
| **total** |  | **9,307** |  |

**Rule**: the *pretrain-only* bucket is never used for labeled classification
reporting — those 7 signers only recorded 9–38 of the 60 words. They feed the SSL
pretraining corpus alongside BdSLW401 and BdSLW102_A.

---

### 1.5 Dataset × Model usage matrix — who consumes what, and how

Plain-English answer to "Which datasets are we using in which model and how?"
Each model is trained / pretrained / evaluated on exactly the datasets listed
in its row below. Same dataset can play different roles across models.

#### Per-model matrix

| Model | Stage | Trained on (labeled) | SSL pretrain pool (unlabeled) | Evaluated on |
|---|---|---|---|---|
| Pose-LSTM | A | BdSLW60-SI train (5,748 clips) | — | BdSLW60-SI val + test; BdSL60-ST (Stage D) |
| ST-GCN Vanilla | A | same | — | same |
| Attention GNN | A | same | — | same |
| Adaptive GNN | A | same | — | same |
| GNN + Bi-LSTM | A | same | — | same |
| GNN + Transformer | A | same | — | same |
| SLGTFormer | A | same | — | same |
| **BlockGCN** | A + C | BdSLW60-SI train | (Option C only) BdSLW60-pretrain + BdSLW102_A + BdSLW401 Front | same |
| **BlockGCN + RQE** | A | BdSLW60-SI train | — | same |
| **SLGTFormer + RQE** | A | BdSLW60-SI train | — | same |
| FlatTemporal (pose) | B | BdSLW60-SI train (pose features) | — | BdSLW60-SI val; *also* BdSLW60-SD for B.3 |
| FlatTemporal (DINOv2) | B | BdSLW60-SI train (DINOv2 features, pre-extracted) | — | BdSLW60-SI val; *also* BdSLW60-SD for B.3 |
| BlockGCN **(SSL-pretrained)** | C | BdSLW60-SI train (fine-tune from pretrained weights) | BdSLW60-pretrain + BdSLW102_A + BdSLW401 Front | same as BlockGCN |

#### Per-dataset matrix

| Dataset | Stage(s) | Role | Models that consume it | How |
|---|---|---|---|---|
| **BdSLW60** | A, B, C, D | primary target | all 12+ classification models | labeled classification on signer-disjoint split (`data/bdsl_si/{train,val,test}_*`) |
| **BdSLW60 (pretrain signers)** | C | unlabeled SSL pool | BlockGCN-SSL | 1,539 clips from signers outside train/val/test (see `data/bdsl_si/pretrain_data.npy` path consumed via pose cache at `data/bdsl_cache/`); fed as unlabeled sequences through SHuBERT masked loss |
| **BdSLW401** | C | unlabeled SSL pool (+ secondary 401-way target, future) | BlockGCN-SSL | 51,098 Front-view clips pose-cached to `data/bdslw401_pose_cache_front/`, iterated by `PretrainFeeder` under masked code-prediction |
| **BdSLW102_A (sentences)** | C | unlabeled SSL pool | BlockGCN-SSL | 3,408 sentence videos pose-cached to `data/bdslw102_a_pose_cache/`; concatenated into SSL manifest |
| **BdSL60-SingleTrial** | D | cross-recording eval | all Stage-A trained models | models trained in Stage A are evaluated with `--phase test` against `data/bdsl60_singletrial_eval/eval_data.npy` (774 clips); measures recording-condition robustness |
| **BdSLW60-SD (legacy random split)** | B (only) | SI→SD gap measurement | FlatTemporal (pose + DINOv2) | duplicate `_sd` configs point at the old `data/bdsl/` NPYs; gap `Top1_SD − Top1_SI` quantifies the identity shortcut |
| **BDSL 49 / BSLD_45 / BdSL47 / BdSL-MNIST** | none | out of scope (image / alphabet / detection) | none in skeleton pipeline | retained on disk only; candidates for a future CNN handshape pretraining branch |

#### The SSL pool composition (the "Option C data" spelled out)

```
Unlabeled SSL pool fed to SHuBERT-style pretraining (total ≈ 56,045 clips):

  1,539   BdSLW60-pretrain signers (U03, U07, U10, U14, U16, U17, U18)
          → word-level, 38 classes, not used for labeled classification
  3,408   BdSLW102_A sentence clips (with + without background masking)
          → sentence-level, no word labels needed
 51,098   BdSLW401 Front-view clips (all 401 words, all BdSLW401 signers)
          → word-level; labels not used here (the SSL stage treats all as unlabeled)
 ───────
 56,045   pose-cache clips fed to the masked-code-prediction objective
```

#### How each "how" actually works (one paragraph each)

* **Labeled classification (Stages A, B, D)**: the feeder loads a
  `(N, C, T, V, M)` NPY, yields `(C, T, V, M)` per sample with a `window_size`
  crop. Model forward → cross-entropy over `num_class` logits.
* **DINOv2 feature extraction (Stage B.1)**: videos are decoded; MediaPipe
  gives hand/face bounding boxes per frame; each box is cropped to 224×224 and
  forwarded through DINOv2-small; the CLS token (384-dim) per region per frame
  is saved. Output NPY shape `(N, 384, T, 3, 1)` where `V=3` = (left hand,
  right hand, face).
* **SHuBERT-style SSL (Stage C)**: `ShubertPretrainer` randomly masks a
  `mask_ratio` fraction of time steps with a learnable token. The backbone
  (BlockGCN with `return_features=True`, adapted to emit per-time features)
  produces `(N, T, feat_dim)`. A linear head predicts the k-means cluster ID
  (1-of-64) per time step. Cross-entropy is computed **only at masked
  positions**. Backbone weights are saved per `save_interval` epochs.
* **SSL → fine-tune handoff (Stage C.4)**: classifier-only `main.py` with
  `--weights <pretrained.pt> --ignore-weights fc.weight fc.bias` loads the
  pretrained backbone state dict, reinitialises the 60-way head, and proceeds
  with standard supervised fine-tuning on BdSLW60-SI.
* **Cross-recording eval (Stage D)**: `main.py --phase test` with a
  `--test-feeder-args` override redirecting to the BdSL60-SingleTrial bundled
  NPY. No training, just forward + accuracy.

---

## 2. Quick-Run Playbook — end-to-end command sequence

Numbered commands to drive the project from a fresh environment to final
results tables. **Each step is one copy-paste block.** Skip ahead if an
earlier artefact already exists (the scripts are resumable via
`--skip-existing`). Every step links to its detailed reference section.

> **Convention**: `[BG]` = run in background (long job); `[FG]` = run in
> foreground and wait (short); `[ADMIN]` = one-time environment action.
> **Durations** assume one Quadro RTX 8000 (48 GB) and CPU-only MediaPipe.

### Phase 0 — environment and verification (one-time)

**Step 1 `[ADMIN]` — activate environment and go to repo root.**
```bash
conda activate bdsl_graph
cd F:\SLGTformer
```

**Step 2 `[ADMIN]` — one-time installs** (only if cloning fresh; already done
on this machine).
```bash
python -m pip install -q pytest rarfile py7zr kaggle
conda install -n bdsl_graph -y -c conda-forge unrar libarchive
```
*Duration*: ~2 min.

**Step 3 `[FG]` — verify environment + GPU + tests green.**
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m pytest tests/ -q
```
*Expected*: `cuda: True Quadro RTX 8000`, **45 passed**.

**Step 4 `[FG]` — verify preprocessed artefacts.**
```bash
python -c "import numpy as np, pickle, json, os
for s in ('train','val','test','pretrain'):
    a = np.load(f'data/bdsl_si/{s}_data.npy', mmap_mode='r')
    with open(f'data/bdsl_si/{s}_label.pkl','rb') as f: names, labels = pickle.load(f)
    print(f'{s:<10}', a.shape, 'labels', len(labels))
print('pose caches:')
for d in ['data/bdsl_cache','data/bdsl60_singletrial_pose_cache','data/bdslw102_a_pose_cache','data/bdslw401_pose_cache_front']:
    n = sum(1 for _ in os.walk(d) for f in _[2] if f.endswith('.npz'))
    print(f'  {d}: {n} .npz')"
```
*Expected counts*: train=5748, val=655, test=1365, pretrain=1539 | pose caches: BdSLW60 ~9307, BdSL60-ST 777, BdSLW102_A 3408, BdSLW401 Front ~51098 (growing).

### Phase A — Option A: rigorous SI benchmark

**Step 5 `[BG]` — launch the 12-model × 3-seed signer-independent sweep.**
```bash
python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 --skip-existing
```
*Duration*: **~2–4 GPU-days**. Resumable — rerun the same command to skip
completed runs. Logs to per-run `work_dir/<stem>_seed<N>/log.txt` and rows
into `results_final.csv`.

**Step 6 `[FG]` — (optional) monitor progress.**
```bash
tail -n 40 results_final.csv
python tools/summarize_seeds.py --csv results_final.csv --markdown | head -30
```

**Step 7 `[FG]` — aggregate Stage A results into a paper-ready table.**
```bash
mkdir -p results
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_A.md
```
*Success check*: `results/stage_A.md` lists one row per config with `N=3`
and the BlockGCN row mean falls within ±3 pp of the pilot's 76.95 %.

### Phase B — Option B: MediaPipe vs DINOv2 feature isolation

**Step 8 `[BG]` — extract DINOv2 hand/face-crop features on BdSLW60.**
```bash
python preprocessing/extract_dinov2_features.py ^
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" ^
    --output-dir data/bdsl_si_dino --cache-dir data/bdsl_dino_cache ^
    --splits train val test ^
    --device cuda --model vit_small_patch14_dinov2.lvd142m --batch-size 64
```
*Duration*: **~3–5 h GPU**. *Output*: `data/bdsl_si_dino/{train,val,test}_data.npy`
shaped `(N, 384, T_max, 3, 1)`.

**Step 9 `[BG]` — pose baseline × 3 seeds (same architecture as Step 10).**
```bash
python tools/run_multiseed.py --single config/bdsl_pose_temporal_si.yaml --seeds 0 1 2
```
*Duration*: **~4–6 h GPU**.

**Step 10 `[BG]` — DINOv2 features × 3 seeds.**
```bash
python tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2
```
*Duration*: **~4–6 h GPU**.

**Step 11 `[ADMIN]` — (optional, for B.3 identity-shortcut measurement) create
signer-dependent variant configs.** Copy the two SI configs and point their
`data_path`/`label_path` at the legacy `data/bdsl/` split:
```bash
cp config/bdsl_pose_temporal_si.yaml  config/bdsl_pose_temporal_sd.yaml
cp config/bdsl_dino_temporal_si.yaml  config/bdsl_dino_temporal_sd.yaml
# Edit each _sd.yaml: change Experiment_name suffix to _sd,
# replace data/bdsl_si/ with data/bdsl/  and data/bdsl_si_dino/ with data/bdsl_dino/
```

**Step 12 `[BG]` — (optional) train SD versions × 3 seeds each.**
```bash
python tools/run_multiseed.py --single config/bdsl_pose_temporal_sd.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_sd.yaml --seeds 0 1 2
```
*Duration*: **~8–12 h GPU**.

**Step 13 `[FG]` — aggregate Stage B + compute identity-shortcut gap.**
```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_B.md
```
Then manually compute, for each representation `R`:
`identity_shortcut(R) = Top1_SD(R) − Top1_SI(R)`.

### Phase C — Option C: BlockGCN × RQE × SHuBERT-style SSL

**Step 14 `[FG]` — wait for BdSLW401 Front pose extraction to complete.**
Currently running in background. Check:
```bash
find data/bdslw401_pose_cache_front -name '*.npz' | wc -l   # want 51098
```

**Step 15 `[FG]` — (re)build SSL pool manifest including the full pose pool.**
```bash
python preprocessing/build_ssl_pool_manifest.py ^
    --cache-dirs ^
        data/bdsl_cache ^
        data/bdslw102_a_pose_cache ^
        data/bdslw401_pose_cache_front ^
    --output data/ssl_pool_manifest.json
```
*Duration*: ~30 s. *Output*: JSON listing ~56,000 clip paths.

**Step 16 `[BG]` — compute k-means cluster targets for SHuBERT masking.**
```bash
python preprocessing/compute_pretrain_targets.py ^
    --manifest data/ssl_pool_manifest.json ^
    --output   data/pretrain_kmeans_targets.npz ^
    --num-clusters 64 --fit-sample 0.10 --seed 0
```
*Duration*: ~30–60 min CPU.

**Step 17 `[BG]` — SSL pretraining (BlockGCN backbone, SHuBERT-style masked).**
```bash
python main_pretrain.py --config config/bdsl_shubert_pretrain.yaml --seed 0
```
*Duration*: **~10–30 h GPU** for 30 epochs. Saves backbone-only checkpoints to
`work_dir/bdsl_shubert_pretrain/pretrained_epoch{N}.pt`.

**Step 18 `[BG]` — fine-tune pretrained backbone on BdSLW60-SI × 3 seeds.**
```bash
# Pick the last checkpoint or the one with lowest final loss.
for seed in 0 1 2; do
  python main.py --config config/bdsl_block_gcn_si.yaml ^
      --seed %seed% -Experiment_name bdsl_block_gcn_shubert_seed%seed% ^
      --weights work_dir/bdsl_shubert_pretrain/pretrained_epoch30.pt ^
      --ignore-weights fc.weight fc.bias
done
```
*Duration*: **~6 h GPU**.

**Step 19 `[FG]` — aggregate Stage C + compare to from-scratch Stage A row.**
```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_C.md
```
*Success check*: `bdsl_block_gcn_shubert` mean Top-1 should exceed the
Stage-A `bdsl_block_gcn_si` mean by a non-zero margin; if not, the SSL
pretraining did not help and the writeup needs to reflect that honestly.

### Phase D — Cross-recording robustness

**Step 20 `[FG]` — (already done) bundle BdSL60-SingleTrial into an eval NPY.**
Already produced this run: `data/bdsl60_singletrial_eval/eval_data.npy` (774 clips).
To regenerate from scratch:
```bash
python preprocessing/bundle_pose_cache_to_npy.py ^
    --cache-dir data/bdsl60_singletrial_pose_cache ^
    --output-dir data/bdsl60_singletrial_eval ^
    --split-name eval --callback bdsl60_singletrial
```

**Step 21 `[FG]` — evaluate each Stage-A checkpoint on the cross-recording test.**
Repeat for each model + seed. Example for BlockGCN seed 0:
```bash
python main.py --config config/bdsl_block_gcn_si.yaml --phase test ^
    --weights work_dir/bdsl_block_gcn_si_seed0/bdsl_block_gcn_si_seed0_model_best.pt ^
    --test-feeder-args "{'data_path':'data/bdsl60_singletrial_eval/eval_data.npy','label_path':'data/bdsl60_singletrial_eval/eval_label.pkl','window_size':120,'random_choose':False,'normalization':True}" ^
    -Experiment_name bdsl_block_gcn_si_seed0_evalST
```
*Duration*: ~2 min per checkpoint.

**Step 22 `[FG]` — aggregate Stage D: cross-recording accuracy drop.**
```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_D.md
```
Then compute `drop_ST(model) = Top1_SI_val(model) − Top1_BdSL60ST(model)`
for each model's 3-seed mean.

### Phase E — Final reporting

**Step 23 `[FG]` — consolidated master results table.**
```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/master_table.md
```

**Step 24 `[FG]` — validate integrity of the project state.**
```bash
python tools/validate_project.py
python -m pytest tests/ -q
```
*Expected*: no errors, 45 tests pass.

**Step 25 `[ADMIN]` — update project memory with final numbers.**
After Stage A completes, append the multi-seed means to
`C:\Users\rimon\.claude\projects\F--SLGTformer\memory\project_first_si_baseline.md`
so future sessions have the real benchmark, not just the pilot's single seed.

### Minimum viable run (if compute is limited)

If you cannot afford the full 2–7 GPU-day program, the minimum set that
produces a defensible paper is **Steps 1–4, 5, 7** (Option A only, 12 models
× 3 seeds). The 22.46 pp identity-shortcut gap from the pilot alone
demonstrates the central claim.

The **Stage B minimum** additionally requires Steps 8, 9, 10, 13 — roughly
one additional GPU-day.

---

## 3. Stage 0 — Prerequisites and verification

Run **once**, before any training.

### 2.1 Environment

```bash
conda activate bdsl_graph
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m pytest tests/ -q                 # should report 44 passed, 0 failed
```

Expected: CUDA available, RTX 8000 (48 GB), all 44 tests green.

### 2.2 Confirm Stage-0 artifacts exist

```bash
# BdSLW60 signer-independent NPYs
python -c "import numpy as np,pickle;
for s in ('train','val','test','pretrain'):
    a=np.load(f'data/bdsl_si/{s}_data.npy',mmap_mode='r'); print(s, a.shape)"
# expect (5748,3,300,27,1) (655,...) (1365,...) (1539,...)

# Per-clip pose caches
echo 'BdSLW60 pose cache:' $(find data/bdsl_cache -name '*.npz' | wc -l)                      # ~9307
echo 'BdSL60-ST pose cache:' $(find data/bdsl60_singletrial_pose_cache -name '*.npz' | wc -l)  # 777
echo 'BdSLW102_A pose cache:' $(find data/bdslw102_a_pose_cache -name '*.npz' | wc -l)         # 3408
echo 'BdSLW401 Front pose:' $(find data/bdslw401_pose_cache_front -name '*.npz' | wc -l)       # target 51098
```

### 2.3 (Conditional) Regenerate Stage-0 outputs from raw videos

Only needed if `data/bdsl_si/` does not exist or has been deleted.

```bash
# a) BdSLW60 signer-independent NPYs (CPU, ~2.5–3 h)
python preprocessing/generate_signer_split_npy.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si --cache-dir data/bdsl_cache \
    --splits train val test pretrain

# b) BdSL60-SingleTrial pose cache (CPU, ~15 min)
python preprocessing/extract_pose_cache.py \
    --root data/bdsl60_singletrial --cache-dir data/bdsl60_singletrial_pose_cache

# c) BdSLW102_A — first extract rars (needs UnRAR.exe in the conda env)
python tools/extract_rars_recursive.py \
    --src "data/BdSLW102_A/Bangla Sign Language Video Data" \
    --dest data/bdslw102_a_videos --filter Sentence
python preprocessing/extract_pose_cache.py \
    --root data/bdslw102_a_videos --cache-dir data/bdslw102_a_pose_cache

# d) BdSLW401 Front pose cache (CPU, ~1.5–3 h once no other MediaPipe job runs)
python preprocessing/extract_pose_cache.py \
    --root data/bdslw401_raw/Front --cache-dir data/bdslw401_pose_cache_front
```

---

## 4. Stage A — Rigorous signer-independent benchmark

**Goal**: mean ± std Top-1 / Top-5 for every model in the repo, on the canonical
signer-disjoint split, at 3 seeds.

### 3.1 Configs exercised

Listed in `experiments_si.yaml`. The 11 entries:

| Model | Config |
|---|---|
| Pose-LSTM | `config/bdsl_pose_lstm_si.yaml` |
| ST-GCN Vanilla | `config/bdsl_st_gcn_vanilla_si.yaml` |
| Attention GNN | `config/bdsl_gnn_si.yaml` |
| Adaptive GNN | `config/bdsl_adaptive_gnn_si.yaml` |
| GNN + Bi-LSTM | `config/bdsl_gnn_lstm_si.yaml` |
| GNN + Transformer | `config/bdsl_gnn_transformer_si.yaml` |
| SLGTFormer | `config/bdsl_slgtformer_si.yaml` |
| **BlockGCN** | `config/bdsl_block_gcn_si.yaml` |
| **BlockGCN + RQE** | `config/bdsl_block_gcn_rqe_si.yaml` |
| FlatTemporal (pose) | `config/bdsl_pose_temporal_si.yaml` (pending Stage B.1 extraction for paired DINOv2 config) |
| FlatTemporal (DINOv2) | `config/bdsl_dino_temporal_si.yaml` (requires Stage B.1 first) |

### 3.2 Execute the sweep

```bash
# Full 11-config × 3-seed sweep. Resumable.
python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 --skip-existing

# One model only (for a targeted rerun)
python tools/run_multiseed.py --single config/bdsl_block_gcn_rqe_si.yaml --seeds 0 1 2
```

### 3.3 Success criteria

After completion, `results_final.csv` contains per-epoch eval rows tagged by
`<stem>_seed<N>`. The aggregator collapses these to one mean ± std row per
experiment:

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_A.md
```

The output should show:
* One row per config, `N=3` for each (if all seeds completed).
* Top-1 mean for BlockGCN consistent with the pilot's 76.95 % (within seed noise).
* SLGTFormer's SI Top-1 **should not** approach the README's 99.41 % SD claim —
  if it does, the data path is wrong (pointing at `data/bdsl/` instead of
  `data/bdsl_si/`).

### 3.4 Time budget

* Pose-LSTM: ~45 min/seed. ST-GCN Vanilla: ~1 h. Attention/Adaptive GNN: ~1–2 h.
* BlockGCN, BlockGCN+RQE: ~1.5–2 h/seed. GNN-LSTM, GNN-Transformer: ~1.5 h.
* SLGTFormer: ~3–4 h/seed (largest).
* FlatTemporal: ~1.5 h/seed.
* **Full sweep estimated at ~2–4 GPU-days** on one RTX 8000.

---

## 5. Stage B — Feature-representation comparison (DINOv2 vs MediaPipe)

**Goal**: show that swapping MediaPipe-27 poses for DINOv2 hand/face-crop features
reduces the signer-identity shortcut (i.e. narrows the SI → SD accuracy gap).

### 4.1 Extract DINOv2 features

```bash
python preprocessing/extract_dinov2_features.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si_dino --cache-dir data/bdsl_dino_cache \
    --splits train val test \
    --device cuda --model vit_small_patch14_dinov2.lvd142m --batch-size 64
```

Output: `data/bdsl_si_dino/<split>_data.npy` shape `(N, 384, T_max, V=3, M=1)`.
Regions: 0=left hand, 1=right hand, 2=face. **Wall clock**: ~3–5 h GPU.

### 4.2 Train the two halves of the head-to-head

```bash
python tools/run_multiseed.py --single config/bdsl_pose_temporal_si.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2
```

Both use `model.flat_temporal.Model` (same architecture — a 6-layer, 8-head
Transformer over flattened per-frame features). They differ only in input shape
(pose: C=3 V=27; DINOv2: C=384 V=3), isolating the representation.

### 4.3 Measure the SI → SD gap

Duplicate the two SI configs pointing at the old random-split data:

```bash
# manual step: copy and edit data paths
#   config/bdsl_pose_temporal_si.yaml -> config/bdsl_pose_temporal_sd.yaml
#   config/bdsl_dino_temporal_si.yaml -> config/bdsl_dino_temporal_sd.yaml
# Change:
#   data/bdsl_si/  ->  data/bdsl/
#   data/bdsl_si_dino/  ->  data/bdsl_dino/    (need to also run B.1 on data/bdsl/ labels)
```

Then:

```bash
python tools/run_multiseed.py --single config/bdsl_pose_temporal_sd.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_sd.yaml --seeds 0 1 2
```

### 4.4 Reporting

Compute, for each representation R ∈ {pose, DINOv2}:
`drop(R) = Top-1(R, SD) − Top-1(R, SI)`.

Hypothesis: `drop(DINOv2) < drop(pose)` — DINOv2 encodes less signer identity,
so the shortcut is smaller, so the drop when moving to SI is smaller.

### 4.5 Time budget

* B.1 DINOv2 extraction: ~3–5 h GPU.
* B.2 two FlatTemporal configs × 3 seeds on SI: ~8–12 h GPU.
* B.3 same on SD: ~8–12 h GPU.
* **Stage B total: ~20–30 h GPU**.

---

## 6. Stage C — Self-supervised pretraining + fine-tuning

**Goal**: pretrain a BlockGCN backbone on the ~56 k unlabeled Bangla-domain pose
sequences (BdSLW401 Front + BdSLW102_A + BdSLW60-pretrain), then fine-tune on
BdSLW60-SI. Compare against the from-scratch BlockGCN baseline from Stage A.

All pieces listed below are **implemented**. See Phase C of the Quick-Run
Playbook (Steps 14–19) for the executable sequence.

### 6.1 Build the SSL pool manifest

```bash
python preprocessing/build_ssl_pool_manifest.py \
    --cache-dirs \
        data/bdsl_cache \
        data/bdslw102_a_pose_cache \
        data/bdslw401_pose_cache_front \
    --output data/ssl_pool_manifest.json
```

*Output*: `data/ssl_pool_manifest.json` listing ~56 k clip paths (exact count
depends on how many BdSLW401 Front clips have been pose-processed).

### 6.2 Compute k-means cluster targets

```bash
python preprocessing/compute_pretrain_targets.py \
    --manifest data/ssl_pool_manifest.json \
    --output   data/pretrain_kmeans_targets.npz \
    --num-clusters 64 --fit-sample 0.10 --seed 0
```

Implementation detail: per-frame feature = flattened 27×3 = 81-dim pose vector.
Uses `sklearn.cluster.MiniBatchKMeans`; fits on 10 % of frames; assigns all
frames across all clips and saves each clip's `(T,)` int32 sequence into a
single compressed NPZ keyed by clip path.

### 6.3 Pretraining entry point — `main_pretrain.py`

```bash
python main_pretrain.py --config config/bdsl_shubert_pretrain.yaml --seed 0
```

Internal flow: loads a BlockGCN backbone with `return_features=True`, wraps it
in `ShubertPretrainer`, iterates the `PretrainFeeder` (pose cache + k-means
targets), trains with AdamW, and saves backbone-only state dicts every
`save_interval` epochs under
`work_dir/bdsl_shubert_pretrain/pretrained_epoch<N>.pt`.

### 6.4 Fine-tune

Use any of the existing classification configs but pass `--weights` pointing at
the saved backbone state dict:

```bash
python main.py --config config/bdsl_block_gcn_si.yaml \
    --seed 0 -Experiment_name bdsl_block_gcn_shubert_seed0 \
    --weights work_dir/pretrain/pretrained_epoch40.pt \
    --ignore-weights fc.weight fc.bias        # drop the 60-class head
```

### 6.5 Reporting

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/stage_C.md
```

Row of interest: `bdsl_block_gcn_shubert` mean ± std compared against
`bdsl_block_gcn_si` (from Stage A). A positive delta is the SSL contribution.

### 6.6 Time budget

* C.1 k-means targets: ~30 min CPU.
* C.2 SSL pretraining: ~8–16 h GPU for one epoch over ~56 k sequences. For 5
  epochs: ~40–80 h GPU. In practice, run 1–2 pretraining epochs first and see.
* C.3 fine-tune × 3 seeds: ~6 h GPU.
* **Stage C total: ~1–3 GPU-days**.

---

## 7. Stage D — Cross-dataset robustness

**Goal**: probe whether models trained on BdSLW60 (multi-trial, continuous-session
recordings) generalise to BdSL60-SingleTrial (one trial per signer-word, potentially
different recording day / production conditions).

### 7.1 Bundle BdSL60-SingleTrial into an eval NPY

**Already done**: `data/bdsl60_singletrial_eval/eval_data.npy` shape `(774, 3, 300, 27, 1)`
has been produced. To regenerate from the pose cache:

```bash
python preprocessing/bundle_pose_cache_to_npy.py \
    --cache-dir data/bdsl60_singletrial_pose_cache \
    --output-dir data/bdsl60_singletrial_eval \
    --split-name eval --callback bdsl60_singletrial
```

Labels are aligned to BdSLW60's class-to-idx mapping via
`splits/word_id_mapping.json` (derived from BdSLW60 filenames — all 60 classes
have a unique W-id). 3 clips are skipped because their W-ids are outside the
BdSLW60 vocabulary (expected).

### 7.2 Evaluate models trained in Stage A on the new test set

```bash
# For each trained checkpoint:
python main.py --config config/bdsl_block_gcn_si.yaml \
    --phase test \
    --weights work_dir/bdsl_block_gcn_si_seed0/bdsl_block_gcn_si_seed0_model_best.pt \
    --test-feeder-args "{'data_path':'data/bdsl60_singletrial_npy/eval_data.npy','label_path':'data/bdsl60_singletrial_npy/eval_label.pkl','window_size':120,'random_choose':False,'normalization':True}" \
    -Experiment_name bdsl_block_gcn_si_seed0_evalST
```

### 7.3 Reporting

For each model, compute the drop:
`drop_ST(model) = Top-1(BdSLW60-SI-val) − Top-1(BdSL60-SingleTrial)`.

A small drop means robust cross-recording generalisation; a large drop means the
model latched onto regularities specific to the BdSLW60 recording session.

### 7.4 Time budget

* Bundling: ~10 min CPU.
* Eval per model: ~2 min GPU.
* **Stage D total: ~1 h** once Stage A checkpoints exist.

---

## 8. Final reporting

### 7.1 Master table

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/master_table.md
```

### 7.2 Paper-ready figures

For each of the four contribution tracks, produce:

* **Table A1** — Stage A headline SI benchmark, mean ± std Top-1/Top-5 for all
  11 configs with bootstrap 95 % CI.
* **Table B1** — Stage B identity-shortcut gap: `drop(pose)` vs `drop(DINOv2)`.
* **Table C1** — Stage C pretraining ablation: from-scratch vs SSL-init.
* **Table D1** — Stage D cross-recording robustness: `drop_ST` for each model.

### 7.3 Project memory snapshot

Durable facts written to `C:\Users\rimon\.claude\projects\F--SLGTformer\memory\`:

* `project_first_si_baseline.md` — 76.95 % pilot number.
* `project_bdslw60_flaws.md` — load-bearing critique of pre-existing repo.
* `project_signer_independent_split.md` — canonical split fixation.

Update `project_first_si_baseline.md` after Stage A completes to record mean ± std.

---

## 9. Troubleshooting

### 8.1 "kaggle.exe is alive but 0 bytes downloaded"

Known bug in `kaggle` 1.7.x + `kagglesdk` — the Python client materialises the
entire response body in RAM via `requests.iter_content()`, which OOMs on
>~30 GB datasets. Workarounds:

* Pin `kaggle==1.6.14`.
* Download via browser (what we actually did for BdSLW401).

### 8.2 "BadRarFile: Failed the read enough data" when extracting BdSLW102_A rars

bsdtar has partial RAR5 support; it lists the archive but fails mid-stream. The
conda-forge `unrar` package ships `UnRAR.exe` at
`C:\Users\rimon\anaconda3\envs\bdsl_graph\Library\bin\UnRAR.exe` but does not
add it to PATH. `tools/extract_rars_recursive.py` auto-detects this location; if
extraction still fails, pass `--unrar <full-path>` explicitly.

### 8.3 "only train part, do not require grad" appearing late in training

The existing harness freezes part of the network during the first
`--only_train_epoch` epoch(s) of warmup, then unfreezes. Expected; not a bug. If
you suspect it is blocking learning, set `only_train_epoch: 0` in the config.

### 8.4 Model training hits NaN / Inf

Most commonly from `base_lr=0.1` being too high with some configs. For the
GNN-LSTM / GNN-Transformer family, drop `base_lr` to 0.05 and rerun.

### 8.5 GPU OOM

The SLGTFormer config is the memory hog. Drop `batch_size` from 16 to 8 in its
config and disable gradient accumulation — there is no accumulation in this
harness, so batch 8 is the minimum.

### 8.6 "results_final.csv" shows unexpected low numbers

Verify you are on the SI data path (`data/bdsl_si/`, not `data/bdsl/`) by
checking the feeder lines at the top of the training log.

---

## 10. Time and compute budget summary

Assuming one RTX 8000 (48 GB VRAM) and CPU-only MediaPipe extraction.

| Phase | Wall clock | Dominant resource |
|---|---:|---|
| Stage 0 data prep (already done) | ~4 h | CPU |
| Stage A (11 × 3 sweep) | ~2–4 days | GPU |
| Stage B.1 DINOv2 extract | ~3–5 h | GPU |
| Stage B.2 + B.3 training | ~16–24 h | GPU |
| Stage C.1 k-means | ~30 min | CPU |
| Stage C.2 SSL pretrain (a few epochs) | ~10–30 h | GPU |
| Stage C.3 fine-tune × 3 | ~6 h | GPU |
| Stage D eval | ~1 h | GPU |
| **Full program (A + B + C + D)** | **~4–7 GPU-days** | |

**Useful one-liners** (for daily monitoring while runs are in flight):

```bash
# How many pose-cache clips exist under BdSLW401 Front (the long-running one)?
find data/bdslw401_pose_cache_front -name '*.npz' | wc -l

# How many signer-independent runs have at least one eval row in the CSV?
python -c "import pandas as pd; d=pd.read_csv('results_final.csv'); print(d['Experiment'].unique())"

# Best Top-1 per experiment (collapses same-name runs across seeds):
python tools/summarize_seeds.py --csv results_final.csv

# GPU usage right now:
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv

# Which background training process is alive?
tasklist 2>&1 | grep -i python
```

Running **A then C then B then D** is the recommended order: A produces the
baselines you will compare C (SSL fine-tune) against; B is an orthogonal
investigation that can slip; D needs A's checkpoints.

---

## Appendix — Files referenced by this runbook

* `preprocessing/generate_signer_split_npy.py` — signer-disjoint BdSLW60 NPY builder
* `preprocessing/extract_pose_cache.py` — generic MediaPipe pose → per-clip .npz
* `preprocessing/extract_dinov2_features.py` — DINOv2 hand+face-crop extractor
* `tools/extract_zip.py` — ZIP64-safe zip extractor
* `tools/extract_rars_recursive.py` — recursive RAR extractor (uses UnRAR.exe)
* `tools/run_multiseed.py` — sweep + multi-seed runner
* `tools/summarize_seeds.py` — mean ± std + bootstrap CI aggregator
* `model/block_gcn.py`, `model/rqe.py`, `model/block_gcn_rqe.py`, `model/shubert_pretrain.py`, `model/flat_temporal.py`
* `splits/signer_independent.json` — canonical signer assignment
* `experiments_si.yaml` — master sweep definition
* `PROJECT_PIPELINE.md` — architectural overview (sibling to this runbook)
