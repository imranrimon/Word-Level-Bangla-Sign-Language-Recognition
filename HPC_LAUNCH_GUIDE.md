# HPC Launch Guide — BdSLW60 main paper + sister paper

Step-by-step transfer and execution playbook for running the full
experimental sweep on HPC after the local development phase. Assumes
SLURM-style or similar batch scheduling.

---

## 0. What to transfer from local to HPC

### Code (≈ small)

The whole repo:

```bash
# From the local machine (Windows PowerShell):
$src = "F:\SLGTformer"
$dst = "<hpc-host>:~/SLGTformer"
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='work_dir' `
      --exclude='*.zip' --exclude='data/' --exclude='external/' --exclude='.tmp_loso' `
      "$src/" "$dst/"
```

Or zip + scp:

```powershell
Compress-Archive -Path F:\SLGTformer\* -DestinationPath SLGTformer_code.zip `
  -CompressionLevel Optimal
# Exclude data/ work_dir/ external/ first
```

### Datasets — selective transfer (~120 GB total if you pull everything)

| What | Path | Size | Needed for | Skip if HPC has it already |
|---|---|---:|---|---|
| Pose cache (BdSL) | `data/bdsl_cache/` | ~500 MB | every BlockGCN/SLGTFormer run | no — required |
| SI NPY bundle | `data/bdsl_si/` | ~1.5 GB | Stage A, Path 2 | no — required |
| Pretrain pose (BdSLW401 front) | `data/bdslw401_pose_cache_front/` | ~3 GB | SSL pretrain | no — required |
| Pretrain pose (BdSLW102_A) | `data/bdslw102_a_pose_cache/` | ~200 MB | SSL pretrain | no — required |
| Pretrain pose (WLASL) | `data/wlasl_pose_cache/` | ~250 MB | cross-lingual SSL | only if running BdSL+ASL variant |
| SSL pool manifests + k-means targets | `data/ssl_pool_manifest*.json`, `data/pretrain_kmeans_targets*.npz` | ~30 MB | SSL pretrain | no — required |
| BdSL60-SingleTrial eval bundle | `data/bdsl60_singletrial_eval/` | ~100 MB | T6 (Stage D) | no — required |
| Raw BdSLW60 mp4 | `Word_level_Bangla_Sign_Language_Dataset/BdSLW30/` | ~10 GB | only if re-extracting pose / Stage B DINOv2 / Path 1 extraction | skip if pose cache is enough |
| Handshape image sets | `data/BdSL-MNIST/`, `data/BdSL47/`, `data/BSLD_45/`, `data/bdsl49_extracted/`, `data/ishara_lipi/` | ~600 MB | Path 1 LoRA training + sister paper S1 | required for sister paper / Path 1 |
| ASL-Citizen videos | `data/ASL_Citizen/` | 46 GB | only if extracting pose to add to SSL pool (future) | skip for now |
| WLASL raw mp4 | `data/wlasl_processed/` | 5 GB | optional (have pose cache) | skip |

**Minimal HPC bundle for the planned experiments** ≈ 5–6 GB.

### Conda environment

Two options:

```bash
# Option 1 — re-create from environment.yml
conda env create -f environment.yml
conda activate bdsl_graph

# Option 2 — if the HPC has a newer base, use Python 3.10/3.11 instead:
conda create -n bdsl_graph python=3.10
conda activate bdsl_graph
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1
pip install timm einops opencv-python mediapipe scikit-learn pandas pyyaml tqdm wandb pytest
```

Confirm CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 1. Preflight (do once after transfer)

```bash
cd ~/SLGTformer
python -m pytest tests/ -q                                              # expect 72 passed
python tools/validate_project.py                                        # expect 0 errors, 6 warnings (pending DINOv2 paths)
python tools/diagnose_cluster_occupancy.py --manifest data/ssl_pool_manifest_bdsl_asl.json --targets data/pretrain_kmeans_targets_bdsl_asl.npz --output results/cluster_occupancy_bdsl_asl_post_transfer.md
```

If the cluster-occupancy report matches what you had locally, the data
transfer is byte-equal and you're safe to launch real training.

---

## 2. Execution order — dependency DAG

```
                  ┌──────────────────────────────────────┐
                  │ Stage A — SI baseline (T1)           │ ← parallel-safe across configs
                  │ run_multiseed.py 12 cfgs × 3 seeds   │
                  └──────────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────────────┐
                  ▼               ▼                       ▼
   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │ Stage D — BdSL60-ST  │  │ LOSO sweep (audit #2)│  │ Stage B — DINOv2     │
   │ (T6)                 │  │ run_loso.py          │  │ video features       │
   │                      │  │                      │  │ extract + flat_temp  │
   └──────────────────────┘  └──────────────────────┘  │ T2 partial           │
                                                       └──────────────────────┘
                                                                  │
                                                                  ▼
   ┌──────────────────────┐                          ┌──────────────────────┐
   │ Option C — SSL       │                          │ Path 1 — Bangla      │
   │ pretrain × 2 variants│                          │ DINOv2 LoRA          │
   │ + fine-tunes (T5)    │                          │ (T3)                 │
   └──────────────────────┘                          └──────────────────────┘
                                                                  │
                                                                  ▼
                                                       ┌──────────────────────┐
                                                       │ Path 2 — handshape   │
                                                       │ KD into BlockGCN     │
                                                       │ (T4)                 │
                                                       └──────────────────────┘

                  ┌──────────────────────────────────────┐
                  │ Sister paper — Path 3                │ ← fully independent
                  │ S1 + S3 + S2 transfer matrix         │
                  └──────────────────────────────────────┘
```

---

## 3. Execution blocks (copy-paste, in order)

### G1 — Stage A headline (T1)  ≈ 3 GPU-days on 1× RTX 8000

```bash
cd ~/SLGTformer
python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 --skip-existing
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/T1_stage_A.md
```

### G2 — SSL pretrain × 2 variants (T5)  ≈ 24 GPU-h

Launch in parallel on 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
    --config config/bdsl_shubert_pretrain_bdsl_only.yaml --seed 42 &
CUDA_VISIBLE_DEVICES=1 python main_pretrain.py \
    --config config/bdsl_shubert_pretrain_bdsl_asl.yaml  --seed 42 &
wait
```

Then 3 fine-tunes × 2 variants:

```bash
for v in bdsl_only bdsl_asl ; do
  for s in 0 1 2 ; do
    python main.py --config config/bdsl_block_gcn_si.yaml --seed $s \
      -Experiment_name bdsl_block_gcn_shubert_${v}_seed${s} \
      --weights work_dir/bdsl_shubert_pretrain_${v}/pretrained_epoch30.pt \
      --ignore-weights fc.weight fc.bias
  done
done
```

### G3 — Stage B (T2 — DINOv2 video baseline)  ≈ 5–8 GPU-h

```bash
# B.1 extract DINOv2 features on BdSLW60 SI
python preprocessing/extract_dinov2_features.py \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si_dino --cache-dir data/bdsl_dino_cache \
    --splits train val test --device cuda \
    --model vit_small_patch14_dinov2.lvd142m --batch-size 64

# B.2 flat-temporal × 3 seeds, pose
python tools/run_multiseed.py --single config/bdsl_pose_temporal_si.yaml --seeds 0 1 2

# B.2 flat-temporal × 3 seeds, DINOv2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2
```

Then ALSO run on the SD split (`data/bdsl/`) for T2's full 4 rows
(see `RUNBOOK_MAIN_PAPER.md §4.4`).

### G4 — Path 1 Bangla-DINOv2 (T3)  ≈ 6–11 GPU-h

```bash
python -m path1_bangla_dinov2.train --config path1_bangla_dinov2/configs/train_lora.yaml --seed 0

python -m path1_bangla_dinov2.extract_features \
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
    --output-dir data/bdsl_si_bdino --cache-dir data/bdsl_bdino_cache \
    --encoder-checkpoint work_dir/bdino_lora/encoder_epoch10.pt \
    --splits train val test --device cuda --batch-size 64

# Then create config/bdsl_bdino_temporal_si.yaml (copy of bdsl_dino_temporal_si.yaml
# with data_path swapped to data/bdsl_si_bdino/), and run:
python tools/run_multiseed.py --single config/bdsl_bdino_temporal_si.yaml --seeds 0 1 2
```

### G5 — Path 2 handshape KD (T4)  ≈ 6 GPU-h

Depends on G4 (teacher checkpoints).

```bash
for s in 0 1 2 ; do
  python -m path2_handshape_kd.train_kd \
    --config path2_handshape_kd/configs/train_kd.yaml --seed $s
done
```

### G6 — Stage D cross-recording OOD (T6)  ≈ 3 GPU-h

After G1 has produced Stage-A checkpoints. Template at
`RUNBOOK_MAIN_PAPER.md §4.8`. Important caveat: report headline T6 on
the **295-clip held-out subset** of SingleTrial (signers NOT in SI
train), not the full 774-clip bundle. See audit fix #7.

### G7 — LOSO sweep (audit #2)  ≈ 9 GPU-days on 1× RTX 8000

Smart sweep: 4 headline models × 11 LOSO folds + 8 baselines × 3 folds.
On 8-GPU HPC this is ~1.5 wall-clock days.

```bash
# Headline models (full 11-fold LOSO at 3 seeds each):
for cfg in config/bdsl_block_gcn_si.yaml ; do
  python tools/run_loso.py --single $cfg \
    --test-signers 1 4 5 6 8 9 11 12 2 13 15 --seeds 0 1 2 --skip-existing
done

# Baseline models (3-fold LOSO for variance estimate):
python tools/run_loso.py --config experiments_si.yaml \
  --test-signers 2 13 15 --seeds 0 1 2 --skip-existing
```

### H1–H3 — Sister paper (Path 3)  ≈ 1–2 GPU-days, fully parallel-safe

```bash
# S1 — linear probe + LoRA, 3 seeds each
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/linear_probe.yaml --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora.yaml --seeds 0 1 2

# S3 — SD vs SI on BdSL47 (after duplicating config to lora_sd.yaml with
# val_users:[], test_users:[], random_val_frac:0.10, random_test_frac:0.10)
cp path3_handshape_benchmark/configs/lora.yaml \
   path3_handshape_benchmark/configs/lora_sd.yaml
# (hand-edit lora_sd.yaml; see RUNBOOK_SISTER_PAPER.md §6.5)
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora_sd.yaml --seeds 0 1 2

# S2 — cross-dataset transfer matrix (one encoder checkpoint required first)
python -m path3_handshape_benchmark.eval_cross_dataset \
    --encoder-dir work_dir/bhc_lora \
    --epoch 10 --seed 0 \
    --output results/S2_transfer_matrix.md
```

### G8 — Final aggregation

```bash
mkdir -p results
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/master_table.md
# Optional bootstrap significance test between adjacent rows:
# python tools/paired_bootstrap.py --csv results_final.csv > results/significance.md
#   (paired_bootstrap.py not yet written; see audit fix #6 for the design)
```

---

## 4. Common HPC pitfalls (Windows-developed → Linux HPC)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mediapipe` | mediapipe is pip-only on Linux, not always in conda | `pip install mediapipe` after env creation |
| `IndexError` on first MediaPipe call | older mediapipe versions on certain Linux kernels | `pip install mediapipe==0.10.18` (known-good) |
| `Permission denied` on `data/bdsl_cache/` | rsync transferred read-only | `chmod -R u+w data/` |
| `UnicodeEncodeError` in stdout | HPC locale defaults to ASCII | `export PYTHONIOENCODING=utf-8` |
| YAML configs reference Windows paths | Forward/back slash mismatch | All configs in this repo already use forward slashes |
| `RuntimeError: CUDA out of memory` | HPC GPU is smaller than RTX 8000 | reduce `batch_size` in YAML (32 → 16) |
| `KeyError: 'cluster_centers'` when loading targets | targets file from older run | re-run `compute_pretrain_targets.py` with `--feature-mode pose_motion` |
| `num_worker > 0` hangs on shared filesystem | NFS + DataLoader workers don't mix well | keep `num_worker: 0` (already set in configs) |

---

## 5. Methodology reminders (post-audit state)

Before the paper goes out, double-check that you actually apply each audit-fix in the reporting:

| Audit | What to remember when reporting |
|---|---|
| #1 temporal cluster targets | All SSL pretrain uses `--feature-mode pose_motion`. Cite this in §4. |
| #6 Top-5 policy | Report BOTH `Top-5 @ best Top-1` and `Top-5 best (indep)` — `summarize_seeds.py` outputs both. |
| #7 SingleTrial overlap | T6 headline = 295-clip held-out subset, full-bundle in appendix. |
| #8 strict cross-dataset pool | SSL pool excludes ALL BdSL signers; the BdSL story is purely fine-tune. |
| #9 27-keypoint | Cite SLGTFormer (Song 2022); paste paragraph from `docs/AUDIT_FIX_9_KEYPOINT_JUSTIFICATION.md`. |
| #10 hand-detection per signer | Include `results/hand_detection_by_signer_pose.md` in appendix as identity-shortcut evidence. |
| #11 KD loss | Sweep kd_weight ∈ {0.1, 1.0, 10.0} once; report the best in T4. |
| #12 cluster occupancy | Diagnostic report in appendix; mention that `--per-source-cap 200000` balanced cluster ownership. |

---

## 6. Anti-checklist (do NOT do these)

- ❌ Re-run k-means on a manifest with `data/bdsl_cache` in it (re-introduces audit fix #8). Always use `_bdsl_only` or `_bdsl_asl` manifests.
- ❌ Use `--feature-mode frame` for SSL pretrain (audit fix #1 — trivial task).
- ❌ Report T6 on the full 774-clip SingleTrial bundle (audit fix #7).
- ❌ Edit `work_dir/<exp>/config.yaml` manually after a run — that breaks downstream `fuse_scores.py`.
- ❌ Mix runs from before/after the `summarize_seeds.py` LOSO grouping fix in the same aggregation pass (re-aggregate from scratch).

---

## 6.5 Paper 2 (Cross-Domain) + Paper 3 (Sentence) — launch blocks

These two papers are scoped to run on local RTX 8000 in ~1 day; HPC just
speeds them up further. Detailed plans in
`PAPER2_CROSS_DATASET_PLAN.md` and `PAPER3_SENTENCE_PLAN.md`.

### P2-G1 — Vocabulary alignment (already done locally) — 0 GPU
```bash
python preprocessing/extract_bdslw401_word_names.py \
    --pdf "data/bdslw401_raw/bdsl words-complete.pdf" \
    --output data/bdslw401_words.json \
    --rewrite-classes data/bdslw401_si/classes.json
python preprocessing/build_bangla_vocab_alignment.py \
    --classes bdslw60=data/bdsl_si/classes.json \
    --classes bdslw401=data/bdslw401_si/classes_romanized.json \
    --output data/bangla_vocab_alignment.json
# Then curator-review the candidates + unmatched buckets.
```

### P2-G2 — Train BdSLW401 (3 architectures, 1 seed)  ≈ 16 + 8 + 12 GPU-h
```bash
# Run sequentially OR parallel on 3 GPUs.
python main.py --config config/bdsl_block_gcn_w401.yaml --seed 0 \
    -Experiment_name bdsl_block_gcn_w401_seed0
python main.py --config config/bdsl_st_gcn_w401.yaml --seed 0 \
    -Experiment_name bdsl_st_gcn_w401_seed0
python main.py --config config/bdsl_ctr_gcn_w401.yaml --seed 0 \
    -Experiment_name bdsl_ctr_gcn_w401_seed0
```

### P2-G3 — BPT fine-tune on BdSLW60 (T3 row)  ≈ 3 GPU-h per arch
```bash
for arch in block_gcn st_gcn ctr_gcn; do
  python main.py --config config/bdsl_${arch}_si.yaml --seed 0 \
    -Experiment_name bdsl_${arch}_bpt_si_seed0 \
    --weights work_dir/bdsl_${arch}_w401_seed0/best.pt \
    --ignore-weights fc.weight fc.bias \
    --base-lr 0.01
done
```
(Use `config/bdsl_block_gcn_si.yaml` / `config/bdsl_st_gcn_vanilla_si.yaml`
/ `config/bdsl_adaptive_gnn_si.yaml` as the per-arch SI targets.)

### P2-G4 — Cross-domain transfer matrix (T2)  ≈ 1 GPU-h total (inference only)
```bash
# Six off-diagonal cells per architecture. Driver:
for src in si w401; do
  for tgt in si w401; do
    if [ $src = $tgt ]; then continue; fi
    python tools/eval_cross_dataset_video.py \
      --checkpoint work_dir/bdsl_block_gcn_${src}_seed0/best.pt \
      --source-config config/bdsl_block_gcn_${src}.yaml \
      --target-data data/bdslw60_si/val_data.npy \
      --target-label data/bdslw60_si/val_label.pkl \
      --target-classes data/bdslw60_si/classes.json \
      --alignment data/bangla_vocab_alignment.json \
      --source-name $src --target-name $tgt \
      --output results/T2_transfer_matrix.jsonl
  done
done

# BdSLW60-SingleTrial held-out 295-clip eval — see audit fix #7.
python tools/eval_cross_dataset_video.py \
    --checkpoint work_dir/bdsl_block_gcn_si_seed0/best.pt \
    --source-config config/bdsl_block_gcn_si.yaml \
    --target-data data/bdsl60_singletrial_eval/eval_data.npy \
    --target-label data/bdsl60_singletrial_eval/eval_label.pkl \
    --target-classes data/bdsl_si/classes.json \
    --alignment data/bangla_vocab_alignment.json \
    --source-name si --target-name si \
    --output results/T2_transfer_matrix.jsonl
```

### P3-G1 — Sentence scratch baseline  ≈ 1 GPU-h
```bash
python main.py --config config/bdsl_block_gcn_w102a_sentence.yaml --seed 0 \
    -Experiment_name bdsl_block_gcn_w102a_scratch_seed0
```

### P3-G2 — Sentence BPT (after P2-G2 produces backbone)  ≈ 1 GPU-h
```bash
python main.py --config config/bdsl_block_gcn_w102a_sentence.yaml --seed 0 \
    -Experiment_name bdsl_block_gcn_w102a_bpt_seed0 \
    --weights work_dir/bdsl_block_gcn_w401_seed0/best.pt \
    --ignore-weights fc.weight fc.bias \
    --base-lr 0.01
```

### G8b — Significance test (audit fix #6 follow-up)
```bash
# Pairwise significance over every experiment group in results_final.csv:
python tools/paired_bootstrap.py --csv results_final.csv \
    --n-resamples 10000 --output results/significance.md

# Or a specific A-vs-B comparison (e.g., BPT vs scratch on BdSLW60):
python tools/paired_bootstrap.py --csv results_final.csv \
    --a bdsl_block_gcn_bpt_si --b bdsl_block_gcn_si --n-resamples 10000
```

---

## 7. Final smoke test on HPC before scaling up

After the env is set up and data is in place:

```bash
# 1. Quick training smoke (~2 min, 1 GPU)
python main.py --config config/bdsl_block_gcn_si_smoke.yaml --seed 0 \
    -Experiment_name bdsl_block_gcn_si_smoke

# 2. Quick SSL smoke (~3 min, 1 GPU)
python main_pretrain.py --config config/bdsl_shubert_pretrain_smoke.yaml --seed 0

# 3. Path 3 smoke (~3 min, 1 GPU)
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora_smoke.yaml --seeds 0
```

If all three produce row(s) in `results_final.csv`, the full pipeline
works on HPC and you can launch G1–G8 with confidence.
