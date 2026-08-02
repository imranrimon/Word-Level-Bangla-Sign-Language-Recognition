# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Word-level Bangla Sign Language (BdSLW60) recognition, evolved into a **signer-independent (SI) benchmarking program** with three contribution tracks (see `PROJECT_PIPELINE.md`, the master reference, and `RUNBOOK_MAIN_PAPER.md` / `RUNBOOK_SISTER_PAPER.md`):

- **Option A** — SI baseline table for every model in the repo (11 models × 3 seeds via `run_multiseed.py`).
- **Option B** — feature isolation: MediaPipe pose vs DINOv2 crops through the same `flat_temporal` head, measuring the signer-identity shortcut (SI vs signer-dependent gap).
- **Option C** — BlockGCN backbone + RQE positional prior + SHuBERT-style masked SSL pretraining on BdSLW401.

Architectures: **SLGTFormer** (LGRPE + TTSA + PAF), **BlockGCN** (+KD, +RQE variants), GNN variants (ST-GCN Vanilla, Attention GNN, Adaptive/CTR-GCN-style, GNN+LSTM, GNN+Transformer), Pose LSTM, and FlatTemporal (pose or DINOv2 input). 27-keypoint skeleton (7 body + 10 per hand).

A **sister paper** on still-image Bangla handshape recognition lives in `bangla_handshape/` (shared library) + `path1_bangla_dinov2/` (LoRA-tune DINOv2), `path2_handshape_kd/` (distill handshape teacher into BlockGCN), `path3_handshape_benchmark/` (signer-disjoint image benchmark). `path4_rgb_baseline/` holds the Kinetics-pretrained RGB video baseline (B2). Each `path*/` has its own README and entry points.

Strategy references: `SOTA_VENUE_STRATEGY.md` (novelty map, mandatory baselines, venue plan), `docs/TOPTIER_NEURIPS_ICLR_PLAN.md` (NeurIPS/ICLR framing), `docs/RECIPE_CONTROL.md` (shared-recipe policy for the 11-arch table), `docs/B1_FOUNDATION_BASELINES.md` (Uni-Sign/SHuBERT baseline runbook; Uni-Sign cloned at `external/Uni-Sign`), `docs/AUDIT_FIX_9_KEYPOINT_JUSTIFICATION.md` (reviewer defense for the 27-node skeleton — inherited from SLGTFormer, not an arbitrary choice). `RUNBOOK.md` is the top-to-bottom operational runbook (Stages A–D); the per-paper runbooks scope subsets.

## Environment

```bash
conda env create -f environment.yml      # env name: bdsl_graph
conda activate bdsl_graph
```

Python 3.8, PyTorch + CUDA 11.8, PyG, timm, einops, mediapipe, opencv. Weights & Biases optional (gated by `wandb:` key in configs). On Linux, `mediapipe` is pip-only (`pip install mediapipe==0.10.18` after the conda env).

### HPC (WVU DollySods / SLURM)

The repo now runs on the WVU DollySods cluster as well as Windows. `scripts/hpc/README_HPC_MIGRATION.md` is the current-state migration walkthrough (code/data/env/launch); `HPC_LAUNCH_GUIDE.md` still holds the execution DAG and pitfalls.

- **`data/` and `work_dir/` are symlinks into `/scratch/<user>/SLGTformer/`.** Don't delete or `mkdir` over them; scratch is purge-prone, so back up completed `work_dir/` checkpoints and `results_final.csv`. Large caches (`bdsl_si`, `bdsl_cache`, `bdslw401_si`, SSL pool) are gitignored and transferred by rsync; the 11 GB `bdsl_si_dino` features are re-extracted on-cluster (a GPU step).
- **Launch via SLURM**, not the `.bat` files: `scripts/hpc/slurm_si_sweep.sbatch` (Option A table + ablations), `slurm_loso_array.sbatch`, `slurm_ssl_pretrain.sbatch`, `slurm_ssl_finetune.sbatch`. Fill the `<PARTITION>` `#SBATCH` line from `sinfo -s` before submitting; each script `source`s `~/miniconda3/etc/profile.d/conda.sh`, activates `bdsl_graph`, and sets `PYTHONIOENCODING=utf-8` (DollySods locale is often ASCII).

## Core commands

| Task | Command |
|---|---|
| Train/test a single config | `python main.py --config config/<name>.yaml [--seed N]` |
| Multi-seed sweep (SI benchmark) | `python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 [--skip-existing]` |
| Multi-seed, one model | `python tools/run_multiseed.py --single config/bdsl_block_gcn_si.yaml --seeds 0 1 2` |
| LOSO sweep | `python tools/run_loso.py --single <cfg> --seeds 0 1 2` (re-bundles per-fold data from pose cache) |
| SSL pretraining | `python main_pretrain.py --config config/bdsl_shubert_pretrain.yaml` |
| Co-training ablation (B3) | `python main_cotrain.py --config config/bdsl_block_gcn_cotrain_si.yaml --seed N` |
| RGB baseline (B2) | `python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml` |
| Aggregate seeds/folds | `python tools/summarize_seeds.py --csv results_final.csv --markdown` |
| Significance test | `python tools/paired_bootstrap.py --a <expA> --b <expB>` |
| Cross-dataset eval | `python tools/eval_cross_dataset_video.py --checkpoint ... --alignment ...` |
| Two-stream score fusion | `python tools/fuse_scores.py --joint_dir work_dir/<joint> --bone_dir work_dir/<bone>` |
| Legacy random-split suite | `python tools/run_experiments.py --config experiments.yaml` |
| Summarize results CSV | `python tools/summarize_results.py --csv results_final.csv` |
| Validate repo integrity | `python tools/validate_project.py` |
| All tests | `pytest tests/` |
| Single test | `pytest tests/test_project_smoke.py::test_dummy_forward_pass_per_model_family -k twin_attention` |
| Test-only a trained run | `python main.py --config <cfg> --phase test --weights <ckpt.pt>` |

Windows convenience launchers live in `scripts/*.bat`. `RUN_GUIDE.md` gives per-stage commands with wall-clock estimates; `HPC_LAUNCH_GUIDE.md` covers the SLURM version.

## Architecture — big picture

### Training pipeline (`main.py`)

- **Config-driven.** YAML file → argparse defaults via `parser.set_defaults(**default_arg)`. Any key in the YAML that isn't a registered argparse flag is rejected with `WRONG ARG`. To add a knob, declare the argparse flag first.
- **Dynamic class loading.** `import_class()` resolves `model:` and `feeder:` strings (e.g. `model.block_gcn.Model`, `feeders.feeder.Feeder`) from the YAML — swap architectures by swapping the string, not the code.
- **Run paths.** `resolve_run_paths()` derives `work_dir = ./work_dir/<Experiment_name>` (or config stem), and `model_saved_name = <work_dir>/<name>_model`. `--timestamp-run` appends `_YYYYMMDD_HHMMSS`. The resolved config is copied to `<work_dir>/config.yaml` — downstream tools (e.g. `fuse_scores.py`) read from that copy, so do **not** hand-edit the saved config.
- **Results logging.** Every run appends one row to `results_final.csv` with columns `Timestamp, Experiment, Epoch, Top1_Acc, Top5_Acc, Top5_Policy, WorkDir`. `Top5_Policy` is always `same_epoch_as_logged_top1` — Top-5 is from the *same* epoch that achieved best Top-1, not the independently best Top-5. Multi-seed runs are tagged `<config_stem>_seed<N>`; LOSO runs `<config_stem>_loso_test<XX>_seed<N>` — `summarize_seeds.py` collapses across both suffixes.
- **Checkpoints / scores.** Per-epoch `.pkl` score files land in `<work_dir>/eval_results/`. `fuse_scores.py` prefers `best_acc.pkl`, else picks the highest-accuracy filename.

### SSL pretraining pipeline (`main_pretrain.py`)

Parallels `main.py` but wraps a pose backbone (`model.block_gcn.Model` with `return_features: True`) in `model.shubert_pretrain.ShubertPretrainer`, serving (pose, cluster-id) pairs from `feeders/pretrain_feeder.py`. Saves **backbone-only** checkpoints that `main.py` fine-tunes via `--weights`/`--ignore-weights`. Upstream steps:

1. `preprocessing/build_ssl_pool_manifest.py` — flat JSON manifest of pose-cache clips (the pool must exclude **all** BdSL val/test signers; `tests/test_ssl_pool_no_leak.py` guards this).
2. `preprocessing/compute_pretrain_targets.py` — MiniBatchKMeans cluster IDs per frame. **Use `--feature-mode pose_motion`** (default); legacy `frame` mode is trivially solvable and exists only as an ablation.

### Models (`model/`)

All models expose `Model(num_class, num_point, num_person, graph, graph_args, in_channels, ...)` and accept input `(N, C, T, V, M)` — batch × channels × time × joints × persons. The smoke test `test_dummy_forward_pass_per_model_family` enforces this contract; any new model must satisfy it.

- `twin_attention.Model` — SLGTFormer. Composes `grpe_attention` (LGRPE + PAF spatial attention), `twins_attention_utils` (TTSA two-stream temporal), and `attention.py` (MHSA/RPE-MHSA). Ablation configs (`bdsl_no_lgrpe/_ttsa/_paf.yaml`) toggle these sub-modules.
- `block_gcn.Model` — BlockGCN; `block_gcn_rqe` adds Relative Quantization Encoding (`rqe.py` — RQE originates with the BdSLW60/BdSLW401 authors, cite not claim), `block_gcn_kd` adds a projection head for handshape distillation (fed by `feeders/kd_feeder.py`), `block_gcn_multihead` is the two-head (60+401) co-training variant driven by `main_cotrain.py` (forward returns a logits tuple — not `main.py`-compatible). `slgtformer_rqe` is the RQE-on-SLGTFormer variant.
- `flat_temporal.Model` — architecture-controlled temporal transformer for Option B; input can be flattened pose or DINOv2 features (V=3: L hand, R hand, face).
- GNN variants all depend on `graph.sign_27.Graph` (27-node adjacency; `graph=wlasl` branch). Changing the skeleton topology means touching `sign_27.py` **and** `flip_index` in `feeders/feeder.py` (used for `random_mirror`).

### Data flow

1. **Raw videos → per-clip pose cache → bundled NPYs.** The canonical SI pipeline is `preprocessing/generate_signer_split_npy.py` (MediaPipe Holistic → `data/bdsl_cache/` `.npz` cache → `data/bdsl_si/{train,val,test,pretrain}_data.npy`). The cache makes re-splitting (e.g. LOSO via `generate_loso_split_bundle.py`) cheap. Legacy random-split data in `data/bdsl/` came from `preprocess_bdsl.py`/`preprocess_bdsl_images.py`.
2. **Canonical SI split** (fixed; recorded in `splits/signer_independent.json`, logic in `preprocessing/bdsl_signer_split.py`): train signers {1,4,5,6,8,9,11,12}, val {15}, test {2,13}, pretrain pool {3,7,10,14,16,17,18}. All headline results use this split.
3. **Other datasets.** BdSLW401 parsing in `bdslw401_meta.py`, bundling in `bundle_bdslw401_pose_to_npy.py`; BdSLW102_A sentence-level in `bundle_bdslw102a_sentence_pose_to_npy.py`; vocab alignment BdSLW60↔BdSLW401 in `build_bangla_vocab_alignment.py` (output feeds `eval_cross_dataset_video.py`).
4. **Bone modality.** `preprocessing/generate_bone_data.py` consumes joint `.npy` and emits bone vectors using edges from `graph.sign_27`. Joint and bone runs are independent; fuse at inference time with `tools/fuse_scores.py`.
5. **Feeder.** `feeders/feeder.Feeder` handles `random_choose`, `window_size` cropping, `random_mirror` (via `flip_index`), and optional `lap_pe`.

### Config conventions

- `bdsl_*.yaml` → video; `bdsl_img_*.yaml` → image (legacy random split). Suffixes: `_si` (signer-independent split — current standard), `_bone` (bone stream), `_w401` (BdSLW401 401-class), `_w102a_sentence` (sentence-level), `_smoke` (fast sanity run), architecture tags (`_gnn`, `_block_gcn`, `_pose_lstm`, …), `_no_lgrpe`/`_no_ttsa`/`_no_paf` (SLGTFormer ablations), `_rqe` (RQE variant). `bdsl_shubert_pretrain*.yaml` are `main_pretrain.py` configs.
- Sweep registries: `experiments_si.yaml` (full SI benchmark, consumed by `run_multiseed.py`/`run_loso.py`) and `experiments.yaml` (legacy suite for `run_experiments.py`). The SI benchmark is split for staged HPC launch: `experiments_si_main.yaml` is the runnable-now pose subset (omits the DINOv2 FlatTemporal row `bdsl_dino_temporal_si.yaml`, which needs Stage B.1 features first), and `experiments_si_ablations.yaml` holds the three SLGTFormer sub-module ablations. A config not registered in the relevant registry won't run in the sweep.

## Gotchas

- **The SI split is the ground truth.** The legacy random-split numbers (e.g. 99.41% README claims) are inflated by the signer-identity shortcut — BlockGCN drops to ~77% Top-1 under SI. Never quote random-split numbers as headline results; new experiments should use `_si` configs and the fixed split above.
- **Accuracy numbers are Top-5-policy-sensitive.** `results_final.csv` records *Top5@Top1* (same epoch as best Top-1), not the independently best Top-5. Use `summarize_results.py`/`summarize_seeds.py` for the policy-aware view rather than eyeballing the CSV.
- **SSL pretraining requires `stride_between_stages: False`** on the BlockGCN backbone so temporal resolution is preserved (`T_out == T_in`) for per-frame masked prediction; classification configs keep the default `True`.
- **SSL cluster targets must use `--feature-mode pose_motion`.** The `frame` mode makes the masked-prediction task trivially solvable by copying neighbouring frames.
- **LOSO variance is the headline variance.** Report mean ± std across folds (signer noise), not across seeds (init noise only).
- **Dual-platform, Windows-authored.** `scripts/*.bat` launchers assume a cwd of the repo root (`cd /d "%~dp0.."`) and runbooks use `^` line continuations (cmd) — `RUNBOOK.md` even hardcodes `F:\SLGTformer`. On Linux/HPC use the `scripts/hpc/*.sbatch` equivalents instead and translate paths. `scripts/detach_run.py` is Windows-only (works around a Ctrl+C process-group kill on scheduled tasks) and has no bearing on SLURM runs.
- **`num_worker`** defaults to 32 in argparse but is usually overridden to 0 in the YAMLs — multi-worker DataLoaders can be flaky on Windows + mediapipe pipelines.
- **`graph.sign_27` uses a `wlasl` branch** for the 27-node BdSL skeleton (the name is historical, from the WLASL lineage). Don't be misled by the label.
