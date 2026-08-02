# Run Guide — Options A, B, C on Signer-Independent BdSLW60

This guide walks through, in order, the commands required to reproduce
every result in the signer-independent BdSLW benchmarking program. Each
stage lists its command and a rough wall-clock estimate on one
Quadro RTX 8000 (48 GB).

All commands assume the repo root `F:\SLGTformer` is the current working
directory and the `bdsl_graph` conda environment is active.

```
conda activate bdsl_graph
```

Outputs per stage are idempotent — a second invocation will skip work whose
caches already exist, so safe to interrupt and resume.

---

## Stage 0 — Prerequisites (once)

**0a. Regenerate the signer-independent BdSLW60 NPYs.**

```
python preprocessing/generate_signer_split_npy.py ^
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" ^
    --output-dir data/bdsl_si ^
    --cache-dir data/bdsl_cache ^
    --splits train val test pretrain
```

- Produces `data/bdsl_si/{train,val,test,pretrain}_data.npy` + label PKLs.
- Per-clip pose caches go to `data/bdsl_cache/<class>/<basename>.npz` so
  re-splitting later is cheap.
- **Wall clock**: ~2.5–3 h on CPU (MediaPipe Holistic).

**0b. Download BdSLW401** (only needed for Option-C SSL pretraining).

```
set KAGGLE_USERNAME=<your_kaggle_username>
set KAGGLE_KEY=<your_kaggle_key>
kaggle datasets download -d hasanssl/bdslw401 -p data/bdslw401_raw --unzip
```

- **Size**: ~48 GB zipped. License CC BY-NC-ND 4.0.
- **Wall clock**: ~1–4 h depending on bandwidth.

---

## Stage A — Option A: rigorous signer-independent benchmark

Runs every existing model *and* the new BlockGCN/RQE variants on the SI
split, at 3 seeds each, then aggregates mean ± std with bootstrap 95% CIs.

```
python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2
python tools/summarize_seeds.py --csv results_final.csv --markdown > results_si.md
```

- Experiments covered (11 models × 3 seeds = **33 training runs**):
  Pose-LSTM, ST-GCN Vanilla, Attention GNN, Adaptive GNN, GNN+LSTM,
  GNN+Transformer, SLGTFormer, BlockGCN, BlockGCN+RQE, FlatTemporal (pose),
  FlatTemporal (DINOv2 — requires Stage B first).
- **Wall clock per single-seed training**: 1–4 h depending on model.
  SLGTFormer is the slowest; Pose-LSTM is the fastest. For the full 33-run
  sweep expect **~2–4 GPU-days**. Use `--skip-existing` to resume.
- `tools/summarize_seeds.py` emits a markdown table of
  `mean ± std, 95% CI` for Top-1 and mean Top-5.

Run a single model instead of the full sweep:

```
python tools/run_multiseed.py --single config/bdsl_block_gcn_rqe_si.yaml --seeds 0 1 2
```

---

## Stage B — Option B: MediaPipe vs DINOv2 feature-isolation

**B.1. Extract DINOv2 hand/face-crop features** from every labeled clip.

```
python preprocessing/extract_dinov2_features.py ^
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" ^
    --output-dir data/bdsl_si_dino ^
    --cache-dir data/bdsl_dino_cache ^
    --splits train val test ^
    --device cuda ^
    --model vit_small_patch14_dinov2.lvd142m ^
    --batch-size 64
```

- Uses MediaPipe to detect hand + face bboxes per frame, then DINOv2-small
  to produce a 384-dim CLS feature per region. V=3 (left hand, right hand,
  face).
- **Wall clock**: ~3–5 h on GPU (batched DINOv2 inference dominates).
- **Output**: `data/bdsl_si_dino/<split>_data.npy` with shape
  `(N, 384, T_max, 3, 1)` compatible with the existing feeder.

**B.2. Train the fair head-to-head.**

The two configs use the same architecture (flat temporal transformer,
6 layers, 8 heads, d_model=256). They differ only in the input features,
so any Top-1 gap is attributable to the representation.

```
python tools/run_multiseed.py --single config/bdsl_pose_temporal_si.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2
```

**B.3. Measure the signer-dependent vs signer-independent *gap*.**

The Option-B claim is that DINOv2 closes the gap between
signer-dependent and signer-independent accuracy (the gap is the size of the
"MediaPipe encodes identity" shortcut). Produce the signer-dependent
comparison by pointing the same two configs at the old random-split data:

```
copy config\bdsl_pose_temporal_si.yaml config\bdsl_pose_temporal_sd.yaml
copy config\bdsl_dino_temporal_si.yaml config\bdsl_dino_temporal_sd.yaml
:: then hand-edit the two _sd.yaml files so data_path points to data/bdsl/
:: (the old random-split NPYs) instead of data/bdsl_si/ / data/bdsl_si_dino/
python tools/run_multiseed.py --single config/bdsl_pose_temporal_sd.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_dino_temporal_sd.yaml --seeds 0 1 2
```

- Report: Top-1(SI) − Top-1(SD). A bigger drop means more identity-shortcut.
  The hypothesis is that the DINOv2 configuration has a *smaller* drop than
  the pose one.

---

## Stage C — Option C: BlockGCN × RQE × SHuBERT-style pretraining

**C.1. Classification pilot** (already wired; runs as part of Stage A):

```
python tools/run_multiseed.py --single config/bdsl_block_gcn_si.yaml --seeds 0 1 2
python tools/run_multiseed.py --single config/bdsl_block_gcn_rqe_si.yaml --seeds 0 1 2
```

**C.2. SHuBERT-style masked pretraining on BdSLW401** (after Stage 0b):

The full SSL pretraining loop requires two additional components not yet
committed to the repo:
  * a BdSLW401 MediaPipe-extraction pass mirroring
    `preprocessing/generate_signer_split_npy.py` (use the BdSLW401 parser
    in `preprocessing/bdslw401_meta.py`);
  * a `main_pretrain.py` entry point that wraps a `block_gcn.Model` in
    `model.shubert_pretrain.ShubertPretrainer`, iterates over pose caches
    + precomputed k-means cluster targets, and saves the backbone-only
    state dict.

Both are next-up scaffolding; ping me to implement once C.1 numbers land.

---

## Summary table command

After all stages (or at any point), produce the master results table:

```
python tools/summarize_seeds.py --csv results_final.csv --markdown > results_table.md
```

Rows whose `Experiment` matches `<stem>_seed<N>` are collapsed into a single
row reporting mean ± std, bootstrap 95% CI for Top-1, and mean Top-5.
Unsuffixed single-seed runs are reported as-is with `N=1`.

---

## Expected wall-clock budget (single GPU)

| Stage | What | Time |
|---|---|---:|
| 0a  | BdSLW60 pose preprocessing | ~2.5–3 h (CPU) |
| 0b  | BdSLW401 download | ~1–4 h (network) |
| A   | 11 models × 3 seeds, full sweep | ~2–4 GPU-days |
| B.1 | DINOv2 extraction on BdSLW60 | ~3–5 h (GPU) |
| B.2 | FlatTemporal pose+DINOv2 × 3 seeds | ~6–10 h (GPU) |
| B.3 | Repeat B.2 on signer-dependent split | same |
| C.1 | BlockGCN / BlockGCN+RQE × 3 seeds | subset of Stage A |
| C.2 | BdSLW401 preprocessing + SHuBERT pretrain + fine-tune | ~1–2 GPU-days |

Round-trip for Options A + B (without C.2): **~3–5 GPU-days** from a cold
start. Option C.2 adds another 1–2 GPU-days on top.
