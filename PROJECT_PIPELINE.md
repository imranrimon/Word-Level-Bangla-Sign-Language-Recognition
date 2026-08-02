# Project Pipeline — Word-Level Bangla Sign Language Recognition (master reference)

This is the single document that summarizes the full research program: what
datasets exist, what role each plays, what stages they feed, and what
commands move data forward through the pipeline. It supersedes scattered
notes — `RUN_GUIDE.md` is the per-command runbook; this file is the
"architecture diagram in markdown."

---

## 1. Goal

Produce a **rigorous, signer-independent benchmark for word-level Bangla
Sign Language Recognition (BdSL)**, plus a small number of novel
contributions:

* **Option A** — clean SI baseline table for every model in the repo.
* **Option B** — DINOv2 feature representation vs MediaPipe-pose, to
  measure the "signer-identity shortcut".
* **Option C** — BlockGCN backbone + Relative Quantization Encoding
  positional prior + SHuBERT-style masked SSL pretraining on BdSLW401.

The end product is a results table + ablation table + an analysis of the
identity-shortcut gap, suitable for an ICCVW MSLR / domain-venue paper.

---

## 2. Datasets on disk and their roles

| Dataset | Size (zip / extracted) | Modality | Pipeline role | Status |
|---|---|---|---|---|
| **BdSLW60** (raw videos) | — / 2.6 GB | video, multi-trial | **Target dataset** for SI classification | ✅ on disk; signer-independent NPYs at `data/bdsl_si/` |
| **BdSLW401** | 49 / 52 GB | video, multi-view (Front + Lateral), multi-trial | **SSL pretraining corpus** (Option C); **secondary target eval** at 401 classes | 🟡 extracting now (bg `bg2xjvk7v`) |
| **BdSL60.zip** | 24 / 25 GB | video, single-trial-per-(signer,word) | **Cross-recording-condition test set** for trained BdSLW60 models | ⏳ queued (extracts after BdSLW401) |
| **BdSLW102_A.zip** | 8.7 / 9.3 GB | video; 50 nested `.rar`s | Potential additional pretraining pool (Word/ subset) | ⛔ **blocked**: no RAR extractor on this machine; install 7-Zip or `conda install -c conda-forge unrar` to enable |
| **BSLD_45.zip** | 1.3 / 1.4 GB | image (94 k .jpg crops) | Out of scope for skeleton pipeline; potential CNN handshape pretraining if an image branch is added later | ⏸ keep zipped |
| **BdSL47.zip** | 2.5 / 2.8 GB | letters + digits (nested `.rar`s) | Different task (alphabet); skip | ⏸ keep zipped |
| **BDSL 49** | 7.4 / 7.9 GB | detection (10 nested `.zip`s) | Different task (detection / segmentation); skip | ⏸ keep zipped |
| **BdSL-MNIST.zip** | 0.2 GB | image alphabet (1 nested `.zip`) | Different task (alphabet); skip | ⏸ keep zipped |

**Tier-1 (used in pipeline, extracting / extracted)**: BdSLW60, BdSLW401,
BdSL60.zip — total ~80 GB of raw video, all parseable for signer ID.

**Tier-2 (deferred, needs setup)**: BdSLW102_A — 50 nested RARs, blocks
on a system RAR extractor. Worth integrating if you want more SSL data.

**Tier-3 (out of scope)**: BSLD_45, BdSL47, BDSL 49, BdSL-MNIST — image
alphabets, detection, etc. Not relevant to the word-level skeleton task.

---

## 3. The pipeline, stage by stage

### Stage 0 — Data preparation (CPU-bound)

```
raw .mp4  ─►  MediaPipe Holistic  ─►  per-clip pose .npz  ─►  bundled per-split .npy
```

Per dataset:

| Dataset | Filename parser | Pose extraction | Output dir |
|---|---|---|---|
| BdSLW60 | `preprocessing/bdsl_signer_split.py` | `preprocessing/generate_signer_split_npy.py` | `data/bdsl_si/` ✅ done |
| BdSLW401 | `preprocessing/bdslw401_meta.py` | *to write*: `preprocessing/preprocess_bdslw401.py` | `data/bdslw401/` (target) |
| BdSL60.zip | reuse BdSLW60 parser (single-trial filenames `U<s>W<w>F.mp4`) — small parser tweak | small adapter using `process_video` | `data/bdsl60_singletrial/` (target) |

Fixed BdSLW60 splits (canonical):
* train signers = {1,4,5,6,8,9,11,12} (5,748 clips)
* val   signers = {15}                  (655   clips)
* test  signers = {2,13}                (1,365 clips)
* pretrain pool = {3,7,10,14,16,17,18}  (1,539 clips, 38 classes only)

### Stage 1 — Pilot training (✅ first pass complete)

* `python main.py --config config/bdsl_block_gcn_si.yaml --seed 0`
* **Result**: BlockGCN (1.4 M params) hit **76.95 % Top-1 / 96.34 % Top-5
  best @ epoch 70**. README claim under random split was 99.41 %, so the
  **identity-shortcut gap is 22.46 percentage points** — this is the
  central motivating measurement for Options B and C.

### Stage A — Option A: full SI benchmark sweep (Option A)

* All 11 models × 3 seeds via `tools/run_multiseed.py`.
* Stats reporting via `tools/summarize_seeds.py` (Top-1 mean ± std,
  bootstrap 95% CI, Top-5 same-epoch).
* **Wall clock**: ~2–4 GPU-days.
* **Output**: `results_si.md` table + `results_final.csv` rows tagged
  `<config_stem>_seed<N>`.

### Stage B — Option B: feature-isolation (DINOv2 vs MediaPipe)

* B.1 (~3–5 h GPU): `preprocessing/extract_dinov2_features.py` produces
  hand+face DINOv2 CLS features, per-clip `(D=384, T, V=3, M=1)`.
* B.2 (~6–10 h GPU): same `model.flat_temporal.Model` trained on (a)
  flattened MediaPipe pose, (b) flattened DINOv2 features. Three seeds each.
* B.3 (~6–10 h GPU): rerun both on the *signer-dependent* (random) split
  too, to compute Top-1(SI) − Top-1(SD). The hypothesis: DINOv2's drop is
  smaller than MediaPipe's, i.e. DINOv2 encodes less identity shortcut.

### Stage C — Option C: BlockGCN × RQE × SHuBERT-style SSL

* C.1 — BlockGCN + RQE classification (subset of Stage A).
* C.2 — *Pending implementation*: `main_pretrain.py` entry point + a
  BdSLW401 MediaPipe-extraction pass + k-means cluster targets, then
  ShubertPretrainer over masked time steps. Fine-tune backbone on
  BdSLW60-SI. Compare to from-scratch BlockGCN.

### Stage D — Cross-dataset robustness checks (NEW with these datasets)

| Train on | Eval on | Question answered |
|---|---|---|
| BdSLW60-SI (any model) | **BdSL60-SingleTrial** (different recording from same dataset family) | Is the model robust to recording-day variation, or did it overfit to multi-trial regularities? |
| BdSLW401 (401-way head) | BdSLW401 test split | Independent benchmark on the bigger dataset |
| BdSLW401-pretrained backbone | BdSLW60-SI fine-tune | Does cross-vocabulary SSL help the small target? |

These transfer evaluations are how the new datasets earn their place in
the project.

### Stage E — Reporting

```
python tools/summarize_seeds.py --csv results_final.csv --markdown > final_table.md
```

For the paper:
1. Headline table — Stage A SI benchmark, all models, mean ± std.
2. Identity-shortcut table — Stage B SI vs SD gap, per representation.
3. Pretraining ablation — Stage C from-scratch vs SSL-init.
4. Robustness — Stage D cross-recording sanity test.

---

## 4. Where we are right now

| Component | State |
|---|---|
| Implementation: BlockGCN, RQE, SHuBERT-pretrain wrapper, flat_temporal, signer split, multi-seed runner, stats aggregator, DINOv2 extractor | ✅ committed and tested (44 smoke tests passing) |
| BdSLW60 SI NPYs | ✅ produced |
| BlockGCN seed-0 SI pilot | ✅ trained — 76.95 % Top-1 |
| BdSLW401 raw videos | 🟡 extracting (bg `bg2xjvk7v`) |
| BdSL60.zip extraction | ⏳ queued |
| BdSLW102_A extraction | ⛔ blocked on RAR tool install |
| BdSLW401 MediaPipe pose pass | ❌ next big CPU job after extraction (~10 h) |
| Stage A multi-seed sweep | ❌ ~2–4 GPU-days, awaiting your green light |
| Stage B DINOv2 + comparison | ❌ ~10 h, awaiting your green light |
| Stage C SSL pretraining | ❌ blocked on BdSLW401 pose preprocessing + missing `main_pretrain.py` |
| Stage D cross-dataset eval | ❌ pending after Stage A |

---

## 5. To enable Tier-2 datasets later

```
# Install a RAR extractor in the conda env
conda install -n bdsl_graph -c conda-forge unrar       # try this first
pip install -n bdsl_graph rarfile                       # python wrapper
# OR install 7-Zip system-wide and put 7z.exe on PATH
# Then BdSLW102_A can be unpacked recursively to data/bdslw102_raw/
```

After RAR support exists, the BdSLW102_A Word/ subset would feed into
either: (a) an additional pretraining corpus alongside BdSLW401, or (b) a
secondary 102-class transfer evaluation.

---

## 6. Disk budget snapshot

| Bucket | Size | Notes |
|---|---:|---|
| Source zips on F: | ~98 GB | safe to delete after extraction once preprocessing is verified |
| Extracted raw videos (target) | ~80 GB | BdSLW60 + BdSLW401 + BdSL60-SingleTrial |
| MediaPipe per-clip caches | ~1 GB / 10 k clips | grows linearly; ~12 GB once BdSLW401 is processed |
| DINOv2 per-clip caches | ~5 GB / 10 k clips at D=384 | only if Option B run |
| Bundled NPYs (per dataset) | ~1 GB each | small |
| Training work_dirs | ~100 MB / model / seed | trivial |
| Free now | 251 GB | comfortable headroom |
