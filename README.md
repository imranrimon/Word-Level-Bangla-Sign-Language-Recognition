# xling-pose-ssl-replication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](pyproject.toml)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey.svg)](.github/workflows/ci.yml)

Code, pre-registrations and archived results for the paper
**"A Single-Run Transfer Law Did Not Replicate: A Pre-Registered Audit of Cross-Lingual Pose
Pretraining for Sign Language Recognition"** (under review).

An earlier version of this work reported a *transfer law* for cross-lingual masked-pose
self-supervised pretraining (SHuBERT-style masked code prediction on a BlockGCN encoder): the gain
of an aligned backbone over a plain pooled backbone, `G = acc(xling) - acc(pool)` at 10% labels, fell
on a line in a label-free pose-motion MMD between the pretraining pool and the target language
(R² = 0.997 on five calibration languages, 7/7 sign-correct). Every point of that line came from one
pretraining run. This repository contains everything needed to re-run the replication:

- **Pretraining and fine-tuning.** Masked-pose SSL (`main_pretrain.py`, `model/shubert_pretrain.py`,
  `feeders/pretrain_feeder.py`) with the pool / xling (gradient-reversal language adversary +
  shared-codebook contrastive term) / mono / distance-gated / lambda(d) / CORAL arms, and the 10%-label
  fine-tuning harness (`main.py`, BlockGCN) for eight target languages.
- **Bundle builders** for BdSLW60, BdSLW401, BdSLW102a, WLASL, LSA64, AUTSL, KArSL, GSL, LSFB-ISOL,
  MINDS-Libras, WL-BISINDO and ASL Citizen (27-keypoint MediaPipe skeleton), and for the
  NTU/HMDB51/UCF101/FineGym HRNet skeletons of the negative control.
- **Distances.** The published one-pass pose-motion MMD (`tools/compute_lang_distance.py`) and the
  corrected v2 distance (`tools/compute_lang_distance_v2.py`: real pool source, dropout-masked speeds,
  25 Hz resampling, shoulder normalisation, unbiased MMD²), plus the robustness sweeps.
- **Two pre-registered analyses**, with their hash-locked pre-registrations:
  the 7-pretraining-seed refit of the published law (`docs/p1_refit_2026-10/`, `tools/p1_refit/`) and
  the corrected v2 rerun after the pipeline audit (`docs/p1_fix_2026-10/`, `tools/p1_fix/`).
- **Checkpoint-aware predictor study** (`tools/p1_ckpt_predictor/`), transferability baselines
  (LogME, energy/Fréchet/proxy-A/kNN distances), the AUTSL leak-free gate/CORAL reruns and true
  monolingual baselines (`docs/audit_reruns_2026-10/`).
- **Integrity tools:** `tools/check_pose_geometry.py` (six assertions on 27-keypoint skeletons) and
  `tools/check_run_complete.py` (a run counts only if it reached its final epoch).
- **No-data reproduction.** `python tools/reproduce.py` re-runs both pre-registered analyses, the
  figure script and the exploratory analyses on the archived results and diffs them against the
  numbers in the paper (CI runs it on every push).

## Key results

All numbers below are regenerated from `results/` by `python tools/reproduce.py` (`--full` for the
per-language intervals of the 7-seed refit) and compared with the archived analysis outputs (`docs/p1_refit_2026-10/refit_results.json`,
`docs/p1_fix_2026-10/results/refit_v2.json`).

| analysis | slope b, posterior median [95% CrI] | P(b < 0) | leave-one-language-out sign | binding verdict |
|---|---|--:|--:|---|
| published single run (pseed 0, OLS, 7 languages) | -24.59 (R² 0.889) | | 7/7 | (hypothesis-generating) |
| **7-seed refit** (49 cells; pre-registered: keep the law iff P(b<0) > 0.95 and LOLO >= 6/7) | -3.24 [-14.83, +8.61] | **0.739** | **2/7** | **REFRAME** |
| replication seeds only (pseeds 1-6) | +0.18 [-12.56, +13.19] | 0.486 | 4/7 | |
| **v2 rerun**, corrected pipeline (24 cells, 8 languages; supported iff the 95% CrI of b < 0 and LOLO >= 6/8) | -4.59 [-11.56, +2.61] | **0.905** | **6/8** | **NOT SUPPORTED** |

Pooled over seven pretraining seeds the published estimator gives R² = 0.001 on the five calibration
languages (published 0.997) and 0.082 on all seven. About half of the single-run variance of `G` is
due to the pretraining run (ICC_pseed 0.51 [0.24, 0.73]), almost none of it shared across languages
(ICC_shared 0.07). In the v2 rerun all eight LOLO predictions are negative, so C2 carries no distance
information; using fine-tuning seeds 0-2 only (sensitivity S9) would have flipped the verdict to
"supported", and the held-out-test-signer analysis (S8) has the opposite slope (+12.93 [+1.61, +24.47]).

Pretraining-seed-pooled gain `G = xling - pool` (VAL Top-1 at 10% labels, percentage points; 95%
two-level hierarchical bootstrap over pretraining seeds, then fine-tuning seeds):

| target | d (v1) | 7-seed refit G [95% CI] | d (v2) | v2 rerun G [95% CI] |
|---|--:|--:|--:|--:|
| BdSL (BdSLW60) | 0.143 | +2.26 [-0.47, +4.91] | 0.081 | +1.51 [-0.58, +3.56] |
| LSA64 | 0.176 | +0.82 [-2.47, +3.97] | 0.157 | -1.90 [-5.36, +1.52] |
| KArSL | 0.344 | +1.53 [-2.67, +5.63] | 0.404 | -1.54 [-4.81, +1.66] |
| AUTSL | 0.411 | +0.01 [-2.77, +2.68] | 0.156 | -2.40 [-5.15, +0.81] |
| GSL | 0.121 | -0.72 [-3.82, +1.98] | 0.091 | -3.18 [-6.83, +0.59] |
| LSFB-ISOL | 0.480 | -0.48 [-3.29, +2.44] | 0.145 | -1.72 [-4.28, +0.60] |
| MINDS-Libras | 0.315 | -2.14 [-4.54, +0.21] | 0.137 | +1.18 [-1.41, +4.00] |
| WL-BISINDO | (0.450) | (excluded in v1) | 0.396 | -2.13 [-4.15, -0.21] |

KArSL uses a signer-dependent validation set in v1 and the signer-disjoint split v2a in v2.
In the v2 rerun no per-language gain is significant after Holm correction over the eight languages.

## Installation

Python 3.8, PyTorch 2.x (CUDA 11.8 for training), PyTorch Geometric, einops.

```bash
# GPU environment (conda env `bdsl_graph`)
conda env create -f environment.yml
conda activate bdsl_graph

# or pip, e.g. CPU only (enough for the tests and the no-data reproduction)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# pose extraction only (newest Python 3.8 wheel with the legacy Holistic API)
pip install --only-binary=:all: mediapipe==0.10.9
```

The repository runs from source: execute every command from the repository root.

## Reproducing the paper's numbers without data

```bash
make reproduce            # = python tools/reproduce.py        (about 5 min on 2 CPU cores)
make reproduce-full       # + every pre-registered sensitivity of the 7-seed refit (10-20 min)
make figures              # Figs. 1-4 -> results/figures/
```

| step | script | reads | compared with |
|---|---|---|---|
| v2 rerun | `tools/p1_fix/refit_v2.py` | `results/results_p1fix_v2.csv`, `docs/p1_fix_2026-10/step2/distances_v2.json` | `docs/p1_fix_2026-10/results/refit_v2.json` (byte-identical) |
| 7-seed refit | `tools/p1_refit/refit.py` | `results/results_final.csv` | `docs/p1_refit_2026-10/refit_results.json` |
| results report | `tools/p1_refit/report.py` | the refit JSON | `docs/p1_refit_2026-10/RESULTS.md` |
| figures | `tools/paper/make_figs.py` | both JSONs, `pooled_gains.json`, `distances_v2.json` | asserts every caption number |
| null simulation | `tools/paper/null_sim.py` | refit JSON | `results/provenance/null_sim.txt` |
| exact REML | `tools/paper/exact_reml_v1.py` | `pooled_gains.json` | `results/provenance/v1_exact_reml_2026-10-23.txt` |
| skeleton-action control | `tools/hrnet_gate.py` | run log, `results/hrnet_domain_mmd_norm.log` | `results/provenance/hrnet_gate_2026-10-22.txt` |
| AUTSL leak-free / true mono | `scripts/hpc/audit_reruns_2026-10/collect.py` | run log | tables of `docs/audit_reruns_2026-10/r1_*.md`, `r2_*.md` |

The analyses count a fine-tune only if it reached its final epoch. Without `work_dir/` they read
that from `results/completed_runs.txt` (all 875 analysed runs had both their final and best
checkpoints when the results were frozen); with the runs present locally they check the
checkpoints. `results/results_final.csv` is the run log restricted to the experiments the paper
uses (one row per evaluation; `Top1_Acc` is validation Top-1, `*_test` rows are held-out test signers).

Other analyses read the archived run log the same way: `tools/p1_law_pretrain_seed_analysis.py`
(per-seed law refits), `tools/p1_pretrain_seed_analysis.py` (BdSL headline), `tools/gating_law_fit.py`
(the original single-run law), `tools/p1_law_baseline_refit.py` (transferability comparison;
LogME values in `results/logme_transfer.json`), `tools/p1_ckpt_predictor/score.py analyze`
(needs the extracted features, see below).

## Data

No video, pose data, k-means targets or checkpoints are distributed. Obtain each dataset from its
owners under its own terms and extract poses locally. Every target is converted to the same
27-keypoint MediaPipe Holistic skeleton (7 body + 10 per hand; undetected joints are stored as 0),
bundled as `data/<name>/<split>_data.npy` of shape `(N, 3, T, 27, 1)` with
`<split>_label.pkl = (clip_ids, labels)`.

| dataset | role | source |
|---|---|---|
| BdSLW60 (Rubaiyeat et al., *Multimedia Tools and Applications* 84:42399, 2025) | target (Bangla), pool exclusion | https://www.kaggle.com/datasets/hasaniut/bdslw60videoclips (CC BY-NC-ND 4.0) |
| BdSLW401 (Rubaiyeat et al., *Scientific Reports* 16:25259, 2026; Shawon et al., *PLOS ONE* 21:e0341909, 2026) | pretraining pool (front view) | https://www.kaggle.com/datasets/hasanssl/bdslw401 (data-use agreement) |
| BdSLW102a (BdSLW102_A) | pretraining pool (3,408 clips) | from its owners |
| WLASL (Li et al., WACV 2020) | pretraining pool | MediaPipe landmarks `landmarks_V3.npz` from Kaggle `abd0kamel/mutemotion-output` |
| LSA64 (Ronchetti et al., CACIC 2016) | target (Argentinian) | official LSA64 distribution |
| AUTSL (Sincan & Keles, IEEE Access 2020) | target (Turkish), official signer-disjoint split | official AUTSL distribution (ChaLearn LAP 2021) |
| KArSL (Sidig et al., ACM TALLIP 2021), signs 1-70 | target (Arabic) | official KArSL distribution |
| GSL isolated (Adaloglou et al., IEEE TMM 2022) | target (Greek) | official GSL distribution |
| LSFB-ISOL (Fink et al., IJCNN 2021), top-60 signs | target (Belgian-French) | https://lsfb.info.unamur.be (cleaned MediaPipe poses) |
| MINDS-Libras (Rezende et al., *Neural Computing and Applications* 2021) | target (Brazilian) | Zenodo record 2667329 |
| WL-BISINDO (Kindy et al., *Procedia Computer Science* 2025, doi:10.1016/j.procs.2025.08.277) | target (Indonesian, v2) | from the dataset authors |
| ASL Citizen (Desai et al., NeurIPS Datasets and Benchmarks 2023) | distance reference | official release |
| NTU RGB+D (Shahroudy et al., CVPR 2016), HMDB51, UCF101, FineGym | skeleton-action control | HRNet 2D skeleton pickles from PYSKL (Duan et al., 2022) |

### Bundles

```bash
# BdSLW60, canonical signer-independent split (train 1,4,5,6,8,9,11,12 / val 15 / test 2,13)
python preprocessing/generate_signer_split_npy.py --dataset-root /path/to/BdSLW60 \
    --output-dir data/bdsl_si --cache-dir data/bdsl_cache --splits train val test pretrain
# targets (signer splits fixed in splits/*.json and preprocessing/*_signer_split.py)
python preprocessing/generate_lsa64_split_npy.py   --dataset-root data/lsa64_raw   --output-dir data/lsa64_si   --cache-dir data/lsa64_cache --splits train val test
python preprocessing/generate_autsl_split_npy.py   --dataset-root data/autsl_download --output-dir data/autsl_si --cache-dir data/autsl_cache --splits train val test --jobs 32
python preprocessing/generate_karsl_split_npy.py   --archive-root data/karsl_download --range 0001-0070 \
    --work-dir data/karsl_extracted --output-dir data/karsl_si --cache-dir data/karsl_cache --jobs 8
python tools/rebundle_karsl_2signer.py             # v1 KArSL: U01+U02 train, random 12% val (signer-dependent)
python tools/p1_fix/make_karsl_si_v2.py            # v2 KArSL: split v2a (U01+U02 / U03 / U03) and v2b
python preprocessing/generate_gsl_split_npy.py     --iso-root data/gsl_download/GSL_isolated \
    --csv-dir data/gsl_download/_split/GSL_split/GSL_isolated \
    --iso-classes data/gsl_download/_split/GSL_split/GSL_isolated/iso_classes.csv \
    --output-dir data/gsl_si --cache-dir data/gsl_cache --splits train val test --jobs 8
python preprocessing/download_lsfb_isol_subset.py  # top-60 signs, signer-disjoint 74/9/16 split
python preprocessing/build_lsfb_si_bundle.py
python preprocessing/generate_libras_split_npy.py  --dataset-root data/libras_download --output-dir data/libras_si \
    --cache-dir data/libras_cache --split-json splits/libras_signer_independent.json --splits train val test --jobs 8
python preprocessing/generate_bisindo_split_npy.py --dataset-root data/bisindo_raw --output-dir data/bisindo_si \
    --cache-dir data/bisindo_cache --split-json splits/bisindo_signer_independent.json --splits train val test --jobs 16
python preprocessing/generate_aslcitizen_split_npy.py --dataset-root data/aslcitizen_download \
    --output-dir data/aslcitizen_si --cache-dir data/aslcitizen_cache --splits train val test --jobs 32
python preprocessing/make_aslcitizen_top60_bundle.py       # ASL-Citizen-60 (distance reference)
```

Run the skeleton check on every corpus you add or convert; training code never notices a scrambled
layout:

```bash
python tools/check_pose_geometry.py --npy data/lsa64_si/train_data.npy --name LSA64
python tools/check_pose_geometry.py --all --markdown geometry.md     # every P1 corpus
```

### Pretraining pools and targets

```bash
# pose caches of the pool sources
python preprocessing/extract_pose_cache.py --root /path/to/BdSLW401/Front --cache-dir data/bdslw401_pose_cache_front
python preprocessing/extract_pose_cache.py --root /path/to/BdSLW102a --cache-dir data/bdslw102_a_pose_cache
python preprocessing/convert_wlasl_landmarks_to_pose_cache.py --landmarks data/raw_wlasl_v3/landmarks_V3.npz \
    --output-dir data/wlasl_pose_cache                                     # v1 (scrambled layout; kept for the audit)
python preprocessing/convert_wlasl_landmarks_to_pose_cache_v2.py \
    --landmarks data/raw_wlasl_v3/landmarks_V3.npz --output-dir data/wlasl_pose_cache_v2   # v2 (geometry-checked)

# pool manifest (75,589 clips: BdSLW401 51,098 + WLASL 21,083 + BdSLW102a 3,408; no BdSLW60 val/test signer)
python preprocessing/build_ssl_pool_manifest.py --cache-dirs data/bdslw401_pose_cache_front \
    data/wlasl_pose_cache data/bdslw102_a_pose_cache --output data/ssl_pool_manifest_bdsl_asl.json
# k-means codes (K=64). v1: pose_motion; v2: dropout-masked pose_motion on the v2 pool
python preprocessing/compute_pretrain_targets.py --manifest data/ssl_pool_manifest_bdsl_asl.json \
    --output data/pretrain_kmeans_targets_bdsl_asl.npz --num-clusters 64 --fit-sample 0.1 --seed 0 \
    --per-source-cap 200000 --feature-mode pose_motion
python preprocessing/compute_pretrain_targets_v2.py --manifest data/ssl_pool_manifest_bdsl_asl_v2.json \
    --output data/pretrain_kmeans_targets_bdsl_asl_v2.npz --num-clusters 64 --fit-sample 0.1 \
    --seed 0 --per-source-cap 200000 --feature-mode pose_motion_masked
```

Skeleton-action control: `preprocessing/build_hrnet_domain_bundle.py --domain <ntu_hrnet|hmdb51|ucf101|gym>`
(PYSKL HRNet pickles), `preprocessing/prep_ntu_ssl_source.py --prefix hrnet --domain <src>` (SSL pool of a
source domain), `tools/compute_hrnet_domain_distance.py` (pairwise MMD), then
`config/hrnet_ssl_pretrain_*.yaml`, `config/hrnet_ft_*.yaml` and `tools/hrnet_gate.py`.

The AUTSL gate/CORAL pools (multi-source, leak-free `noautsl` / `autsltrain`) and the true
monolingual pools are built by `scripts/hpc/audit_reruns_2026-10/prep.py`, which asserts that no
clip of a target's validation or test signers enters that target's pool.

## Pretraining and fine-tuning

```bash
# backbones (BlockGCN, no stride between stages, mask ratio 0.3, K=64, epoch-30 backbone is used)
python main_pretrain.py --config config/bdsl_shubert_pretrain_bdsl_asl.yaml --seed 0     # pool
python main_pretrain.py --config config/bdsl_shubert_pretrain_xlingual.yaml --seed 0     # xling (GRL 0.1 + contrast 0.1)
python main_pretrain.py --config config/bdsl_shubert_pretrain_bdsl_only.yaml --seed 0    # mono (BdSL)
python main_pretrain.py --config config/p1fix/bdsl_shubert_pretrain_xlingual_v2.yaml --seed 0   # v2 arms

# 10%-label fine-tune of one backbone (lr 0.01, 100 epochs, subset seed 42, no clipping), then the
# best-validation checkpoint scored once on the held-out test signers
python tools/run_resumable_seed.py --config config/lsa64_si_lr01_f10.yaml --seed 0 \
    --exp lsa64_si_lr01_f10_xling_seed0 \
    --init-weights work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt \
    --test-config config/lsa64_si_test.yaml --test-exp lsa64_si_lr01_f10_xling_seed0_test
```

A pretraining seed `p > 0` is passed as `--seed p` with the experiment named
`<pretrain config>_pseed<p>`; fine-tunes of it are named `<stem>_lr01_f10_<arm>_ps<p>_seed<s>` (v1) and
`<stem>_f10_v2_<arm>_ps<p>_seed<s>` (v2). The SLURM launchers that produced the paper's runs are in
`scripts/hpc/` (see its README): `slurm_p1_pool_pretrain_seed.sbatch` / `slurm_p1_pretrain_seed.sbatch`
(pretraining seeds), `slurm_p1_law_pseed_ft{,_packed}.sbatch` (all law fine-tunes of one backbone),
`slurm_p1_pseed_ft.sbatch` (BdSL headline), `p1_fix/` (the whole v2 rerun, steps 1-6),
`audit_reruns_2026-10/` (AUTSL leak-free and true-mono reruns), `slurm_lambdad_coral_*.sbatch`,
`slurm_ft_hrnet.sbatch` and `slurm_p1_ckpt_predictor.sbatch`.

## Distances

```bash
python tools/compute_lang_distance.py                 # v1: published one-pass MMD (400 target vs 800 source clips)
python tools/compute_lang_distance_v2.py --out docs/p1_fix_2026-10/step2/distances_v2.json --draws 20
python tools/p1_fix/distance_table.py > distances_v2.md   # v1 -> v2 decomposition (Table: distance steps)
python tools/p1_refit/realpool_dist.py                 # v1 descriptor against the real pool (not pre-registered)
python tools/distance_robustness.py; python tools/scaleinv_mmd_test.py; python tools/articulator_distance.py
```

## Pre-registered analyses

| study | pre-registration | analysis | archived output |
|---|---|---|---|
| 7-seed refit (v1 backbones) | `docs/p1_refit_2026-10/PREREG.md` (+ `PREREG.sha256`, `PROVENANCE.md`) | `tools/p1_refit/refit.py`, `report.py` | `refit_results.json`, `RESULTS.md`, `pooled_gains.json` |
| v2 rerun (fixed pipeline) | `docs/p1_fix_2026-10/PREREG.md`, `PREREG_addendum_A_distances.md` (+ hashes, `PROVENANCE.md`) | `tools/p1_fix/refit_v2.py` | `results/refit_v2.json`, `RESULTS.md`, `LOG.md` |

```bash
python tools/p1_refit/refit.py --pseeds 0-6 --out results/refit_v1 --alt-dist docs/p1_refit_2026-10/realpool_distances.json
python tools/p1_refit/report.py --json results/refit_v1/refit_results.json --md results/refit_v1/RESULTS.md
python tools/p1_fix/refit_v2.py --out results/refit_v2
```

To analyse your own runs, pass `--csv results_final.csv` (the live log that `main.py` appends to)
or `--csv results_p1fix_v2.csv`, and keep `work_dir/` so completeness is checked from checkpoints.
The pre-registration files are distributed read-only. The v2 `PREREG.md` is byte-identical to the
hash-locked original. The v1 `PREREG.md` and the v2 Addendum A are copies in which cluster job numbers
were replaced by labels (`J01`-`J45`); `PROVENANCE.md` in each directory gives both hashes.

## Integrity checks

| check | command |
|---|---|
| a run counts only if it reached its final epoch (checkpoints are pruned, so a `model-best.pt` proves nothing) | `python tools/check_run_complete.py work_dir/<run>`; every fine-tune of both analyses: `--p1-sweep` |
| skeleton geometry of a corpus (body/hand slots, left/right, scale, motion) | `python tools/check_pose_geometry.py --npz-dir <cache> --name <corpus>` |
| missing joints are stored as (0, 0): mask them before speeds and distances | `tools/compute_lang_distance_v2.py`, `preprocessing/compute_pretrain_targets_v2.py` |
| pretraining pools exclude every target's validation/test signers | `scripts/hpc/audit_reruns_2026-10/prep.py` (assertions), `tests/test_ssl_pool_no_leak.py` |
| unit and smoke tests (CPU, no data) | `make test` |

Checkpoints are selected on validation; test-signer numbers score the best-validation checkpoint
once. The law quantity `G` itself is a best-epoch validation quantity, as originally claimed.

## Project structure

```
.
├── main.py                 fine-tune / evaluate one config (YAML -> argparse, dotted-path model loading)
├── main_pretrain.py        masked-pose SSL pretraining (pool / xling / mono / gate / CORAL arms)
├── model/                  BlockGCN backbone, SHuBERT-style pretrainer, distance gate + CORAL
├── feeders/                skeleton feeder (10% stratified subsets) and the SSL pool feeder
├── graph/                  27-node sign skeleton (and COCO-17 for the HRNet control)
├── preprocessing/          pose extraction, per-language bundlers and signer splits, pools, k-means targets
├── tools/                  distances, checks, transferability baselines, reproduce.py
│   ├── p1_refit/           pre-registered 7-seed refit
│   ├── p1_fix/             pre-registered v2 rerun (refit_v2.py) and audit-step tools
│   ├── p1_ckpt_predictor/  checkpoint-aware predictor study
│   └── paper/              figure, null-simulation and exact-REML scripts
├── config/                 pretraining and fine-tuning configs (config/p1fix/: the v2 rerun)
├── splits/                 fixed signer-independent splits
├── results/                archived run logs, completion list, LogME values, provenance outputs
├── docs/                   pre-registrations, results, logs and program notes (index: docs/README.md)
├── scripts/hpc/            SLURM launchers of the paper's runs
└── tests/                  CPU unit and smoke tests
```

Development: `make check` runs ruff, the tests and the no-data reproduction, as CI does
(`.github/workflows/ci.yml`). See `CONTRIBUTING.md`.

## Source datasets

If you use this code, cite the datasets you use, as listed in the Data section (BdSLW60, BdSLW401,
LSA64, AUTSL, KArSL, GSL, LSFB-ISOL, MINDS-Libras, WL-BISINDO, WLASL, ASL Citizen, NTU RGB+D, HMDB51,
UCF101, FineGym), and BlockGCN (Zhou et al., CVPR 2024) and SHuBERT (Gueuwou et al., ACL 2025) for the
encoder and the pretraining objective.

## Citation

Citation: to be added after review.

## License

The code in this repository is released under the [MIT License](LICENSE). The license covers the
code only. The datasets keep their own terms (for example, BdSLW60 is CC BY-NC-ND 4.0 and BdSLW401 is
governed by its owners' data-use agreement), and pose sequences, labels, k-means targets or weights
derived from any of them are not covered by the MIT License.
