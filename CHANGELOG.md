# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-25

Initial public release accompanying the replication paper (under review).

### Added
- Masked-pose SSL pretraining (`main_pretrain.py`, `model/shubert_pretrain.py`,
  `feeders/pretrain_feeder.py`) on a BlockGCN encoder with the pool, cross-lingual (gradient-reversal
  language adversary + shared-codebook contrastive term), monolingual, distance-gated, lambda(d) and
  CORAL arms; 10%-label fine-tuning harness (`main.py`).
- 27-keypoint MediaPipe bundlers and signer-independent splits for BdSLW60, LSA64, AUTSL, KArSL
  (v1 and v2a/v2b), GSL, LSFB-ISOL, MINDS-Libras, WL-BISINDO and ASL Citizen; pose caches and pool
  manifests for BdSLW401, BdSLW102a and WLASL (v1 converter and the corrected v2 converter); HRNet
  skeleton bundles for the NTU/HMDB51/UCF101/FineGym control.
- Pose-motion distances: the published one-pass MMD and the corrected v2 distance, robustness sweeps,
  transferability baselines (LogME, energy, Frechet, proxy-A, kNN).
- The two pre-registered analyses with their pre-registrations, hashes, provenance notes, logs and
  archived outputs: the 7-pretraining-seed refit (`tools/p1_refit/`, `docs/p1_refit_2026-10/`) and
  the corrected v2 rerun (`tools/p1_fix/`, `docs/p1_fix_2026-10/`).
- Checkpoint-aware predictor study (`tools/p1_ckpt_predictor/`), AUTSL leak-free and true-monolingual
  reruns (`scripts/hpc/audit_reruns_2026-10/`, `docs/audit_reruns_2026-10/`).
- Integrity tools: pose-geometry checker, run-completeness checker (`--p1-sweep`).
- Archived run logs restricted to the paper's experiments, the completion list, and the
  no-data reproduction of the paper's numbers and figures (`tools/reproduce.py`, `tools/paper/`).
- Packaging and tooling: `pyproject.toml`, `requirements.txt`, `environment.yml`, `Makefile`, ruff and
  pre-commit configuration, CPU-only GitHub Actions CI.
