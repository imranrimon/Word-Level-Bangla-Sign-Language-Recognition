#!/bin/bash -l
# =====================================================================
# WVU DollySods — AUTSL (Turkish SL) MediaPipe pose extraction, OFFICIAL split.
# 36,302 clips (train 28142 / val 4418 / test 3742); CPU-bound, 32 workers.
# ~4-5 h. Resumable via data/autsl_cache/. Submit from ~/SLGTformer:
#     sbatch scripts/hpc/slurm_autsl_extract.sbatch
# =====================================================================
#SBATCH -J autsl_extract
#SBATCH -c 32
#SBATCH -n 1
#SBATCH -p gpu_2day
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# NB: no `set -u` — conda's MKL activation scripts reference unbound vars.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
python -c "import mediapipe as mp; assert hasattr(mp.solutions,'holistic'); print('mediapipe',mp.__version__)"

python -u preprocessing/generate_autsl_split_npy.py \
  --dataset-root data/autsl_download \
  --output-dir   data/autsl_si \
  --cache-dir    data/autsl_cache \
  --splits train val test \
  --max-frames 300 \
  --jobs "${SLURM_CPUS_PER_TASK:-32}"

echo "AUTSL SI bundle written to data/autsl_si/ (official signer-disjoint split)"
