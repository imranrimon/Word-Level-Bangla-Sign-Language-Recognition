#!/bin/bash -l
# =====================================================================
# WVU DollySods — LSA64 (Argentine SL) MediaPipe pose extraction + SI bundle.
# CPU-bound (no GPU needed); parallelized across cores. Submit from ~/SLGTformer:
#     sbatch scripts/hpc/slurm_lsa64_extract.sbatch
# 3,200 clips at ~5-9 s/clip single-process → ~20-40 min with 16 workers.
# Resumable: per-clip .npz cache in data/lsa64_cache/ is reused on re-run.
# =====================================================================
#SBATCH -J lsa64_extract
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -p gpu_2day                 # only GPU partitions exist; run CPU-only here
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# NB: no `set -u` — conda's MKL activation scripts reference unbound vars.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

# MediaPipe on Python 3.8 must be the cp38 wheel (0.10.9); 1.0.0 breaks on 3.8.
python - <<'PY'
import mediapipe as mp
assert hasattr(mp.solutions, "holistic"), "mediapipe.solutions.holistic missing"
print("mediapipe", mp.__version__)
PY

python -u preprocessing/generate_lsa64_split_npy.py \
  --dataset-root data/lsa64_raw \
  --output-dir   data/lsa64_si \
  --cache-dir    data/lsa64_cache \
  --split-json   splits/lsa64_signer_independent.json \
  --splits train val test \
  --max-frames 300 \
  --jobs "${SLURM_CPUS_PER_TASK:-16}"

echo "LSA64 SI bundle written to data/lsa64_si/"
