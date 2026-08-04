#!/bin/bash -l
# =====================================================================
# WVU DollySods — LOSO headline variance as a JOB ARRAY. All 11
# leave-one-signer-out folds run IN PARALLEL (one GPU each) instead of
# sequentially -> ~9 GPU-days becomes ~1 wall-clock day. The 24GB
# dscog[005-030] pool (26 nodes) is ideal here for availability.
# Submit from ~/SLGTformer:  sbatch scripts/hpc/slurm_loso_array.sbatch
#
# NOTE: pre-build the fold data ONCE before submitting so 11 tasks don't
# regenerate it at the same time (CPU-only, fast) — run fold U01 alone first:
#   python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
#       --test-signers 1 --seeds 0 --skip-existing
# then submit this array for the rest.
# =====================================================================
#SBATCH -J slgt_loso
#SBATCH -a 0-10                      # 11 folds
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

# NB: no set -u (conda MKL activation not nounset-safe)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

SIGNERS=(1 4 5 6 8 9 11 12 2 13 15)
TS=${SIGNERS[$SLURM_ARRAY_TASK_ID]}
echo "LOSO fold: test signer U$TS"
python -u tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
    --test-signers "$TS" --seeds 0 1 2 --skip-existing
