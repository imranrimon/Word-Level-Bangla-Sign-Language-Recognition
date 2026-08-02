#!/bin/bash
# =====================================================================
# LOSO headline variance as a SLURM JOB ARRAY — the HPC payoff: all 11
# leave-one-signer-out folds run IN PARALLEL (one GPU each) instead of
# sequentially, turning ~9 GPU-days into ~1 wall-clock day.
# Submit:  sbatch scripts/hpc/slurm_loso_array.sbatch
# Each array task trains one fold; run_loso.py regenerates that fold's
# data from data/bdsl_cache on first use (guard against races: see NOTE).
# =====================================================================
#SBATCH --job-name=slgt_loso
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --account=<ALLOCATION>
#SBATCH --array=0-10                       # 11 folds = 11 signers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out   # %A=arrayjobid, %a=task index

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load anaconda3 2>/dev/null || true
module load cuda 2>/dev/null || true
source activate bdsl_graph
export PYTHONIOENCODING=utf-8

# 11 full-vocab signers -> one per array index
SIGNERS=(1 4 5 6 8 9 11 12 2 13 15)
TS=${SIGNERS[$SLURM_ARRAY_TASK_ID]}

# NOTE: to avoid 11 tasks regenerating fold data simultaneously, pre-build the
# folds ONCE before submitting this array (fast, CPU-only):
#   python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
#       --test-signers 1 4 5 6 8 9 11 12 2 13 15 --seeds 0 --skip-existing --prepare-only
# (if --prepare-only isn't available, just run fold U01 alone first, then submit.)

echo "LOSO fold: test signer U$TS"
python -u tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
    --test-signers "$TS" --seeds 0 1 2 --skip-existing
