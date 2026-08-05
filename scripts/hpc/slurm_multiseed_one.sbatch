#!/bin/bash -l
# =====================================================================
# WVU DollySods — run one config across 3 seeds with an explicit name.
#   CONFIG=<stem> NAME=<experiment-name-prefix> [SEEDS="0 1 2"] sbatch scripts/hpc/slurm_multiseed_one.sbatch
# Logs <NAME>_seed<N> to results_final.csv. Used for clean arch re-runs and
# the feature-isolation (pose vs DINOv2) runs.
# =====================================================================
#SBATCH -J multiseed_one
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
: "${CONFIG:?set CONFIG}"; : "${NAME:?set NAME}"; SEEDS="${SEEDS:-0 1 2}"
for s in $SEEDS; do
  echo "=== ${NAME}_seed${s} (config/${CONFIG}.yaml) ==="
  python -u main.py --config "config/${CONFIG}.yaml" --seed "$s" -Experiment_name "${NAME}_seed${s}"
done
echo "done: ${NAME}"
