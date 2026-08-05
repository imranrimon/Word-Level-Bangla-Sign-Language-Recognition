#!/bin/bash -l
# =====================================================================
# WVU DollySods — held-out TEST-signer eval of trained -best.pt checkpoints.
# Params (env): TEST_CONFIG (config stem for the test split), EXPS (space-list
# of experiment names whose work_dir/<exp>/<exp>_model-best.pt is evaluated).
# Logs <exp>_test to results_final.csv. Example:
#   TEST_CONFIG=bdsl_block_gcn_si_lr01_test EXPS="a b c" sbatch scripts/hpc/slurm_test_eval.sbatch
# =====================================================================
#SBATCH -J test_eval
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

TEST_CONFIG="${TEST_CONFIG:?set TEST_CONFIG}"
EXPS="${EXPS:?set EXPS (space-list of experiment names)}"
for exp in $EXPS; do
  W="work_dir/${exp}/${exp}_model-best.pt"
  if [ -f "$W" ]; then
    echo "=== TEST eval: $exp ==="
    python -u main.py --config "config/${TEST_CONFIG}.yaml" --phase test \
      --weights "$W" -Experiment_name "${exp}_test"
  else
    echo "WARN: $W missing; skip"
  fi
done
echo "=== done: test eval for $TEST_CONFIG ==="
