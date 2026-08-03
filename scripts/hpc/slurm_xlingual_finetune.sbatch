#!/bin/bash -l
# =====================================================================
# WVU DollySods — cross-lingual GENERALIZATION fine-tune (Phase 3).
# Fine-tunes the SSL backbones on a target's SI split. Parameterized by env:
#   TARGET       train config stem (required), e.g. lsa64_si / autsl_si / bdsl_block_gcn_si
#   CONDS        space-list of conditions (default "mono xling pool")
#   SEEDS        space-list (default "0 1 2")
#   TEST_CONFIG  optional test-split config stem for held-out-signer eval
# Logs <TARGET>_<cond>_seed<N> (+ _test). Examples:
#   TARGET=lsa64_si TEST_CONFIG=lsa64_si_test sbatch --dependency=afterok:<bb> scripts/hpc/slurm_xlingual_finetune.sbatch
#   TARGET=bdsl_block_gcn_si CONDS=pool        sbatch --dependency=afterok:<bb> scripts/hpc/slurm_xlingual_finetune.sbatch
# =====================================================================
#SBATCH -J xling_finetune
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 47:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

TARGET="${TARGET:?set TARGET (config stem)}"
CONDS="${CONDS:-mono xling pool}"
SEEDS="${SEEDS:-0 1 2}"

XLING=work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt
MONO=work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt
POOL=work_dir/bdsl_shubert_pretrain_bdsl_asl/pretrained_epoch30.pt
bb_for() { case "$1" in mono) echo "$MONO";; xling) echo "$XLING";; pool) echo "$POOL";; *) echo "";; esac; }

echo "TARGET=$TARGET CONDS='$CONDS' SEEDS='$SEEDS' TEST_CONFIG='${TEST_CONFIG:-}'"
for s in $SEEDS; do
  for cond in $CONDS; do
    bb=$(bb_for "$cond")
    if [ -z "$bb" ] || [ ! -f "$bb" ]; then echo "FATAL: backbone for '$cond' missing ($bb)"; exit 1; fi
    exp="${TARGET}_${cond}_seed${s}"
    echo "=== $exp (init from $cond) ==="
    python -u main.py --config "config/${TARGET}.yaml" --seed "$s" \
      --weights "$bb" --ignore-weights fc.weight fc.bias -Experiment_name "$exp"
    if [ -n "${TEST_CONFIG:-}" ]; then
      W="work_dir/${exp}/${exp}_model-best.pt"
      if [ -f "$W" ]; then
        echo "=== $exp: held-out test eval ==="
        python -u main.py --config "config/${TEST_CONFIG}.yaml" --phase test \
          --weights "$W" -Experiment_name "${exp}_test"
      fi
    fi
  done
done
echo "=== done: $TARGET generalization fine-tune ==="
