#!/bin/bash -l
# =====================================================================
# WVU DollySods — pretrain ONE SSL backbone from a config (env PRETRAIN_CONFIG).
#   PRETRAIN_CONFIG=config/<name>.yaml sbatch scripts/hpc/slurm_ssl_pretrain_one.sbatch
# -> work_dir/<Experiment_name>/pretrained_epoch30.pt
# =====================================================================
#SBATCH -J ssl_pretrain_one
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
: "${PRETRAIN_CONFIG:?set PRETRAIN_CONFIG}"
python -c "import torch;print('cuda',torch.cuda.is_available())"
python -u main_pretrain.py --config "$PRETRAIN_CONFIG" --seed 0
echo "done: $PRETRAIN_CONFIG"
