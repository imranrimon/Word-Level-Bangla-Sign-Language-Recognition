#!/bin/bash -l
# =====================================================================
# WVU DollySods — LSA64 (Argentine SL) signer-independent training.
# 1 GPU. Submit from ~/SLGTformer after data/lsa64_si/ exists:
#     sbatch scripts/hpc/slurm_lsa64_train.sbatch          # seed 0
#     SEED=1 sbatch scripts/hpc/slurm_lsa64_train.sbatch   # other seeds
# =====================================================================
#SBATCH -J lsa64_train
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# NB: no `set -u` — conda's MKL activation scripts reference unbound vars.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

python -u main.py --config config/lsa64_si.yaml --seed "${SEED:-0}"
