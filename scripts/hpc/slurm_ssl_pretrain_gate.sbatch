#!/bin/bash -l
# =====================================================================
# WVU DollySods — SSL pretraining for THE GATE: two backbones.
#   * cross-lingual mechanism (ASL+BdSL pool + GRL language-invariance +
#     shared-codebook contrastive) -> work_dir/bdsl_shubert_pretrain_xlingual/
#   * monolingual (BdSL-only masked)                -> work_dir/bdsl_shubert_pretrain_bdsl_only/
# Each saves backbone-only pretrained_epoch{5..30}.pt. Sequential, 1 GPU.
#     sbatch scripts/hpc/slurm_ssl_pretrain_gate.sbatch
# =====================================================================
#SBATCH -J ssl_pretrain_gate
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 36:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# NB: no `set -u` — conda's MKL activation scripts reference unbound vars.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "=== [1/2] cross-lingual masked pose SSL (mechanism) ==="
python -u main_pretrain.py --config config/bdsl_shubert_pretrain_xlingual.yaml --seed 0

echo "=== [2/2] monolingual masked pose SSL (BdSL-only) ==="
python -u main_pretrain.py --config config/bdsl_shubert_pretrain_bdsl_only.yaml --seed 0

echo "backbones ready:"
ls -la work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt \
       work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt
