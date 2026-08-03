#!/bin/bash -l
# =====================================================================
# WVU DollySods — SSL pretraining: the POOL-ONLY ablation backbone.
# ASL+BdSL pool with the cross-lingual MECHANISM OFF (lambda_adv=lambda_contrast=0)
# = masked pose SSL on mixed data, no invariance/alignment. Completes the
# ablation ladder: scratch < mono < POOL < mechanism(xlingual).
#     sbatch scripts/hpc/slurm_ssl_pretrain_pool.sbatch
# -> work_dir/bdsl_shubert_pretrain_bdsl_asl/pretrained_epoch30.pt
# =====================================================================
#SBATCH -J ssl_pretrain_pool
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 36:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
python -c "import torch;print('cuda',torch.cuda.is_available())"

python -u main_pretrain.py --config config/bdsl_shubert_pretrain_bdsl_asl.yaml --seed 0
ls -la work_dir/bdsl_shubert_pretrain_bdsl_asl/pretrained_epoch30.pt
