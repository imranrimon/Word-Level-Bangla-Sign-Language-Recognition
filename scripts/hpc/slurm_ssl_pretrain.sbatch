#!/bin/bash -l
# =====================================================================
# WVU DollySods — cross-lingual SSL PRE-training (the proposed mechanism).
# Runs masked pose SSL with GRL language-invariance + shared-codebook
# contrastive on the BdSL+ASL pool -> work_dir/bdsl_shubert_pretrain_xlingual/
# pretrained_epoch30.pt, which slurm_ssl_finetune then fine-tunes.
# The pool-only (bdsl_asl) and monolingual (bdsl_only) backbones already exist
# (transferred in work_dir/) and serve as ablations, so only xlingual re-runs.
# Prefer a 46GB (dscog031-034) or 80GB (dscog035-036) node here.
# Submit from ~/SLGTformer:  sbatch scripts/hpc/slurm_ssl_pretrain.sbatch
# =====================================================================
#SBATCH -J slgt_ssl_pretrain
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
##SBATCH -w dscog033
#SBATCH -t 36:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

python -u main_pretrain.py --config config/bdsl_shubert_pretrain_xlingual.yaml --seed 0
echo "xlingual SSL pretrain done -> work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt"
