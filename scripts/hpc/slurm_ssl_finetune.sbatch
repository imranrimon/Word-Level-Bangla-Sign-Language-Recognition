#!/bin/bash -l
# =====================================================================
# WVU DollySods — Option C (T5) SSL fine-tunes. 3-way: pool-only (bdsl_asl,
# ablation), monolingual (bdsl_only, ablation), and the PROPOSED cross-lingual
# mechanism (xlingual — RE-PRETRAIN first with slurm_ssl_pretrain, then finetune).
# THE gate: xlingual must beat pool-only / from-scratch by >=3 pp on SI test.
# Submit from ~/SLGTformer:  sbatch scripts/hpc/slurm_ssl_finetune.sbatch
# =====================================================================
#SBATCH -J slgt_ssl_ft
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
##SBATCH -w dscog033
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

# backbone dir per variant: bdsl_asl -> work_dir/bdsl_shubert_pretrain (no suffix)
for v in bdsl_asl bdsl_only; do
  for s in 0 1 2; do
    exp="bdsl_block_gcn_shubert_${v}_seed${s}"
    if [ -d "work_dir/$exp/eval_results" ]; then echo "[skip] $exp"; continue; fi
    if [ "$v" = "bdsl_asl" ]; then bb=work_dir/bdsl_shubert_pretrain; else bb=work_dir/bdsl_shubert_pretrain_bdsl_only; fi
    python -u main.py --config config/bdsl_block_gcn_si.yaml --seed "$s" \
      -Experiment_name "$exp" \
      --weights "$bb/pretrained_epoch30.pt" --ignore-weights fc.weight fc.bias
  done
done
echo "SSL fine-tunes done"
