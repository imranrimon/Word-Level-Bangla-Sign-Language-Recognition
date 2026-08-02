#!/bin/bash
# =====================================================================
# Option C (T5) — SSL fine-tune. The two masked-SSL BlockGCN backbones are
# already pretrained (transfer pretrained_epoch30.pt per README §2, or re-run
# main_pretrain.py per HPC_LAUNCH_GUIDE §G2). This fine-tunes each on BdSLW60-SI
# x 3 seeds. THE decision gate: cross-lingual (bdsl_asl) must beat monolingual
# (bdsl_only) by >=3 pp on SI test (see docs/TOPTIER_NEURIPS_ICLR_PLAN.md).
# Submit:  sbatch scripts/hpc/slurm_ssl_finetune.sbatch
# =====================================================================
#SBATCH --job-name=slgt_ssl_ft
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --account=<ALLOCATION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load anaconda3 2>/dev/null || true
module load cuda 2>/dev/null || true
source activate bdsl_graph
export PYTHONIOENCODING=utf-8

for v in bdsl_asl bdsl_only; do
  for s in 0 1 2; do
    exp="bdsl_block_gcn_shubert_${v}_seed${s}"
    if [ -d "work_dir/$exp/eval_results" ]; then echo "[skip] $exp"; continue; fi
    # bdsl_asl backbone dir is work_dir/bdsl_shubert_pretrain (no suffix)
    if [ "$v" = "bdsl_asl" ]; then bb=work_dir/bdsl_shubert_pretrain; else bb=work_dir/bdsl_shubert_pretrain_bdsl_only; fi
    python -u main.py --config config/bdsl_block_gcn_si.yaml --seed "$s" \
      -Experiment_name "$exp" \
      --weights "$bb/pretrained_epoch30.pt" --ignore-weights fc.weight fc.bias
  done
done
echo "SSL fine-tunes done"
