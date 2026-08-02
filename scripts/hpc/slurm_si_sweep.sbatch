#!/bin/bash -l
# =====================================================================
# WVU DollySods — SI baseline sweep (Option A) + SLGTFormer ablations,
# 1 GPU, sequential. Submit from ~/SLGTformer:
#     sbatch scripts/hpc/slurm_si_sweep.sbatch
# Before submitting: pick a partition (and optionally an idle node) with
#     sinfo -s          # partitions + idle counts
#     scontrol show node dscog033
# The models are small (batch 16-32) so a 24GB dscog[005-030] node is plenty;
# 46GB dscog[031-034] gives headroom for the DINOv2 row.
# =====================================================================
#SBATCH -J slgt_si_sweep
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
##SBATCH -w dscog033                # <-- optional: pin a specific idle node
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results

# activate the conda env (miniconda in $HOME; source is required in a batch shell)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8            # DollySods locale is often ASCII
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# Phase 1 — Option A pose table (11 archs x 3 seeds); skip-existing = resumable
python -u tools/run_multiseed.py --config experiments_si_main.yaml    --seeds 0 1 2 --skip-existing
# Phase 2 — SLGTFormer sub-module ablations
python -u tools/run_multiseed.py --config experiments_si_ablations.yaml --seeds 0 1 2 --skip-existing
# Phase 3 — Option B DINOv2 row (transferred, or re-extract on HPC first)
if [ -f data/bdsl_si_dino/train_data.npy ]; then
  python -u tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2 --skip-existing
fi

python -u tools/summarize_seeds.py --csv results_final.csv --markdown > results/T1_stage_A.md
echo "SI sweep done"
