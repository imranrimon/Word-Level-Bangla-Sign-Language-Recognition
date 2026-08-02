#!/bin/bash
# =====================================================================
# SI baseline sweep (Option A) + SLGTFormer ablations, one GPU, sequential.
# Submit:  sbatch scripts/hpc/slurm_si_sweep.sbatch
# EDIT the #SBATCH lines marked <...> for your cluster (partition, account,
# time limit). `scontrol show partition` / `sinfo` list what's available.
# =====================================================================
#SBATCH --job-name=slgt_si_sweep
#SBATCH --partition=<GPU_PARTITION>       # e.g. gpu, a100, v100
#SBATCH --account=<ALLOCATION>            # your project/allocation code (omit if not required)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%x_%j.out     # stdout+stderr (%x=name, %j=jobid)
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- environment (adjust module names to your cluster) ---
module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true
module load cuda 2>/dev/null || true
source activate bdsl_graph
export PYTHONIOENCODING=utf-8            # HPC locale is often ASCII (guide §4)

python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# Phase 1 — Option A pose table (11 archs x 3 seeds), skip-existing = resumable
python -u tools/run_multiseed.py --config experiments_si_main.yaml   --seeds 0 1 2 --skip-existing
# Phase 2 — SLGTFormer sub-module ablations
python -u tools/run_multiseed.py --config experiments_si_ablations.yaml --seeds 0 1 2 --skip-existing
# Phase 3 — Option B DINOv2 row (needs data/bdsl_si_dino; re-extract on HPC first if not transferred)
if [ -f data/bdsl_si_dino/train_data.npy ]; then
  python -u tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2 --skip-existing
fi

python -u tools/summarize_seeds.py --csv results_final.csv --markdown > results/T1_stage_A.md
echo "SI sweep done"
