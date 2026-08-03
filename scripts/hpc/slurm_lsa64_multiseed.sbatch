#!/bin/bash -l
# =====================================================================
# WVU DollySods — LSA64 3-seed signer-independent baseline (from scratch).
# Trains seeds 0-2 (each -> its own work_dir/lsa64_si_seed<N>/ + -best.pt),
# then evaluates each best model on the held-out TEST signers (9-10).
# 1 GPU, sequential. Submit from ~/SLGTformer:
#     sbatch scripts/hpc/slurm_lsa64_multiseed.sbatch
# Aggregate afterwards: python tools/summarize_seeds.py --csv results_final.csv --markdown
# =====================================================================
#SBATCH -J lsa64_multiseed
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 12:00:00
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

# 1) Train 3 seeds (each saves ...lsa64_si_seed<N>_model-best.pt); resumable.
python -u tools/run_multiseed.py --single config/lsa64_si.yaml --seeds 0 1 2 --skip-existing

# 2) Held-out TEST-signer eval per seed (canonical SI number, tagged per seed).
for s in 0 1 2; do
  W="work_dir/lsa64_si_seed${s}/lsa64_si_seed${s}_model-best.pt"
  if [ -f "$W" ]; then
    echo "=== TEST-signer eval, seed ${s} ==="
    python -u main.py --config config/lsa64_si_test.yaml --phase test \
      --weights "$W" -Experiment_name "lsa64_si_test_seed${s}"
  else
    echo "WARN: $W not found; skipping seed ${s} test eval"
  fi
done

echo "=== done. aggregate with tools/summarize_seeds.py ==="
