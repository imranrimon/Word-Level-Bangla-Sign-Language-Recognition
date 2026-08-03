#!/bin/bash -l
# =====================================================================
# WVU DollySods — AUTSL 3-seed signer-independent baseline (from scratch),
# on AUTSL's OFFICIAL signer-disjoint split. Trains seeds 0-2 (each -> its own
# work_dir/autsl_si_seed<N>/ + -best.pt selected on official val), then evaluates
# each best model on the official held-out TEST signers.
# Pre-staged with a dependency so it fires only after extraction succeeds:
#     sbatch --dependency=afterok:<extract_jobid> scripts/hpc/slurm_autsl_multiseed.sbatch
# AUTSL is ~28k train clips -> much longer than LSA64 (allow up to 47h).
# Aggregate: python tools/summarize_seeds.py --csv results_final.csv --markdown
# =====================================================================
#SBATCH -J autsl_multiseed
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 47:00:00
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

# Guard: don't train if the extraction bundle is missing/incomplete.
for s in train val test; do
  if [ ! -f "data/autsl_si/${s}_data.npy" ]; then
    echo "FATAL: data/autsl_si/${s}_data.npy missing — extraction incomplete"; exit 1
  fi
done

# 1) Train 3 seeds (each saves ...autsl_si_seed<N>_model-best.pt); resumable.
python -u tools/run_multiseed.py --single config/autsl_si.yaml --seeds 0 1 2 --skip-existing

# 2) Held-out TEST-signer eval per seed (official test signers), tagged per seed.
for s in 0 1 2; do
  W="work_dir/autsl_si_seed${s}/autsl_si_seed${s}_model-best.pt"
  if [ -f "$W" ]; then
    echo "=== AUTSL TEST-signer eval, seed ${s} ==="
    python -u main.py --config config/autsl_si_test.yaml --phase test \
      --weights "$W" -Experiment_name "autsl_si_test_seed${s}"
  else
    echo "WARN: $W not found; skipping seed ${s} test eval"
  fi
done

echo "=== done. aggregate with tools/summarize_seeds.py ==="
