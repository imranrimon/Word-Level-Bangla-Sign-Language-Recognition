#!/bin/bash -l
# =====================================================================
# WVU DollySods — LSA64 (Argentine SL) signer-independent train + test.
# Trains BlockGCN (select on val signer 8, saving a -best.pt), then evaluates
# that best model on the held-out TEST signers (9-10) = canonical SI number.
# 1 GPU. Submit from ~/SLGTformer after data/lsa64_si/ exists:
#     sbatch scripts/hpc/slurm_lsa64_train.sbatch          # seed 0
# NB multi-seed: direct main.py reuses work_dir/lsa64_si, so seeds clobber —
# use `tools/run_multiseed.py --single config/lsa64_si.yaml --seeds 0 1 2`.
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

# 1) Train (selects best on val signer 8, writes ...lsa64_si_model-best.pt)
python -u main.py --config config/lsa64_si.yaml --seed "${SEED:-0}"

# 2) Evaluate the best model on the held-out TEST signers (9-10) = SI headline
BEST=work_dir/lsa64_si/lsa64_si_model-best.pt
if [ -f "$BEST" ]; then
  echo "=== Held-out TEST-signer eval (signers 9-10) ==="
  python -u main.py --config config/lsa64_si_test.yaml --phase test --weights "$BEST"
else
  echo "WARN: $BEST not found; skipping test-signer eval"
fi
