#!/bin/bash -l
# =====================================================================
# WVU DollySods — THE GATE: SI fine-tune on BdSLW60, 3 conditions x 3 seeds.
#   scratch | monolingual-SSL init | cross-lingual-SSL init
# Each condition selects best on the SI val signer (saving -best.pt) and logs
# to results_final.csv as bdsl_gate_<cond>_seed<N>. Decision: xling - mono >= 3 pp.
# Pre-stage after pretraining:
#     sbatch --dependency=afterok:<pretrain_jobid> scripts/hpc/slurm_gate_finetune.sbatch
# =====================================================================
#SBATCH -J gate_finetune
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

# NB: no `set -u` — conda's MKL activation scripts reference unbound vars.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

XLING=work_dir/bdsl_shubert_pretrain_xlingual/pretrained_epoch30.pt
MONO=work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt
for f in "$XLING" "$MONO"; do
  [ -f "$f" ] || { echo "FATAL: backbone $f missing — pretraining incomplete"; exit 1; }
done

for s in 0 1 2; do
  echo "=== seed $s: scratch ==="
  python -u main.py --config config/bdsl_block_gcn_si.yaml --seed "$s" \
    -Experiment_name "bdsl_gate_scratch_seed${s}"

  echo "=== seed $s: monolingual-SSL init ==="
  python -u main.py --config config/bdsl_block_gcn_si.yaml --seed "$s" \
    --weights "$MONO" --ignore-weights fc.weight fc.bias \
    -Experiment_name "bdsl_gate_mono_seed${s}"

  echo "=== seed $s: cross-lingual-SSL init ==="
  python -u main.py --config config/bdsl_block_gcn_si.yaml --seed "$s" \
    --weights "$XLING" --ignore-weights fc.weight fc.bias \
    -Experiment_name "bdsl_gate_xling_seed${s}"
done

echo "=== GATE readout (val-selected SI Top-1) ==="
python tools/summarize_seeds.py --csv results_final.csv --markdown 2>/dev/null | grep -iE 'gate|Experiment|---' || true
echo "Decision: mean(xling) - mean(mono) >= 3 pp ?"
