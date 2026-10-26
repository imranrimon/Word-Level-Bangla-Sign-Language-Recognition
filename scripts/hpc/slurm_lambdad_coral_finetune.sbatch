#!/bin/bash -l
# =====================================================================
# @10% fine-tunes of SSL backbones from a manifest (one line per array task). Used for the
# AUTSL gate / CORAL ablations (original rows of the paper's AUTSL appendix table):
#   lambdad_autsl -> AUTSL @10%  (continuous lambda(d) gate)
#   coral_autsl   -> AUTSL @10%  (CORAL alignment)
# and, with other backbone directories, for the hard-gated and uniform-weight pools.
#
# Manifest lines: CFG|BB_DIR|EXP|FRAC|SEED  (the script appends /pretrained_epoch30.pt), e.g.
#   config/autsl_si_lr01.yaml|work_dir/bdsl_shubert_pretrain_lambdad_autsl|lambdad_autsl_f10_seed0|0.1|0
#   config/autsl_si_lr01.yaml|work_dir/bdsl_shubert_pretrain_coral_autsl|coral_autsl_f10_seed0|0.1|0
# Default manifest: slurm_logs/lambdad_coral_finetune_manifest.txt (override with MAN=...).
# Idempotent (skips runs with eval_results/best_acc.pkl), resumes from the last epoch, and
# requeues while a backbone is not yet pretrained.
# =====================================================================
#SBATCH -J lamd_coral_ft
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
#SBATCH --array=0-14%8
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

MAN=${MAN:-slurm_logs/lambdad_coral_finetune_manifest.txt}
[ -f "$MAN" ] || { echo "manifest $MAN missing"; exit 1; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAN")
[ -n "$LINE" ] || { echo "no manifest line for idx $SLURM_ARRAY_TASK_ID -> no-op"; exit 0; }
IFS='|' read -r CFG BB EXP FRAC SEED <<< "$LINE" || true
[ -n "$EXP" ] || { echo "empty EXP -> no-op"; exit 0; }
WD="work_dir/${EXP}"

# --- work_dir writability guard (retry, else requeue) ---
ok=0
for i in $(seq 1 30); do
  ls work_dir/ >/dev/null 2>&1 || true
  mkdir -p "$WD" 2>/dev/null && touch "$WD/.rw_$$" 2>/dev/null && { rm -f "$WD/.rw_$$"; ok=1; break; }
  sleep 20
done
[ "$ok" = 1 ] || { echo "scratch down -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "GPU dead -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

# --- idempotent skip: already finished ---
if [ -f "$WD/eval_results/best_acc.pkl" ]; then echo "=== ALREADY DONE $EXP ==="; exit 0; fi

# --- backbone must be pretrained; requeue until it appears ---
BBCK="$BB/pretrained_epoch30.pt"
if [ ! -f "$BBCK" ]; then echo "backbone $BBCK not ready -> requeue"; scontrol requeue "$SLURM_JOB_ID"; sleep 5; exit 1; fi

# --- resume from last epoch if a partial run exists, else fresh from backbone ---
CKPT=$(ls -t "$WD/${EXP}_model-"*.pt 2>/dev/null | head -1) || true
if [ -n "$CKPT" ]; then
  EP=$(basename "$CKPT" | sed -E "s/.*_model-([0-9]+)\.pt/\1/")
  echo "=== RESUME $EXP from $CKPT @ ep $((EP+1)) ==="
  WARGS="--weights $CKPT --start-epoch $((EP+1))"
else
  echo "=== FRESH $EXP from $BBCK ==="
  WARGS="--weights $BBCK --ignore-weights fc.weight fc.bias"
fi

echo "=== λ(d)/CORAL finetune: exp=$EXP backbone=$BB frac=$FRAC seed=$SEED ==="
python main.py --config "$CFG" --seed "$SEED" -Experiment_name "$EXP" \
  $WARGS --train-fraction "$FRAC" --num-worker 8
echo "=== done $EXP ==="
