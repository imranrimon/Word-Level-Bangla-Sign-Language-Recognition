#!/bin/bash -l
# =====================================================================
# Continuous lambda(d) gate and CORAL alignment: masked-pose SSL pretrains for the AUTSL
# gate/CORAL ablations (paper App. "AUTSL: leak-free gate/CORAL ablations", original rows).
#
#   idx cfg                                              purpose
#   0   bdsl_shubert_pretrain_lambdad_autsl.yaml         lambda(d) sigmoid gate, AUTSL target
#   1   bdsl_shubert_pretrain_coral_autsl.yaml           CORAL alignment, AUTSL target
#
# NOTE: these original runs used the multi-source pool data/ssl_pool_manifest_multisrc.json,
# which contains AUTSL validation and test clips (the leak found by the audit). The leak-free
# reruns are scripts/hpc/audit_reruns_2026-10/slurm_audit_reruns.sbatch (JOBSET=A and C).
#
# Each produces work_dir/<Experiment_name>/pretrained_epoch30.pt for a downstream
# @10% fine-tune (slurm_lambdad_coral_finetune.sbatch). main_pretrain.py auto-resumes from
# work_dir/<exp>/train_state.pt on requeue.
#
# Prerequisites: data/ssl_pool_manifest_multisrc.json + data/pretrain_kmeans_targets_multisrc.npz
# (preprocessing/build_ssl_pool_manifest.py, preprocessing/compute_pretrain_targets.py
# --feature-mode pose_motion).
#   sbatch scripts/hpc/slurm_lambdad_coral_pretrain.sbatch
# =====================================================================
#SBATCH -J lamd_coral_pre
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
#SBATCH --array=0-1
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

CFGS=(
  config/bdsl_shubert_pretrain_lambdad_autsl.yaml
  config/bdsl_shubert_pretrain_coral_autsl.yaml
)
CFG="${CFGS[$SLURM_ARRAY_TASK_ID]}"
[ -n "$CFG" ] || { echo "no config for idx $SLURM_ARRAY_TASK_ID -> no-op"; exit 0; }
[ -f "$CFG" ] || { echo "config $CFG missing"; exit 1; }
EXP=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['Experiment_name'])" "$CFG")
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

# --- idempotent skip: backbone already produced -> nothing to do ---
if [ -f "$WD/pretrained_epoch30.pt" ]; then
  echo "=== $EXP already has pretrained_epoch30.pt; skip ==="
  exit 0
fi

# --- guard: shared multi-source pool manifest + k-means targets must exist ---
MANJSON=data/ssl_pool_manifest_multisrc.json
TGTNPZ=data/pretrain_kmeans_targets_multisrc.npz
if [ ! -f "$MANJSON" ] || [ ! -f "$TGTNPZ" ]; then
  echo "BLOCKER [$EXP]: missing $MANJSON and/or $TGTNPZ."
  echo "  Build the multi-source pool with preprocessing/build_ssl_pool_manifest.py +"
  echo "  compute_pretrain_targets.py (--feature-mode pose_motion)."
  exit 1
fi

echo "=== λ(d)/CORAL SSL pretrain: idx=$SLURM_ARRAY_TASK_ID exp=$EXP cfg=$CFG ==="
python -u main_pretrain.py --config "$CFG" --seed 0   # auto-resumes from train_state.pt
echo "=== done $EXP -> $WD/pretrained_epoch30.pt ==="
