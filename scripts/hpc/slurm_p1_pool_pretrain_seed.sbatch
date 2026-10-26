#!/bin/bash -l
# =====================================================================
# P1 item6 (2026-10-10): PRETRAINING-SEED replication of the POOL backbone used by
# every point of the distance-gating law (gain = xling - pool, @10% val Top-1).
# Mirrors scripts/hpc/slurm_p1_pretrain_seed.sbatch (item5, mono/xling) exactly:
#   pool = config/bdsl_shubert_pretrain_bdsl_asl.yaml (data/ssl_pool_manifest_bdsl_asl.json,
#          data/pretrain_kmeans_targets_bdsl_asl.npz -- the SAME files the seed-0 backbone used)
# Only the pretraining seed changes (init, masks, crops, shuffle). The original was
# --seed 0 (main_pretrain.py default). One array task per seed (1, 2) so each gets its
# own GPU and the per-seed fine-tune arrays can chain on afterok:<job>_<seed>.
# Resume-safe: main_pretrain.py auto-resumes from train_state.pt; a finished run
# (pretrained_epoch30.pt present) is skipped, never restarted.
#   sbatch -a 1-2 scripts/hpc/slurm_p1_pool_pretrain_seed.sbatch
# =====================================================================
#SBATCH -J p1_pool_pseed_pt
#SBATCH -a 1-2
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 64G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 8:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# 8-h chunks (2026-10-12): Slurm sends USR1 15 min before the limit; requeue so every run
# resumes from its last checkpoint (a plain TIMEOUT is final; --requeue only covers node failures).
trap 'echo "[chunk] $(date "+%F %T") time limit near -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
# work_dir writability guard (shared file systems may mount lazily)
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
# GPU-health guard
if ! python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.zeros(1).cuda().add_(1).item()==1.0) else 1)" 2>/dev/null; then
  echo "GPU unusable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1
fi

# ARM=pool (default) | xling. 2026-10-11: generalised for the 7-seed scale-up (seeds 3-6, both arms);
# xling = config/bdsl_shubert_pretrain_xlingual.yaml, i.e. exactly item5's slurm_p1_pretrain_seed.sbatch
# ARM=xling recipe (same manifest/targets as pool), one seed per array task instead of two per GPU.
#   sbatch -a 3-6 --export=ALL,ARM=xling scripts/hpc/slurm_p1_pool_pretrain_seed.sbatch
case "${ARM:-pool}" in
  pool)  EXP=bdsl_shubert_pretrain_bdsl_asl ;;
  xling) EXP=bdsl_shubert_pretrain_xlingual ;;
  *) echo "ARM must be pool|xling (got '$ARM')"; exit 2 ;;
esac
CFG="config/${EXP}.yaml"
S=${SLURM_ARRAY_TASK_ID:?array task id = pretraining seed}
WD="work_dir/${EXP}_pseed${S}"
echo "P1 ${ARM:-pool} pretrain-seed replication: seed $S ($CFG) -> $WD on $(hostname)"
if [ -f "$WD/pretrained_epoch30.pt" ]; then echo "[done] $WD"; exit 0; fi
python -u main_pretrain.py --config "$CFG" --seed "$S" --work-dir "$WD" \
  >> "slurm_logs/${EXP}_pseed${S}.log" 2>&1 &
wait $!   # backgrounded so the USR1 requeue trap fires on time
[ -f "$WD/pretrained_epoch30.pt" ] || { echo "missing $WD/pretrained_epoch30.pt"; exit 1; }
echo "=== ${ARM:-pool} pretrain seed $S done ==="
