#!/bin/bash -l
# =====================================================================
# P1 item5 (2026-10-08): PRETRAINING-SEED replication of the headline BdSL @10%
# xling-vs-mono gain (+4.2pp). Re-runs the two headline SSL backbones with new
# pretraining seeds; everything else identical to the originals:
#   mono  = config/bdsl_shubert_pretrain_bdsl_only.yaml (ssl_pool_manifest_bdsl_only.json,
#           pretrain_kmeans_targets_bdsl_only.npz)
#   xling = config/bdsl_shubert_pretrain_xlingual.yaml  (ssl_pool_manifest_bdsl_asl.json,
#           pretrain_kmeans_targets_bdsl_asl.npz)
# The k-means cluster-ID targets are the SAME .npz files the seed-0 backbones used, so
# only the pretraining seed (backbone/head init, mask positions, crops, shuffle) changes.
# Originals were run with --seed 0; this script runs --seed 1 and --seed 2 concurrently
# on one GPU -> work_dir/<orig_exp>_pseed<S>/pretrained_epoch30.pt.
# Resume-safe: main_pretrain.py auto-resumes from train_state.pt; a finished run
# (pretrained_epoch30.pt present) is skipped, never restarted.
#   sbatch --export=ALL,ARM=mono  scripts/hpc/slurm_p1_pretrain_seed.sbatch
#   sbatch --export=ALL,ARM=xling scripts/hpc/slurm_p1_pretrain_seed.sbatch
# =====================================================================
#SBATCH -J p1_pseed_pt
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 64G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 47:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# work_dir writability guard (shared file systems may mount lazily)
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
# GPU-health guard
if ! python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.zeros(1).cuda().add_(1).item()==1.0) else 1)" 2>/dev/null; then
  echo "GPU unusable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1
fi

case "$ARM" in
  mono)  EXP=bdsl_shubert_pretrain_bdsl_only ;;
  xling) EXP=bdsl_shubert_pretrain_xlingual ;;
  *) echo "ARM must be mono|xling (got '$ARM')"; exit 2 ;;
esac
CFG="config/${EXP}.yaml"
echo "P1 pretrain-seed replication: $ARM ($CFG) on $(hostname)"

pids=()
for S in 1 2; do
  WD="work_dir/${EXP}_pseed${S}"
  if [ -f "$WD/pretrained_epoch30.pt" ]; then echo "[done] $WD"; continue; fi
  python -u main_pretrain.py --config "$CFG" --seed "$S" --work-dir "$WD" \
    >> "slurm_logs/${EXP}_pseed${S}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
for S in 1 2; do [ -f "work_dir/${EXP}_pseed${S}/pretrained_epoch30.pt" ] || fail=$((fail+1)); done
echo "=== $ARM pretrains done (failures: $fail) ==="
[ "$fail" = 0 ]
