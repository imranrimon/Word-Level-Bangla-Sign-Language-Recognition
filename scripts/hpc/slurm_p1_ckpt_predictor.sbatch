#!/bin/bash -l
# =====================================================================
# P1 checkpoint-aware G predictor (2026-10-22): feature extraction for every READY pool/xling backbone
# (pseeds 0-6) on the 7 law targets' TRAIN clips + a proportional real-pool source sample, and the
# data-only MMD recomputed against the real pool. Idempotent: cached (lang, backbone) files are skipped,
# so re-submitting after more pretrains finish only extracts the new backbones.
#   sbatch scripts/hpc/slurm_p1_ckpt_predictor.sbatch
# Outputs: work_dir/p1_ckpt_predictor/{<lang>/<arm>_ps<p>.npz, data_mmd.json}
# =====================================================================
#SBATCH -J p1_ckpt_pred
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 64G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 8:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
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

trap 'echo "[chunk] $(date "+%F %T") time limit near -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

echo "P1 ckpt predictor extraction on $(hostname) $(date)"
python -u tools/p1_ckpt_predictor/datammd.py &
wait $!
python -u tools/p1_ckpt_predictor/extract.py --pseeds 0-6 &
wait $!
python -u tools/p1_ckpt_predictor/score.py compute &
wait $! || true
echo "done $(date)"
