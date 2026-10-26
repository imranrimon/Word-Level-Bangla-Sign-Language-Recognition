#!/bin/bash -l
# =====================================================================
# P1 fix program (2026-10): generic CPU-side analysis job (data reads must run on a compute
# node). Runs "$CMD" from the repo root.
#   sbatch --export=ALL,CMD="python -u tools/check_pose_geometry.py ..." -J p1fix_geo scripts/hpc/p1_fix/slurm_p1fix_cpu.sbatch
# =====================================================================
#SBATCH -J p1fix_cpu
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem 64G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 8:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH -o slurm_logs/p1_fix/%x_%j.out
#SBATCH -e slurm_logs/p1_fix/%x_%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
trap 'echo "[chunk] $(date "+%F %T") time limit near -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
echo "[p1fix_cpu] $(hostname) $(date '+%F %T'): $CMD"
bash -c "$CMD" &
wait $!
echo "[p1fix_cpu] done $(date '+%F %T')"
