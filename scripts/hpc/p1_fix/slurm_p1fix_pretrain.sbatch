#!/bin/bash -l
# =====================================================================
# P1 fix (2026-10) step 5: v2 SSL pretraining (fixed WLASL skeleton + v2 k-means targets).
# ONE job, 2 GPUs (submit-cap friendly): GPU lane 0 = pool arm, lane 1 = xling arm, each lane runs
# pretraining seeds 0,1,2 concurrently (3 runs per GPU, the program's cap).
#   pool  -> config/p1fix/bdsl_shubert_pretrain_bdsl_asl_v2.yaml
#   xling -> config/p1fix/bdsl_shubert_pretrain_xlingual_v2.yaml
# 30 epochs, stride_between_stages False (as v1). Resume-safe: main_pretrain.py auto-resumes from
# <wd>/train_state.pt (saved every epoch in the v2 configs); finished runs (pretrained_epoch30.pt) skipped.
# 8-h chunks: USR1 -> requeue -> every run resumes.
#   sbatch scripts/hpc/p1_fix/slurm_p1fix_pretrain.sbatch
# =====================================================================
#SBATCH -J p1fix_pt
#SBATCH -c 24
#SBATCH -n 1
#SBATCH --gpus 2
#SBATCH --mem 200G
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
if ! python -c "import torch,sys; n=torch.cuda.device_count(); sys.exit(0 if n>0 and all(torch.zeros(1).cuda(i).add_(1).item()==1.0 for i in range(n)) else 1)" 2>/dev/null; then
  echo "GPU unusable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1
fi
[ -f data/pretrain_kmeans_targets_bdsl_asl_v2.npz ] || { echo "targets missing"; exit 3; }
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
[ "${#GPUS[@]}" -ge 2 ] || GPUS=("${GPUS[0]}" "${GPUS[0]}")
echo "P1fix v2 pretrain on $(hostname) GPUs ${GPUS[*]} $(date '+%F %T')"
k=0
for arm in pool xling; do
  case $arm in pool) EXP=bdsl_shubert_pretrain_bdsl_asl_v2 ;; xling) EXP=bdsl_shubert_pretrain_xlingual_v2 ;; esac
  for S in 0 1 2; do
    WD="work_dir/${EXP}_pseed${S}"
    if [ -f "$WD/pretrained_epoch30.pt" ]; then echo "[done] $WD"; continue; fi
    echo "launch $arm seed $S on GPU ${GPUS[$k]} -> $WD"
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} python -u main_pretrain.py --config "config/p1fix/${EXP}.yaml" --seed "$S" \
      --work-dir "$WD" >> "slurm_logs/p1_fix/${EXP}_pseed${S}.log" 2>&1 &
  done
  k=$((k+1))
done
wait || true
fail=0
for EXP in bdsl_shubert_pretrain_bdsl_asl_v2 bdsl_shubert_pretrain_xlingual_v2; do for S in 0 1 2; do
  [ -f "work_dir/${EXP}_pseed${S}/pretrained_epoch30.pt" ] || { echo "INCOMPLETE ${EXP}_pseed${S}"; fail=$((fail+1)); }
done; done
echo "=== v2 pretrain done $(date '+%F %T') (incomplete: $fail) ==="
[ "$fail" = 0 ]
