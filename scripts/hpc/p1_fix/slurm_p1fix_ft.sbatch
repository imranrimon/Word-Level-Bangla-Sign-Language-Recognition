#!/bin/bash -l
# =====================================================================
# P1 fix (2026-10) step 5: v2 law fine-tunes for ONE pretraining seed PS, both arms.
# 2 GPUs: GPU lane 0 = pool arm, lane 1 = xling arm; <=3 concurrent runs per GPU.
# Per arm: 8 target languages x 5 fine-tune seeds (0-4) = 40 runs, recipe config/p1fix/<stem>_f10_v2.yaml
# (= the P1 @10% law recipe; KArSL on the signer-disjoint karsl_si_v2a split), backbone
#   pool  = work_dir/bdsl_shubert_pretrain_bdsl_asl_v2_pseed<PS>/pretrained_epoch30.pt
#   xling = work_dir/bdsl_shubert_pretrain_xlingual_v2_pseed<PS>/pretrained_epoch30.pt
# Best-val checkpoint is also scored on held-out TEST signers where a separate test split exists.
# Resume-safe via tools/run_resumable_seed.py (finished runs skipped, partial runs resume).
# Results -> results_p1fix_v2.csv. Experiment names: <stem>_f10_v2_<arm>_ps<PS>_seed<f>[_test]
#   sbatch --dependency=afterok:<pool_pt>_<PS>:<xling_pt>_<PS> --export=ALL,PS=0 scripts/hpc/p1_fix/slurm_p1fix_ft.sbatch
# =====================================================================
#SBATCH -J p1fix_ft
#SBATCH -c 24
#SBATCH -n 1
#SBATCH --gpus 2
#SBATCH --mem 180G
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
[ -n "${PS:-}" ] || { echo "PS (pretraining seed) is empty"; exit 2; }
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
[ "${#GPUS[@]}" -ge 2 ] || GPUS=("${GPUS[0]}" "${GPUS[0]}")
MAXPAR=3
STEMS="gsl_si autsl_si lsfb_si karsl_si bdsl_block_gcn_si lsa64_si libras_si bisindo_si"   # cost-sorted
echo "P1fix v2 law ft ps${PS} on $(hostname) GPUs ${GPUS[*]} $(date '+%F %T')"

lane () {   # $1 = arm, $2 = gpu
  local arm=$1 gpu=$2 PEXP
  case $arm in pool) PEXP=bdsl_shubert_pretrain_bdsl_asl_v2 ;; xling) PEXP=bdsl_shubert_pretrain_xlingual_v2 ;; esac
  local W=work_dir/${PEXP}_pseed${PS}/pretrained_epoch30.pt
  [ -f "$W" ] || { echo "missing backbone $W"; return 3; }
  for st in $STEMS; do for s in 0 1 2 3 4; do
    local EXP="${st}_f10_v2_${arm}_ps${PS}_seed${s}" T=()
    [ -f "config/p1fix/${st}_test_v2.yaml" ] && T=(--test-config "config/p1fix/${st}_test_v2.yaml" --test-exp "${EXP}_test")
    CUDA_VISIBLE_DEVICES=$gpu python -u tools/run_resumable_seed.py --config "config/p1fix/${st}_f10_v2.yaml" --seed "$s" \
      --exp "$EXP" --init-weights "$W" "${T[@]}" >> "slurm_logs/p1_fix/${EXP}.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 20; done
  done; done
  wait || true
}
lane pool "${GPUS[0]}" &
lane xling "${GPUS[1]}" &
wait || true
fail=0
for arm in pool xling; do for st in $STEMS; do for s in 0 1 2 3 4; do
  EXP="${st}_f10_v2_${arm}_ps${PS}_seed${s}"
  [ -f "work_dir/${EXP}/${EXP}_model-99.pt" ] && [ -f "work_dir/${EXP}/${EXP}_model-best.pt" ] || { echo "INCOMPLETE $EXP"; fail=$((fail+1)); }
done; done; done
echo "=== v2 law ft ps${PS} done $(date '+%F %T') (incomplete: $fail) ==="
[ "$fail" = 0 ]
