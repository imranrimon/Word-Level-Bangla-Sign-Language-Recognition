#!/bin/bash -l
# =====================================================================
# P1 item6 7-seed scale-up (2026-10-11): ALL law fine-tunes for one (ARM, PS) backbone,
# packed into 3 array tasks (submit-cap friendly). Recipe identical to
# slurm_p1_law_pseed_ft.sbatch (seeds 1/2): config/<lang>_si_lr01_f10.yaml (10% labels,
# subset_seed 42, lr 0.01, 100 ep, no clipping, fc re-init), backbone
#   pool  = work_dir/bdsl_shubert_pretrain_bdsl_asl_pseed<PS>/pretrained_epoch30.pt
#   xling = work_dir/bdsl_shubert_pretrain_xlingual_pseed<PS>/pretrained_epoch30.pt
# Fine-tune seeds 0-2 for all 7 languages, plus 3-4 for GSL/LSFB/LIBRAS (27 runs); best-val
# checkpoint scored on held-out TEST for BdSL/LSA64/AUTSL (P1 tab:test).
# Task k in {0,1,2} takes every 3rd run of a cost-sorted list and runs them through a pool of
# AT MOST 3 concurrent runs on its one GPU. Resume-safe via tools/run_resumable_seed.py.
#   sbatch -a 0-2 --dependency=afterok:<pt>_<PS> --export=ALL,ARM=pool,PS=3 scripts/hpc/slurm_p1_law_pseed_ft_packed.sbatch
# Experiment names: <stem>_lr01_f10_<ARM>_ps<PS>_seed<S>[_test]
# =====================================================================
#SBATCH -J p1_law_ft7
#SBATCH -a 0-2
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 80G
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

case "$ARM" in
  pool)  PEXP=bdsl_shubert_pretrain_bdsl_asl ;;
  xling) PEXP=bdsl_shubert_pretrain_xlingual ;;
  *) echo "ARM must be pool|xling (got '$ARM')"; exit 2 ;;
esac
[ -n "$PS" ] || { echo "PS (pretraining seed) is empty"; exit 2; }
W=work_dir/${PEXP}_pseed${PS}/pretrained_epoch30.pt
[ -f "$W" ] || { echo "missing backbone $W"; exit 3; }

declare -A CFG=( [bdsl_block_gcn_si]=config/bdsl_block_gcn_si_lr01_f10.yaml [gsl_si]=config/gsl_si_lr01_f10.yaml
  [lsa64_si]=config/lsa64_si_lr01_f10.yaml [karsl_si]=config/karsl_si_lr01_f10.yaml
  [autsl_si]=config/autsl_si_lr01_f10.yaml [lsfb_si]=config/lsfb_si_lr01_f10.yaml
  [libras_si]=config/libras_si_lr01_f10.yaml )
declare -A TCFG=( [bdsl_block_gcn_si]=config/bdsl_block_gcn_si_lr01_test.yaml
  [lsa64_si]=config/lsa64_si_test.yaml [autsl_si]=config/autsl_si_test.yaml )
# cost-sorted run list (stem:seed), heaviest first
RUNS=()
for st in gsl_si autsl_si lsfb_si karsl_si bdsl_block_gcn_si lsa64_si libras_si; do
  case $st in gsl_si|lsfb_si|libras_si) SS="0 1 2 3 4" ;; *) SS="0 1 2" ;; esac
  for s in $SS; do RUNS+=("$st:$s"); done
done
k=${SLURM_ARRAY_TASK_ID:?}
MINE=(); for i in "${!RUNS[@]}"; do [ $((i % 3)) = "$k" ] && MINE+=("${RUNS[$i]}"); done
echo "P1 law ft ${ARM} ps${PS} task $k on $(hostname): ${MINE[*]}"

run_one() {
  local st=${1%%:*} s=${1##*:}
  local EXP="${st}_lr01_f10_${ARM}_ps${PS}_seed${s}"
  local TARGS=(); [ -n "${TCFG[$st]:-}" ] && TARGS=(--test-config "${TCFG[$st]}" --test-exp "${EXP}_test")
  python -u tools/run_resumable_seed.py --config "${CFG[$st]}" --seed "$s" --exp "$EXP" \
    --init-weights "$W" "${TARGS[@]}" >> "slurm_logs/${EXP}.log" 2>&1
}

for r in "${MINE[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n || true; done
  run_one "$r" &
done
wait || true
# verify by artefacts (exit codes of reaped background jobs are not reliable to count)
fail=0
for r in "${MINE[@]}"; do
  st=${r%%:*}; s=${r##*:}; EXP="${st}_lr01_f10_${ARM}_ps${PS}_seed${s}"
  [ -f "work_dir/${EXP}/${EXP}_model-99.pt" ] && [ -f "work_dir/${EXP}/${EXP}_model-best.pt" ] || { echo "INCOMPLETE $EXP"; fail=$((fail+1)); }
  if [ -n "${TCFG[$st]:-}" ]; then grep -q ",${EXP}_test," results_final.csv || { echo "NO TEST ROW ${EXP}_test"; fail=$((fail+1)); }; fi
done
echo "=== ${ARM} ps${PS} task $k done (incomplete: $fail) ==="
[ "$fail" = 0 ]
