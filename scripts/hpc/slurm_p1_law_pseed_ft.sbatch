#!/bin/bash -l
# =====================================================================
# P1 item6 (2026-10-10): distance-gating-law fine-tunes on the pretraining-seed
# replicate backbones. Law quantity = xling - pool (best-epoch VAL Top-1 @10% labels,
# lr 0.01, 100 ep, subset_seed 42, NO grad clipping, fc re-initialised). Every target
# language fine-tunes the SAME two backbones:
#   pool  = work_dir/bdsl_shubert_pretrain_bdsl_asl_pseed<PS>/pretrained_epoch30.pt
#   xling = work_dir/bdsl_shubert_pretrain_xlingual_pseed<PS>/pretrained_epoch30.pt
# Configs are the original <lang>_si_lr01 configs materialised with the f10 recipe
# (config/<lang>_si_lr01_f10.yaml; BdSL reuses item5's bdsl_block_gcn_si_lr01_f10.yaml).
# Where P1 reports held-out TEST (BdSL, LSA64, AUTSL: tab:test) the best-val checkpoint
# is scored once on TEST, as the original *_seed<S>_test runs were.
#
# One array task = one language; its SEEDS (<=3) run concurrently on one GPU.
# Env: ARM=pool|xling  PS=1|2  SEEDS="0 1 2" (or "3 4")  [BB=<override backbone> TAG=<override tag>]
# Array index -> language: 0 BdSL, 1 GSL, 2 LSA64, 3 KArSL, 4 AUTSL, 5 LSFB, 6 LIBRAS
# Experiment names: <stem>_lr01_f10_<ARM>_ps<PS>_seed<S>[_test]  (item5 naming)
# Resume-safe on requeue via tools/run_resumable_seed.py.
#   sbatch -a 0-6 --export=ALL,ARM=pool,PS=1,SEEDS="0 1 2" scripts/hpc/slurm_p1_law_pseed_ft.sbatch
# =====================================================================
#SBATCH -J p1_law_pseed_ft
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 80G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 24:00:00
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

# work_dir writability guard (shared file systems may mount lazily)
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
# GPU-health guard
if ! python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.zeros(1).cuda().add_(1).item()==1.0) else 1)" 2>/dev/null; then
  echo "GPU unusable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1
fi

STEMS=(bdsl_block_gcn_si gsl_si lsa64_si karsl_si autsl_si lsfb_si libras_si)
CFGS=(config/bdsl_block_gcn_si_lr01_f10.yaml config/gsl_si_lr01_f10.yaml config/lsa64_si_lr01_f10.yaml \
      config/karsl_si_lr01_f10.yaml config/autsl_si_lr01_f10.yaml config/lsfb_si_lr01_f10.yaml \
      config/libras_si_lr01_f10.yaml)
TCFGS=(config/bdsl_block_gcn_si_lr01_test.yaml "" config/lsa64_si_test.yaml "" config/autsl_si_test.yaml "" "")
i=${SLURM_ARRAY_TASK_ID:?array index = language}
STEM=${STEMS[$i]}; CFG=${CFGS[$i]}; TCFG=${TCFGS[$i]}
case "$ARM" in
  pool)  PEXP=bdsl_shubert_pretrain_bdsl_asl ;;
  xling) PEXP=bdsl_shubert_pretrain_xlingual ;;
  *) echo "ARM must be pool|xling (got '$ARM')"; exit 2 ;;
esac
[ -n "$PS" ] || { echo "PS (pretraining seed) is empty"; exit 2; }
W=${BB:-work_dir/${PEXP}_pseed${PS}/pretrained_epoch30.pt}
TAG=${TAG:-${ARM}_ps${PS}}
SEEDS=${SEEDS:-0 1 2}
[ -f "$W" ] || { echo "missing backbone $W"; exit 3; }
n=$(echo $SEEDS | wc -w); [ "$n" -le 3 ] || { echo "max 3 concurrent runs per GPU (got $n)"; exit 2; }
echo "P1 law fine-tune: $STEM $TAG seeds [$SEEDS] backbone $W test '${TCFG}' on $(hostname)"

pids=()
for S in $SEEDS; do
  EXP="${STEM}_lr01_f10_${TAG}_seed${S}"
  TARGS=(); [ -n "$TCFG" ] && TARGS=(--test-config "$TCFG" --test-exp "${EXP}_test")
  python -u tools/run_resumable_seed.py --config "$CFG" --seed "$S" --exp "$EXP" \
    --init-weights "$W" "${TARGS[@]}" >> "slurm_logs/${EXP}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== $STEM $TAG fine-tunes done (failed: $fail) ==="
[ "$fail" = 0 ]
