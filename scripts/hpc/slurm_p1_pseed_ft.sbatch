#!/bin/bash -l
# =====================================================================
# P1 item5 (2026-10-08): headline @10% fine-tune on a given set of SSL backbones.
# Recipe = the headline bdsl_block_gcn_si_lr01_f10_{mono,xling} runs exactly
# (config/bdsl_block_gcn_si_lr01_f10.yaml: lr 0.01, 100 ep, subset_seed 42, NO grad
# clipping, fc dropped), only the backbone weights differ. Fine-tune seeds 0,1,2.
# Best-val checkpoint is then scored once on held-out TEST
# (config/bdsl_block_gcn_si_lr01_test.yaml), as the original *_seed<S>_test runs were.
# All runs of the task go concurrently on one GPU; resume-safe via run_resumable_seed.py.
#
# RUNS (env) = space-separated "TAG=WEIGHTS" items; experiment names are
#   bdsl_block_gcn_si_lr01_f10_<TAG>_seed<S>  and  ..._seed<S>_test
#   sbatch --export=ALL,RUNS="mono_ps1=work_dir/.../pretrained_epoch30.pt ..." scripts/hpc/slurm_p1_pseed_ft.sbatch
# =====================================================================
#SBATCH -J p1_pseed_ft
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem 64G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 24:00:00
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

CFG=config/bdsl_block_gcn_si_lr01_f10.yaml
TCFG=config/bdsl_block_gcn_si_lr01_test.yaml
[ -n "$RUNS" ] || { echo "RUNS is empty"; exit 2; }
echo "P1 pseed fine-tunes on $(hostname): $RUNS"

pids=()
for item in $RUNS; do
  TAG="${item%%=*}"; W="${item#*=}"
  [ -f "$W" ] || { echo "missing backbone $W for $TAG"; exit 3; }
  for S in 0 1 2; do
    EXP="bdsl_block_gcn_si_lr01_f10_${TAG}_seed${S}"
    python -u tools/run_resumable_seed.py --config "$CFG" --seed "$S" --exp "$EXP" \
      --init-weights "$W" --test-config "$TCFG" --test-exp "${EXP}_test" \
      >> "slurm_logs/${EXP}.log" 2>&1 &
    pids+=($!)
  done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== fine-tunes done (failed: $fail) ==="
[ "$fail" = 0 ]
