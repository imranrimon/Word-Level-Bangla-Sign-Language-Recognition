#!/bin/bash -l
# =====================================================================
# P1: LogME (feature-based transferability) for all 20 source->target pairs.
# For each (s,t): extract src_s backbone features on target-t train, compute LogME
# vs target-t labels. Correlated with the pose-motion MMD + transfer gain in the
# transferability-estimator table. One GPU job,
# ~20 quick forward-pass extractions; continues past any single-pair failure.
#   sbatch scripts/hpc/slurm_logme_transfer.sbatch
# =====================================================================
#SBATCH -J logme_xfer
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 2:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
set -o pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8 PYTHONPATH="$SLURM_SUBMIT_DIR"
for i in $(seq 1 30); do ls data/ >/dev/null 2>&1 && break; sleep 10; done

declare -A BB=(
  [bdsl]=work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt
  [gsl]=work_dir/bdsl_shubert_pretrain_src_gsl/pretrained_epoch30.pt
  [lsa64]=work_dir/bdsl_shubert_pretrain_src_lsa64/pretrained_epoch30.pt
  [karsl]=work_dir/bdsl_shubert_pretrain_src_karsl/pretrained_epoch30.pt
  [autsl]=work_dir/bdsl_shubert_pretrain_src_autsl/pretrained_epoch30.pt
)
declare -A CFG=(
  [bdsl]=config/bdsl_si_lr01_pairft.yaml
  [gsl]=config/gsl_si_lr01_pairft.yaml
  [lsa64]=config/lsa64_si_lr01_pairft.yaml
  [karsl]=config/karsl_si_lr01_pairft.yaml
  [autsl]=config/autsl_si_lr01_pairft.yaml
)
LANGS="bdsl gsl lsa64 karsl autsl"
for t in $LANGS; do for s in $LANGS; do
  [ "$s" = "$t" ] && continue
  echo "=== LogME ${s}2${t} ==="
  python -u tools/logme_transfer.py --config "${CFG[$t]}" --backbone "${BB[$s]}" --tag "${s}2${t}" || echo "WARN ${s}2${t} failed"
done; done
echo "=== LogME transfer done ==="; cat data/logme_transfer.json 2>/dev/null
