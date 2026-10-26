#!/bin/bash
# sbatch with retry when the per-user submit limit is reached (AssocMaxSubmitJobLimit): retry every 120 s.
#   bash scripts/hpc/p1_fix/sbatch_retry.sh "<description>" <sbatch args...>
# Appends "<jobid> <description>" to slurm_logs/p1_fix_jobids.txt and prints the job id.
cd "$(dirname "$0")/../../.."   # repository root
DESC="$1"; shift
while true; do
  out=$(sbatch "$@" 2>&1); rc=$?
  if [ $rc = 0 ]; then
    id=$(echo "$out" | awk '/Submitted batch job/{print $4}')
    mkdir -p slurm_logs; echo "$id $(date '+%F %T') $DESC" >> slurm_logs/p1_fix_jobids.txt
    echo "$id"; exit 0
  fi
  if echo "$out" | grep -q "AssocMaxSubmitJobLimit\|job submit limit"; then
    echo "[$(date '+%T')] submit cap full; retry in 120 s" >&2; sleep 120; continue
  fi
  echo "sbatch failed: $out" >&2; exit 1
done
