#!/bin/bash
# Launch chain for the v2 rerun: (targets job) -> ONE 2-GPU pretrain job (6 runs) -> one 2-GPU fine-tune
# job per pretraining seed (afterok on the pretrain job). All submissions via sbatch_retry.sh.
#   bash scripts/hpc/p1_fix/launch_step5.sh <targets_jobid|none>
cd "$(dirname "$0")/../../.."   # repository root
R=scripts/hpc/p1_fix/sbatch_retry.sh
DEP=""; [ "$1" != "none" ] && DEP="--dependency=afterok:$1"
PT=$(bash $R "p1fix_pt v2 pretrain pool+xling seeds 0-2 (2 GPUs, 3 runs/GPU)" $DEP scripts/hpc/p1_fix/slurm_p1fix_pretrain.sbatch) || exit 1
echo "pretrain job $PT"
for PS in 0 1 2; do
  FT=$(bash $R "p1fix_ft v2 law fine-tunes PS=$PS (afterok $PT)" -J p1fix_ft_ps$PS --dependency=afterok:$PT \
       --export=ALL,PS=$PS scripts/hpc/p1_fix/slurm_p1fix_ft.sbatch) || exit 1
  echo "ft ps$PS: $FT"
done
