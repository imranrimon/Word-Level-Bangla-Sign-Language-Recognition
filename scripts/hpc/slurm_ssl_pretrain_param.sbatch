#!/bin/bash -l
# Parameterized SSL pretraining. Submit with:
#   sbatch --export=ALL,CFG=config/<name>.yaml scripts/hpc/slurm_ssl_pretrain_param.sbatch
#SBATCH -J ssl_pretrain
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 36:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
: "${CFG:?must pass CFG=config/<name>.yaml via --export}"
# work_dir writability guard (shared file systems may mount lazily) + GPU health guard
ok=0; for i in $(seq 1 18); do touch data/.rw_$$ 2>/dev/null && { rm -f data/.rw_$$; ok=1; break; }; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "GPU dead -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
echo "=== SSL pretrain: $CFG ==="
python -u main_pretrain.py --config "$CFG" --seed 0
echo "=== done: $CFG ==="
