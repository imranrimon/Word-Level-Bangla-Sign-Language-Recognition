#!/bin/bash -l
# P1 cross-domain: finetune matrix on HRNet action domains @10% (COCO-17, V=17).
# Manifest: CFG|BACKBONE(scratch|<dir>)|EXP|FRAC|SEED. gain = SSL-init - scratch vs MMD.
#SBATCH -J ft_hrnet
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --array=0-19
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
for d in hmdb51 ucf101 gym ntu_hrnet; do ls data/hrnet_${d}/train_data.npy >/dev/null 2>&1 || true; done
MAN=slurm_logs/hrnet_ft_manifest.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAN"); [ -n "$LINE" ] || { echo "no line"; exit 0; }
IFS='|' read -r CFG BB EXP FRAC SEED <<< "$LINE"
WD="work_dir/${EXP}"; mkdir -p "$WD"
if [ -f "$WD/eval_results/best_acc.pkl" ]; then echo "=== DONE $EXP ==="; exit 0; fi
CKPT=$(ls -t "$WD/${EXP}_model-"[0-9]*.pt 2>/dev/null | head -1) || true
if [ -n "$CKPT" ]; then EP=$(basename "$CKPT" | sed -E "s/.*_model-([0-9]+)\.pt/\1/"); echo "=== RESUME $EXP @ $((EP+1)) ==="; WARG="--weights $CKPT --start-epoch $((EP+1))";
elif [ "$BB" = "scratch" ]; then echo "=== FRESH(scratch) $EXP ==="; WARG="";
else echo "=== FRESH(SSL $BB) $EXP ==="; WARG="--weights $BB/pretrained_epoch30.pt --ignore-weights fc.weight fc.bias"; fi
python main.py --config "$CFG" --seed "$SEED" -Experiment_name "$EXP" $WARG --train-fraction "$FRAC" --num-worker 8
echo "=== done $EXP ==="
