#!/bin/bash -l
# =====================================================================
# 2026-10 P1 audit reruns (leak-free SSL). One self-contained job per JOBSET:
#   A = R1 AUTSL-excluded multisrc pool: 5 backbones (pool6_uniform, dgxl,
#       dgxl_autsl, lambdad_autsl, poolnoalign) -> AUTSL @10/25% x 3 seeds
#   B = R2 train-signers-only monolingual SSL (LSA64, KArSL, AUTSL, GSL)
#       -> each target @ the paper's budgets x 3 seeds
#   C = R1 AUTSL-train-signers-only multisrc pool: CORAL + matched no-align pool
#       -> AUTSL @10/25% x 3 seeds
# Pools/configs are built by scripts/hpc/audit_reruns_2026-10/prep.py (already run).
# Pretrains run concurrently on the one GPU; fine-tunes 4 at a time. Everything is
# idempotent and requeue-safe: k-means skips if its .npz exists, main_pretrain.py
# auto-resumes from train_state.pt, fine-tunes resume from their latest checkpoint
# and are only skipped once our own .audit_done marker exists (best_acc.pkl is
# written mid-run, so it is NOT a completion marker).
# Submit: sbatch --export=ALL,JOBSET=A -J audit_A scripts/hpc/audit_reruns_2026-10/slurm_audit_reruns.sbatch
# =====================================================================
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH --mem=96G
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
#SBATCH -t 4-00:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/audit_reruns
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-INTEL}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8 PYTHONPATH="$SLURM_SUBMIT_DIR"
LOGD=slurm_logs/audit_reruns

# scratch + GPU guards
ok=0; for i in $(seq 1 30); do ls work_dir/ >/dev/null 2>&1 || true; touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; sleep 20; done
[ "$ok" = 1 ] || { echo "scratch down -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "GPU dead -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

kmeans() {  # manifest output
  if [ -f "$2" ]; then echo "[kmeans] exists $2"; return 0; fi
  echo "[kmeans] $1 -> $2"
  python -u preprocessing/compute_pretrain_targets.py --manifest "$1" --output "${2%.npz}.partial.npz" \
    --num-clusters 64 --feature-mode pose_motion --per-source-cap 200000
  mv "${2%.npz}.partial.npz" "$2"
}

pretrain_one() {  # config-stem
  local cfg="config/$1.yaml" wd="work_dir/$1"
  if [ -f "$wd/pretrained_epoch30.pt" ]; then echo "[pretrain] done $1"; return 0; fi
  echo "[pretrain] start $1 $(date)"
  python -u main_pretrain.py --config "$cfg" --seed 0 >> "$LOGD/pretrain_$1.log" 2>&1 \
    && echo "[pretrain] ok $1 $(date)" || echo "[pretrain] FAILED $1 (see $LOGD/pretrain_$1.log)"
}

ft_one() {  # CFG|BB|EXP|FRAC|SEED
  IFS='|' read -r CFG BB EXP FRAC SEED <<< "$1"
  local wd="work_dir/$EXP"
  if [ -f "$wd/.audit_done" ]; then echo "[ft] done $EXP"; return 0; fi
  if [ ! -f "$BB/pretrained_epoch30.pt" ]; then echo "[ft] SKIP $EXP: backbone $BB missing"; return 0; fi
  mkdir -p "$wd"
  local ck wargs
  ck=$(ls -t "$wd/${EXP}_model-"*.pt 2>/dev/null | head -1) || true
  if [ -n "$ck" ]; then
    local ep; ep=$(basename "$ck" | sed -E "s/.*_model-([0-9]+)\.pt/\1/")
    wargs="--weights $ck --start-epoch $((ep+1))"; echo "[ft] resume $EXP @ ep $((ep+1))"
  else
    wargs="--weights $BB/pretrained_epoch30.pt --ignore-weights fc.weight fc.bias"; echo "[ft] fresh $EXP"
  fi
  if python main.py --config "$CFG" --seed "$SEED" -Experiment_name "$EXP" $wargs \
       --train-fraction "$FRAC" --num-worker 3 >> "$LOGD/ft_$EXP.log" 2>&1; then
    touch "$wd/.audit_done"; echo "[ft] ok $EXP"
  else
    echo "[ft] FAILED $EXP (see $LOGD/ft_$EXP.log)"
  fi
}
export -f pretrain_one ft_one
export LOGD

FT=()
add_ft() {  # cfg backbone-stem exp-stem fracs...
  local cfg=$1 bb=$2 stem=$3; shift 3
  for f in "$@"; do
    local tag; tag=$(python -c "print('f%d' % round(float('$f')*100))")
    for s in 0 1 2; do FT+=("$cfg|work_dir/$bb|${stem}_${tag}_seed${s}|$f|$s"); done
  done
}

case "$JOBSET" in
  A)
    python -u scripts/hpc/audit_reruns_2026-10/prep.py r1noautsl   # manifest + MMD(src->AUTSL train) + configs (idempotent)
    kmeans data/ssl_pool_manifest_multisrc_noautsl.json data/pretrain_kmeans_targets_multisrc_noautsl.npz
    PRE=(bdsl_shubert_pretrain_pool6_uniform_noautsl bdsl_shubert_pretrain_dgxl_noautsl
         bdsl_shubert_pretrain_dgxl_autsl_noautsl bdsl_shubert_pretrain_lambdad_autsl_noautsl
         bdsl_shubert_pretrain_poolnoalign_noautsl)
    NPRE=3
    C=config/autsl_si_lr01.yaml
    add_ft $C bdsl_shubert_pretrain_pool6_uniform_noautsl   ft_autsl_pool6_noautsl      0.1 0.25
    add_ft $C bdsl_shubert_pretrain_dgxl_noautsl            ft_autsl_dgxl_noautsl       0.1 0.25
    add_ft $C bdsl_shubert_pretrain_dgxl_autsl_noautsl      ft_autslA_noautsl           0.1 0.25
    add_ft $C bdsl_shubert_pretrain_lambdad_autsl_noautsl   lambdad_autsl_noautsl       0.1 0.25
    add_ft $C bdsl_shubert_pretrain_poolnoalign_noautsl     ft_autsl_poolnoalign_noautsl 0.1 0.25
    ;;
  B)
    for t in lsa64 karsl autsl gsl; do
      kmeans data/ssl_pool_manifest_${t}_trainonly.json data/pretrain_kmeans_targets_${t}_trainonly.npz
    done
    PRE=(bdsl_shubert_pretrain_monotr_gsl bdsl_shubert_pretrain_monotr_autsl
         bdsl_shubert_pretrain_monotr_karsl bdsl_shubert_pretrain_monotr_lsa64)
    NPRE=4
    add_ft config/lsa64_si_lr01.yaml       bdsl_shubert_pretrain_monotr_lsa64 lsa64_si_lr01 0.1 0.25 0.5
    add_ft config/karsl_si_lr01.yaml       bdsl_shubert_pretrain_monotr_karsl karsl_si_lr01 0.1 0.25
    add_ft config/autsl_si_lr01.yaml       bdsl_shubert_pretrain_monotr_autsl autsl_si_lr01 0.1 0.25
    add_ft config/gsl_si_lr01_pairft.yaml  bdsl_shubert_pretrain_monotr_gsl   gsl_si_lr01   0.1
    # exp names: <t>_si_lr01_f<NN>_seed<s> would collide with the originals -> insert _monotr
    for i in "${!FT[@]}"; do FT[$i]=$(echo "${FT[$i]}" | sed -E 's/_si_lr01_(f[0-9]+)_seed/_si_lr01_\1_monotr_seed/'); done
    ;;
  C)
    kmeans data/ssl_pool_manifest_multisrc_autsltrain.json data/pretrain_kmeans_targets_multisrc_autsltrain.npz
    PRE=(bdsl_shubert_pretrain_coral_autsl_autsltrain bdsl_shubert_pretrain_poolnoalign_autsltrain)
    NPRE=2
    C=config/autsl_si_lr01.yaml
    add_ft $C bdsl_shubert_pretrain_coral_autsl_autsltrain  coral_autsl_autsltrain          0.1 0.25
    add_ft $C bdsl_shubert_pretrain_poolnoalign_autsltrain  ft_autsl_poolnoalign_autsltrain 0.1 0.25
    ;;
  *) echo "JOBSET must be A, B or C"; exit 1 ;;
esac

echo "=== JOBSET $JOBSET: ${#PRE[@]} pretrains (x$NPRE concurrent), ${#FT[@]} fine-tunes (x4) ==="
printf '%s\n' "${PRE[@]}" | xargs -P "$NPRE" -I{} bash -c 'pretrain_one "$@"' _ {}
printf '%s\n' "${FT[@]}"  | xargs -P 4 -I{} bash -c 'ft_one "$@"' _ {}
echo "=== JOBSET $JOBSET finished $(date) ==="
