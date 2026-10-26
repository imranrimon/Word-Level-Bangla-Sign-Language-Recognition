#!/bin/bash
# Step 3 (P1 fix): KArSL. (a) evaluate every existing KArSL law checkpoint (best-val, selected on the
# signer-DEPENDENT val) on the held-out signer U03; (b) build signer-disjoint bundles v2a/v2b.
set -eo pipefail
OUT=docs/p1_fix_2026-10/step3; mkdir -p $OUT
CK=()
for arm in pool xling mono scratch; do for ps in "" _ps1 _ps2; do for s in 0 1 2; do
  e=karsl_si_lr01_f10_${arm}${ps}_seed${s}; [ -f work_dir/$e/${e}_model-best.pt ] && CK+=(work_dir/$e/${e}_model-best.pt)
done; done; done
echo "${#CK[@]} checkpoints"
python -u tools/p1_fix/eval_ckpts.py --config config/karsl_si_lr01_f10.yaml \
  --data data/karsl_si/test_data.npy --labels data/karsl_si/test_label.pkl --ckpts "${CK[@]}" --out $OUT/karsl_u03_test_eval.json
python -u tools/p1_fix/eval_ckpts.py --config config/karsl_si_lr01_f10.yaml \
  --data data/karsl_si/val_data.npy --labels data/karsl_si/val_label.pkl --ckpts "${CK[@]}" --out $OUT/karsl_sdval_eval.json
[ -f data/karsl_si_v2b/test_label.pkl ] || python -u tools/p1_fix/make_karsl_si_v2.py | tee $OUT/make_karsl_si_v2.txt
