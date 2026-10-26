#!/bin/bash
# Step 1 (P1 fix): verify the v1 scramble, convert WLASL to the v2 cache (+bone), check geometry.
set -eo pipefail
V3=data/raw_wlasl_v3/landmarks_V3.npz
OUT=docs/p1_fix_2026-10/step1
mkdir -p $OUT
[ -f $OUT/verify_v1_scramble.txt ] || python -u tools/p1_fix/verify_wlasl_v1_scramble.py $V3 | tee $OUT/verify_v1_scramble.txt
python -u preprocessing/convert_wlasl_landmarks_to_pose_cache_v2.py --landmarks $V3 \
  --output-dir data/wlasl_pose_cache_v2 --bone-dir data/wlasl_pose_cache_bone_v2 --skip-existing | tee -a $OUT/convert_v2.txt
python -u tools/check_pose_geometry.py --all --n 300 --json $OUT/geometry_all.json --markdown $OUT/geometry_all.md | tee $OUT/geometry_all.txt
