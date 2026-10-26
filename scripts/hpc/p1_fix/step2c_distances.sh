#!/bin/bash
# Step 2c: rerun of v2 distances after fixing the v1-replica source weighting (50/50), + the unmodified v1 tool.
set -eo pipefail
OUT=docs/p1_fix_2026-10/step2
python -u tools/compute_lang_distance.py | tee $OUT/v1_tool_rerun.txt
python -u tools/compute_lang_distance_v2.py --out $OUT/distances_v2.json --draws 20 | tee $OUT/distances_v2.txt
for bw in pair pooled; do python tools/p1_fix/distance_table.py --bw $bw > $OUT/distances_v2_${bw}bw.md; done
