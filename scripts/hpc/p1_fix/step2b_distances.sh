#!/bin/bash
# Step 2b (P1 fix): LSFB fps regression check, KArSL per-signer geometry, then v2 distances.
set -eo pipefail
OUT=docs/p1_fix_2026-10/step2; mkdir -p $OUT
python - <<'PY' | tee $OUT/lsfb_fps_check.txt
import csv, numpy as np, os, random
D = "data/lsfb_download"
rows = list(csv.DictReader(open(f"{D}/instances.csv"))); random.Random(0).shuffle(rows)
T, dur = [], []
for r in rows:
    p = f"{D}/poses/pose/{r['id']}.npy"
    if os.path.exists(p):
        T.append(np.load(p, mmap_mode="r").shape[0]); dur.append((int(r["end"]) - int(r["start"])) / 1000)
    if len(T) >= 2000: break
T, dur = np.array(T, float), np.array(dur)
b, a = np.polyfit(dur, T, 1)
print(f"LSFB n={len(T)}: T = {a:.2f} + {b:.2f} * duration_s  (slope = frame rate); median T {np.median(T)}, median dur {np.median(dur):.3f}s")
PY
for s in train val test; do python -u tools/check_pose_geometry.py --npy data/karsl_si/${s}_data.npy --name "KArSL $s" --n 300; done | tee $OUT/karsl_geometry_by_split.txt
python -u tools/compute_lang_distance_v2.py --out $OUT/distances_v2.json --draws 20 | tee $OUT/distances_v2.txt
