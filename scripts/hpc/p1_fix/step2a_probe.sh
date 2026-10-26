#!/bin/bash
set -eo pipefail
mkdir -p docs/p1_fix_2026-10/step2 docs/p1_fix_2026-10/step5
python -u tools/p1_fix/kmeans_dropout_diag.py --manifest data/ssl_pool_manifest_bdsl_asl_v2.json --n-clips 4000 --out docs/p1_fix_2026-10/step5/kmeans_dropout_diag.json
python -u tools/p1_fix/probe_fps.py --out docs/p1_fix_2026-10/step2/fps_probe.json
