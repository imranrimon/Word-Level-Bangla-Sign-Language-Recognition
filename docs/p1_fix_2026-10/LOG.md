# P1 fix-and-rerun program (2026-10) — running log

Program: fix the WLASL skeleton scramble, the distance tool (source / descriptor / fps / MMD),
check KArSL signer independence, pre-register, rebuild pool+xling backbones on the fixed pool,
rerun the law fine-tunes, analyse per pre-registration. All new artefacts are versioned (`_v2`);
no existing data file, cache, manifest, config or core script was modified in place.

Every data read ran on a compute node (the login node could not read part of the shared file system).

## 2026-10-22

* 04:55 Located the source of `landmarks_V3.npz`: Kaggle dataset `abd0kamel/mutemotion-output`
  ("MuteMotion: WLASL MediaPipe Encoded", Apache-2.0), named in the docstring of
  `preprocessing/convert_wlasl_landmarks_to_pose_cache.py`. Kaggle credentials exist
  (`~/.kaggle/kaggle.json`); no acceptance step needed.
* The dataset card states the V3 order explicitly:
  **"The order is [ Right Hand (21), Left Hand (21), Pose (33), Face (478) ]"** (553 points).
  The old converter assumed [Pose 33, Face 468, LH 21, RH 21, extras 10]. So the old
  "body" slots [0,12,11,14,13,16,15] were right-hand landmarks (wrist, middle-tip, middle-DIP,
  ring-PIP, ring-MCP, ring-tip, ring-DIP) and the old "hand" slots 501..542 were face-mesh
  points. This explains every audit symptom (shoulder width 0.018; body slots missing
  together 38% of frames = right hand not detected; "hands" near-static).
  Layout still verified empirically below before conversion.
* Download: `kaggle datasets download -f landmarks_V3.npz` buffered the whole file in RAM on the
  login node (RSS 3.7 GB and growing) -> killed; re-downloaded with curl (Kaggle API, basic auth
  from kaggle.json) to `data/raw_wlasl_v3/`. Also
  `WLASL_parsed_data.json` (21,083 entries; video path, gloss, frame_start/end, split).
* 05:01 Job J34 (debug CPU): **mechanism proven** — the v1 cache is bit-identical to the V3
  indices the v1 converter used on 300/300 sampled clips, and the key sets match (21,083). Under the
  documented order, v1 slots 0-6 ("body") = right-hand landmarks RH0, RH12, RH11, RH14, RH13, RH16,
  RH15 and v1 slots 7-26 ("hands") = face-mesh points FACE426..FACE467
  (`docs/p1_fix_2026-10/step1/verify_v1_scramble.txt`).
* Empirical layout check on 150 raw V3 clips (`scratchpad v3_layout.py`): block 0-20 wrist is
  0.020 from pose[16] (subject R wrist) vs 0.321 from pose[15]; block 21-41 wrist 0.022 from pose[15];
  pose shoulder width 0.294; nose above shoulders 100%; face pt 76 (=face idx 1, nose tip) 0.012 from
  pose nose; per-frame each block all-or-none (missing = zeros). Person's right shoulder is on the
  image left in 100% of frames (non-mirrored), same convention as preprocess_bdsl.py.
* Wrote `preprocessing/convert_wlasl_landmarks_to_pose_cache_v2.py` (pose 42+[0,12,11,14,13,16,15],
  RH 0+[0,4,5,8,9,12,13,16,17,20], LH 21+same; channel 2 = binary detection as v1) with pre-write
  and post-write geometry assertions + per-clip wrist check; and `tools/check_pose_geometry.py`.
  Converted 21,083 clips -> `data/wlasl_pose_cache_v2` and `data/wlasl_pose_cache_bone_v2`
  (make_bone_pool transform). Pre/post checks PASS (shoulder 0.30, R/L wrist 0.06/0.07 s from the
  correct pose wrist vs 1.01/0.75 s from the wrong one, hand span 0.26 s, hand/shoulder speed 13.6,
  per-clip wrist check 95.5%).
* Geometry check of every P1 corpus (`docs/p1_fix_2026-10/step1/geometry_all.md`): **only WLASL v1
  FAILS** (all six assertions: shoulder 0.019, hand "wrists" 14-17 s from pose wrists, span 2.6 s,
  hand/shoulder speed 0.1, nose-up 0.22). All other corpora PASS incl. WLASL v2. Notes: BdSLW102a
  sentence clips are framed wider (shoulder 0.125) and only 176/300 sampled clips had >=2 detected
  frames; KArSL hands almost never missing (dropout speed share 0.07) and hand/shoulder speed ratio
  only 2.3 (19-frame clips); LSFB has no dropout (authors' cleaned poses).
* 05:06 Job J35 (debug GPU) — **Step 3 KArSL.** Re-scored all 24 existing KArSL @10% checkpoints
  (pool/xling × pseed 0-2 × ft seed 0-2, + mono, scratch; best-val ckpts selected on the signer-DEPENDENT
  val) with `tools/p1_fix/eval_ckpts.py` (no results_final.csv writes). The evaluator reproduces every
  stored val accuracy exactly (sanity). Held-out signer U03 (3,526 clips, 70 classes, chance 1.43%):
  | arm | pseed | SD-val top1 | U03 top1 | U03 top5 |
  |---|---|---|---|---|
  | pool | 0/1/2 | 69.9 / 63.3 / 66.2 | 11.2 / 8.4 / 10.0 | 39.3 / 29.3 / 29.3 |
  | xling | 0/1/2 | 65.6 / 70.3 / 67.4 | 8.8 / 12.1 / 7.9 | 30.1 / 35.4 / 35.7 |
  | mono / scratch | 0 | 60.1 / 45.0 | 6.2 / 6.6 | 23.4 / 22.5 |
  G(SD val) per pseed −4.29 / +6.97 / +1.26 (pooled +1.31); G(U03) −2.36 / +3.68 / −2.02 (pooled −0.24).
  **Verdict: the KArSL law point is not a signer-independent measurement** — its accuracies (55–73%)
  collapse to 6–12% on an unseen signer; the original G is a signer-dependent quantity. KArSL can only
  be a valid SI point with a signer-disjoint val. Built (from the existing arrays, no re-extraction)
  `data/karsl_si_v2a` (train = all U01+U02 clips 7,058, val = test = U03) and `data/karsl_si_v2b`
  (train U01 3,554 / val U02 3,504 / test U03 3,526). v2a is used for the v2 law (same training
  signers as before; selection on a held-out signer); at 10% labels it is ~10 clips/class and will sit
  near the learnability floor (~10% top-1 ≈ 7× chance).
* 05:07 Job J36 (debug CPU) — **k-means dropout diagnostic** (4,000 clips of the v2 pool, K=64):
  under the original `pose_motion` features 99.4% of the squared motion mass is in detect/lose
  transition frames (11.1% of frames), 26/64 clusters are ≥50% transition frames and hold 10.9% of
  frames; under `pose_motion_masked` 0 such clusters (max 49%). Pre-set rule (≥10% of frames or ≥10%
  of clusters) -> **v2 targets use `pose_motion_masked`** (`preprocessing/compute_pretrain_targets_v2.py`,
  a wrapper adding only this mode). Consequence: the v2 backbones differ from v1 in BOTH the WLASL
  skeleton and the k-means target features (both are bug fixes; their separate effects are not
  identifiable from this rerun).
* **FPS verification** (`docs/p1_fix_2026-10/step2/fps_registry.json`, `fps_probe.json`): no extractor
  subsamples frames (WLASL: V3 T == video frame count on 25/25 range-fetched videos). Container rates:
  BdSLW60 clips 30.0 (but the unsegmented sources are 15–30 fps; segmentation re-encoded to 30 ->
  duplicated frames in some clips); BdSLW401 front 30.0 (91/91 range-fetched from Kaggle) ; LSA64 59.94;
  AUTSL 30; KArSL 30; LIBRAS 30.2; BISINDO 25.7–30 (median 30); ASL-Citizen ~30; WLASL source videos
  29.97/30 (70%), 23.98 (12%), 25 (5%) -> **WLASL_v0.3.json's fps=25 is not the container rate**; GSL
  ships JPEG frames (30 fps per the dataset description, unverifiable); LSFB authors' poses (50 fps
  documented; checked by T-vs-duration regression in the distance job); BdSLW102a assumed 30.
  distance_robustness.py's FPS map (GSL 30, BdSL 30, LSA64 60, KArSL 30, AUTSL 30) was right for 4/5 but
  multiplying all speeds of a corpus by a constant only rescales, and the pool source was never rescaled.
* Created `data/ssl_pool_manifest_bdsl_asl_v2.json` (75,589 clips; identical to v1 except 21,083 WLASL
  paths -> `data/wlasl_pose_cache_v2`), `config/p1fix/bdsl_shubert_pretrain_{bdsl_asl,xlingual}_v2.yaml`
  (identical except manifest + targets), fine-tune configs `config/p1fix/<stem>_f10_v2.yaml` (identical
  P1 @10% recipe; KArSL -> karsl_si_v2a; BISINDO gets the recipe its original f10 runs used; results ->
  `results_p1fix_v2.csv`) and test configs `<stem>_test_v2.yaml` (all except KArSL, whose test = val).
  All validated against main.py's argparse (no WRONG ARG).
* Submit cap: the user's other projects hold the 100-job cap -> `scripts/hpc/p1_fix/sbatch_retry.sh`
  retries every 120 s.
* 05:17 **PREREG written and locked** (`docs/p1_fix_2026-10/PREREG.md`, sha256
  bd28232ae8703eaf28cf6b261238f8eb41f5f456970ad497fe4d403d6c95fd75, mtime 2026-10-22 05:17:19 EDT,
  chmod a-w) — before targets, backbones, fine-tunes or full-sample v2 distances existed
  (`results_p1fix_v2.csv` did not exist). Distance values will be added as a timestamped addendum
  (separate file) before any fine-tune result exists.
* Wrote `tools/compute_lang_distance_v2.py` (smoke-tested on 40-60 clips; numbers discarded) and
  `tools/p1_fix/refit_v2.py` (debugged on the already-public v1 pseed 0-2 data: reproduces item6's
  picture — slope CI spans 0, LOLO 2/7).
* Repacked pretraining for the shared submit cap: ONE 2-GPU job (lane 0 = pool seeds 0-2, lane 1 =
  xling seeds 0-2, 3 runs/GPU) and one 2-GPU fine-tune job per pretraining seed (lane per arm, 40 runs,
  ≤3/GPU). v2 pretrain configs checkpoint every epoch (`save_interval: 1`) so a requeue loses <1 epoch.
  Launch chain `scripts/hpc/p1_fix/launch_step5.sh` (afterok on the targets job).
* 05:37 Job J37 (debug CPU) — first full distance run. Side results: **LSFB frame rate confirmed**
  (T = 3.8 + 49.8·duration, n=2,000 -> 50 fps). **KArSL U03 differs in capture**: geometry PASSES for all
  splits, but U03 clips are longer (median T 28 vs 19) and have the hands undetected far more often
  (missing R/L 0.36/0.33 vs 0.02/0.01 for U01/U02; dropout speed share 0.23 vs 0.07) — so the U03
  collapse is signer + capture shift, not only signer identity.
  **Bug found in my own tool before any use:** for dict-specified (v1-replica) sources the draw
  subsample used the loaded-pool shares (3,000 WLASL vs only 1,539 BdSLW60-pretrain clips available ->
  66/34) instead of the v1 tool's 50/50; the manifest-based sources were correct. Fixed (stratified by
  the source's mixing weights) and rerun (job p1fix_step2c, which also reruns the unmodified v1 tool as
  a replication check). First-run outputs kept as `step2/run1_srcweight_bug_*` (not used).
* 05:48 Job J38: v2 distances recomputed; the unmodified v1 tool reproduces the paper values
  exactly, and the fixed replica row (old descriptor, 400 WLASL-v1 + 400 BdSLW60, pair bandwidth) matches
  them within draw SD (e.g. BdSL 0.1409±0.0066 vs 0.1429; LSFB 0.4553±0.0128 vs 0.4798).
  **PREREG addendum A** (frozen v2 x-axis, sha256 e19f78f8…, written 05:48:22, before any v2 backbone
  or fine-tune): BdSL 0.0810, GSL 0.0913, LIBRAS 0.1373, LSFB 0.1447, AUTSL 0.1557, LSA64 0.1572,
  BISINDO 0.3959, KArSL 0.4041 (MMD²_u; draw SD 0.007–0.017). Under the fixed inputs the v1 "far"
  targets AUTSL (0.41) and LSFB (0.48) become as close as LSA64/LIBRAS; only KArSL and BISINDO stay far.
  Decomposition in `step2/distances_v2_{pooled,pair}bw.md`.
* 05:56 Job J39: `data/pretrain_kmeans_targets_bdsl_asl_v2.npz` (pose_motion_masked, K=64, cap 200k/source;
  fit corpus 350,211 frames: BdSLW401 200,000 (capped from 402,585), WLASL-v2 125,205, BdSLW102a 25,006);
  74,169 clips with targets (v1 pool: 74,168); cluster occupancy 0.14%–5.3%.
* 06:00 Submitted: J41 p1fix_pt (2 GPUs, pool+xling × pseeds 0-2), fine-tunes J42/J43/J44 (PS 0/1/2, afterok J41). Pretrain pending (Priority).

## 2026-10-22 / 23 (completion; entries added 2026-10-23 from the SLURM logs)

* 2026-10-22 15:06 Job J41 (pretrain) requeued once at its wall-time limit; resumed from per-epoch checkpoints.
  22:48 `=== v2 pretrain done (incomplete: 0) ===` -> 6 backbones (pool/xling x pseeds 0-2).
* 2026-10-23 06:32 Fine-tune jobs J42/J43/J44 each requeued once at the wall-time limit (resume).
  Done: ps0 07:09, ps2 08:19, ps1 08:24, each `(incomplete: 0)`. `results_p1fix_v2.csv`: 240 law runs + 210 TEST evals.
* 08:43 Job J45 (compute node): `tools/p1_fix/refit_v2.py --out docs/p1_fix_2026-10/results`, run once.
  PREREG.md sha256 re-verified unchanged (bd28232a...). **VERDICT: NOT SUPPORTED** (b -4.59 [-11.56, +2.61],
  P(b<0) 0.905 -> C1 fails; LOLO 6/8 -> C2 passes). 1 exclusion (KArSL ps0 pool seed1, gross failure 5.10 < 0.5 x 11.29).
  Sensitivities: S9 (ft seeds 0-2) SUPPORTED; S8 (TEST) slope +12.93 [+1.61, +24.47] (opposite sign).
* Wrote `RESULTS.md`, `README.md`; filled the v2 placeholders in the paper draft; `fig_v2law.pdf` from
  `tools/paper/make_figs.py --only v2`.
* 2026-10-23 (audit round): applied the independent paper audit's fixes to the paper draft and `RESULTS.md`:
  t-intervals over pseeds + Holm (make_figs.py `v2_numbers`, asserted: no t-CI excludes 0; min Holm p 0.23 bootstrap);
  v_l-omission caveat; interrupted-evaluation disclosure (10/240 runs, verified from results_p1fix_v2.csv: 6 x 21 rows,
  4 x 19 rows); pre-registration commit 2e745eb (2026-10-22 15:53) precedes the first v2 result (22:53). v1 exact REML
  recomputed: `tools/paper/exact_reml_v1.py` -> `results/provenance/v1_exact_reml_2026-10-23.txt`.
