# P1 fix-and-rerun program (2026-10): index

The program fixed the inputs of the P1 distance-gating law and re-tested the law under a pre-registered
analysis. The fixed inputs were:

* the WLASL skeleton scramble;
* the distance tool's source, descriptor, frame rate and estimator;
* dropout-dominated k-means targets;
* KArSL's signer-dependent validation set;
* the post-hoc exclusion of BISINDO.

**Outcome: NOT SUPPORTED.** The slope's 95% CrI is [−11.56, +2.61] with P(b<0) = 0.905, so C1 fails; LOLO is
6/8, so C2 passes at the threshold. See `RESULTS.md`. All artefacts are versioned (`_v2`), and no pre-existing
data file, cache, manifest, config or core script was modified in place. The chronological record is
`LOG.md`.

## Read in this order

| file | what |
|---|---|
| `PREREG.md` | Pre-registration of the v2 rerun: design, 8 languages, distance definition, exclusions, model, binding rule C1 ∧ C2, sensitivities S1–S9. Read-only. |
| `PREREG.sha256` | SHA-256 + mtime of `PREREG.md` (`bd28232a…`, 2026-10-22 05:17:19 EDT) and of Addendum A (`e19f78f8…`, 05:48:22). |
| `PREREG_addendum_A_distances.md` | The frozen v2 x-axis, d_ℓ per language. Written before any v2 backbone or fine-tune. Read-only. |
| `LOG.md` | Running log: every step, job, check and bug found, with timestamps. |
| `RESULTS.md` | Verdict, posterior, LOLO, per-language G, sensitivities, v1-vs-v2 comparison, completeness, deviations. |
| `results/refit_v2.json` | Raw output of the pre-registered analysis (`tools/p1_fix/refit_v2.py`, job J45). |

## Step artefacts

| step | directory | contents (job) |
|---|---|---|
| 1 WLASL scramble | `step1/` | `verify_v1_scramble.txt`: the v1 cache equals the wrongly-indexed V3 landmarks on 300/300 clips (J34). `convert_v2.txt`: pre/post-write geometry of `data/wlasl_pose_cache_v2`. `geometry_all.{md,json,txt}`: six-assertion geometry check of every P1 corpus; only WLASL v1 fails. |
| 2 distance | `step2/` | `fps_registry.json`, `fps_probe.json`: native frame rates. `lsfb_fps_check.txt`: LSFB at 50 fps. `karsl_geometry_by_split.txt`. `distances_v2.{json,txt}` and `distances_v2_{pooled,pair}bw.md`: v2 distances and the v1→v2 decomposition (J38). `v1_tool_rerun.txt`: the unmodified v1 tool reproduces the paper values. `run1_srcweight_bug_*`: superseded first run (J37), not used. |
| 3 KArSL | `step3/` | `karsl_sdval_eval.json`, `karsl_u03_test_eval.json`: v1 checkpoints re-scored on the held-out signer U03 (J35). `make_karsl_si_v2.txt`: splits `data/karsl_si_v2a` (used) and `v2b`. |
| 5a targets | `step5/` | `kmeans_dropout_diag.json`: 99.4% of motion mass in transition frames and 26/64 dropout clusters, so `pose_motion_masked` targets (J36). |
| 5b–c rerun | — | Pretraining job J41 (pool + xling × pseeds 0–2). Fine-tunes J42 / J43 / J44 (PS 0 / 1 / 2; 240 runs + 210 TEST evals) → `results_p1fix_v2.csv`, logs in `slurm_logs/p1_fix/`. |
| 6 analysis | `results/` | `refit_v2.json` (J45, compute node, 2026-10-23 08:43). |

## Code and configs

* Conversion and checks:
  * `preprocessing/convert_wlasl_landmarks_to_pose_cache_v2.py`
  * `tools/check_pose_geometry.py`
  * `tools/p1_fix/verify_wlasl_v1_scramble.py`
  * `tools/check_run_complete.py`
* Distance:
  * `tools/compute_lang_distance_v2.py`
  * `tools/p1_fix/distance_table.py`
  * `tools/p1_fix/probe_fps.py`
* KArSL:
  * `tools/p1_fix/eval_ckpts.py`
  * `tools/p1_fix/make_karsl_si_v2.py`
* Targets:
  * `tools/p1_fix/kmeans_dropout_diag.py` (holds the pre-set rule)
  * `preprocessing/compute_pretrain_targets_v2.py`
* Analysis: `tools/p1_fix/refit_v2.py`, which reuses the grid machinery of `tools/p1_refit/refit.py`.
* Configs:
  * `config/p1fix/bdsl_shubert_pretrain_{bdsl_asl,xlingual}_v2.yaml`
  * `config/p1fix/<stem>_f10_v2.yaml` (fine-tune)
  * `config/p1fix/<stem>_test_v2.yaml` (TEST scoring; none for KArSL)
* Data:
  * `data/ssl_pool_manifest_bdsl_asl_v2.json`
  * `data/wlasl_pose_cache_v2`
  * `data/wlasl_pose_cache_bone_v2`
  * `data/pretrain_kmeans_targets_bdsl_asl_v2.npz`
  * `data/karsl_si_v2a`
  * `data/karsl_si_v2b`
* Launch scripts (`scripts/hpc/p1_fix/`):
  * `step*.sh`
  * `slurm_p1fix_{cpu,pretrain,ft}.sbatch`
  * `launch_step5.sh`
  * `sbatch_retry.sh`

## Downstream

* Paper: §7.4 and Tables `tab:v2G` / `tab:v2sens`. The figure is `results/figures/fig_v2law.pdf`,
  made by `python tools/paper/make_figs.py --only v2`. That script also computes and asserts the derived
  numbers: per-pseed OLS, the v1-vs-v2 table, and the observation that every LOLO prediction is negative.
* The preceding 7-pseed refit on the v1 backbones is `docs/p1_refit_2026-10/` (verdict: REFRAME).
