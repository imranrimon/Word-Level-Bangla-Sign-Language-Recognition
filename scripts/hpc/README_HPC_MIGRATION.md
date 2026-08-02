# Moving the project to HPC — current-state walkthrough (2026-07-30)

Complements the older `HPC_LAUNCH_GUIDE.md` (still valid for the execution DAG,
pitfalls §4, and methodology reminders §5) with what changed since: new run
registries, Option C SSL already pretrained, the top-tier plan, and actual SLURM
job scripts. Do the 4 parts in order.

## 1. Code  (git — versioned; recommended)
You have ~139 uncommitted changes (recent configs/scripts/docs). Commit + push,
then clone on HPC:
```bash
# local:
git add -A && git commit -m "SI program, Option C, top-tier plan, HPC scripts"
git push origin main
# HPC:
git clone https://github.com/imranrimon/Word-Level-Bangla-Sign-Language-Recognition.git SLGTformer
```
`results_final.csv` is tracked, so your completed-run results travel with the code.
(Alternative if you'd rather not commit yet: `rsync -avz --exclude='.git' --exclude='data/'
--exclude='work_dir/' --exclude='external/' F:/SLGTformer/ <host>:~/SLGTformer/`.)

## 2. Data  (gitignored → transfer separately; ~5–6 GB minimal)
Skip the 11 GB `bdsl_si_dino` — re-extract it on HPC (a GPU step, guide §G3). Move
only what can't be regenerated cheaply:
```bash
# from local (Git Bash / WSL), one rsync per needed dir:
HPC=<user>@<host>:~/SLGTformer
rsync -avz data/bdsl_si            $HPC/data/          # 1.6 G  SI NPY bundle
rsync -avz data/bdsl_cache         $HPC/data/          # 91 M   pose cache (LOSO needs this)
rsync -avz data/bdslw401_si        $HPC/data/          # 4.7 G  B3 cotrain aux
rsync -avz data/bdslw401_pose_cache_front $HPC/data/   # SSL pool
rsync -avz data/ssl_pool_manifest*.json data/pretrain_kmeans_targets*.npz $HPC/data/
rsync -avz data/bdsl60_singletrial_eval $HPC/data/     # T6
# SSL backbones already trained locally (5.5 MB each) — move so you skip re-pretraining:
rsync -avz work_dir/bdsl_shubert_pretrain{,_bdsl_only}/pretrained_epoch30.pt \
      $HPC/work_dir/  # (recreate the two dirs on HPC first)
```
After transfer, verify byte-equality with the cluster-occupancy check in guide §1.

## 3. Environment
```bash
conda env create -f environment.yml && conda activate bdsl_graph
pip install mediapipe==0.10.18            # pip-only on Linux (guide §4)
python -c "import torch;print(torch.cuda.is_available(),torch.cuda.device_count())"
python -m pytest tests/ -q                # sanity
```

## 4. Launch via SLURM (edit the `<...>` #SBATCH lines first)
```bash
# smoke first (guide §7), then chain the real jobs so each waits for the last:
jid1=$(sbatch --parsable scripts/hpc/slurm_si_sweep.sbatch)
# Option C SSL fine-tune after the sweep (put the 6 fine-tunes from guide §G2 in an sbatch):
jid2=$(sbatch --parsable --dependency=afterok:$jid1 scripts/hpc/slurm_ssl_finetune.sbatch)
# LOSO array (11 folds in PARALLEL) after Option C:
sbatch --dependency=afterok:$jid2 scripts/hpc/slurm_loso_array.sbatch
```
`squeue -u $USER` to watch; `results_final.csv` fills as jobs finish; aggregate with
`tools/summarize_seeds.py`.

## What HPC unlocks vs the Windows box
- **No console-kill problem** — SLURM jobs are isolated; the `detach_run.py` hack is
  unnecessary here.
- **Parallelism** — run the 11 Option-A models and the 11 LOSO folds concurrently
  across GPUs instead of serially (days → hours).
- **B1 needs no WSL2** — HPC is Linux, so Uni-Sign + deepspeed run natively. Use the
  science steps in `docs/B1_FOUNDATION_BASELINES.md` directly (ignore the WSL2 runbook).

## Decide before you start
- **Cut over or mirror?** If you fully move, stop the local `SLGT_*` tasks
  (`Unregister-ScheduledTask`) so they don't keep burning the local GPU.
- **Scheduler**: these scripts are SLURM. If your HPC is PBS/LSF, the job directives
  differ — tell me which and I'll convert them.
