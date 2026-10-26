# Cluster launchers (SLURM)

The SLURM scripts that produced the paper's runs. Before submitting, set the partition on the
`#SBATCH -p <PARTITION>` line (`sinfo -s`) and, if your cluster needs it, an account
(`#SBATCH -A ...`). Every script `source`s `~/miniconda3/etc/profile.d/conda.sh`, activates the
`bdsl_graph` environment and runs from `$SLURM_SUBMIT_DIR`, so submit from the repository root.
Long jobs run in 8-hour chunks: SLURM sends `USR1` 15 minutes before the limit, the job requeues
itself, and pretraining (`main_pretrain.py`, `train_state.pt`) and fine-tuning
(`tools/run_resumable_seed.py`) resume from their latest checkpoint.

| script | purpose |
|---|---|
| `slurm_ssl_pretrain_param.sbatch` | one pretraining run: `sbatch --export=ALL,CFG=config/<name>.yaml ...` |
| `slurm_p1_pool_pretrain_seed.sbatch` | pool / xling backbones at further pretraining seeds (`work_dir/<exp>_pseed<p>/`) |
| `slurm_p1_pretrain_seed.sbatch` | mono / xling backbones at pretraining seeds 1-2 (BdSL headline) |
| `slurm_p1_law_pseed_ft.sbatch`, `slurm_p1_law_pseed_ft_packed.sbatch` | all 10%-label law fine-tunes of one backbone (`ARM=pool|xling`, `PS=<pseed>`), with test-signer scoring for BdSL, LSA64, AUTSL |
| `slurm_p1_pseed_ft.sbatch` | BdSL headline fine-tunes of a list of backbones (`RUNS="<tag>=<weights> ..."`), with test-signer scoring |
| `slurm_p1_ckpt_predictor.sbatch` | feature extraction and scoring for the checkpoint-aware predictor study |
| `slurm_logme_transfer.sbatch` | LogME of single-source backbones on each target (`results/logme_transfer.json`) |
| `slurm_lambdad_coral_pretrain.sbatch`, `slurm_lambdad_coral_finetune.sbatch` | original (leaky-pool) lambda(d) gate and CORAL arms on AUTSL |
| `audit_reruns_2026-10/` | leak-free AUTSL gate/CORAL pools and true monolingual pools (`prep.py`), all their pretrains and fine-tunes (`slurm_audit_reruns.sbatch`, `JOBSET=A|B|C`), tables (`collect.py`) |
| `slurm_ft_hrnet.sbatch` | skeleton-action control fine-tunes (manifest-driven) |
| `p1_fix/` | the v2 rerun, steps 1-6: WLASL conversion and geometry (`step1_convert.sh`), frame-rate and k-means diagnostics (`step2a_probe.sh`), v2 distances (`step2b_distances.sh`, `step2c_distances.sh`), KArSL re-scoring and split v2a (`step3_karsl.sh`), masked k-means targets (`step5a_targets.sh`), pretraining (`slurm_p1fix_pretrain.sbatch`) and fine-tuning (`slurm_p1fix_ft.sbatch`) chained by `launch_step5.sh`; generic CPU job `slurm_p1fix_cpu.sbatch` |

The `*.sh` step scripts are run on a compute node through `slurm_p1fix_cpu.sbatch`
(`sbatch --export=ALL,CMD="bash scripts/hpc/p1_fix/step2b_distances.sh" ...`) or directly with
`bash` from the repository root.
