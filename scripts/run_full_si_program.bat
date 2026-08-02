@echo off
REM ==========================================================================
REM Main-paper GPU training program (signer-independent), run SEQUENTIALLY on
REM the single RTX 8000 so runs never contend for VRAM. Launch via Task
REM Scheduler (survives session teardown / window close / logoff).
REM
REM   Phase 1  Option A pose baseline table   (11 archs x 3 seeds, skip-existing)
REM   Phase 2  SLGTFormer sub-module ablations (3 configs x 3 seeds)
REM   Phase 3  Option B DINOv2 FlatTemporal    (A12; needs Stage B.1 features)
REM   Phase 4  B2 RGB baseline (S3D seed2) + I3D
REM   Phase 5  B3 co-training ablation         (3 seeds; longest, last)
REM
REM Progress survives everything via results_final.csv rows + per-phase logs.
REM ==========================================================================
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\envs\bdsl_graph\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
REM Reboot-resilient: if the whole program already finished, do nothing (an
REM AtStartup trigger must not re-run a completed program). If it did NOT
REM finish, fall through and let --skip-existing / the caches resume cheaply.
if exist logs\full_si_done.marker exit /b 0
REM concurrency lock: refuse a 2nd instance (e.g. AtLogOn re-fire while running).
REM Cleared at completion; if a run is ever hard-killed, delete logs\full_si.lock.
if exist logs\full_si.lock exit /b 0
echo %date% %time% > logs\full_si.lock
echo [%date% %time%] START/RESUME full SI program >> logs\full_si_program.log

echo [%date% %time%] PHASE 1: Option A pose baseline table >> logs\full_si_program.log
"%PY%" -u tools/run_multiseed.py --config experiments_si_main.yaml --seeds 0 1 2 --skip-existing > logs\full_si_optA.log 2>&1

echo [%date% %time%] PHASE 2: SLGTFormer ablations >> logs\full_si_program.log
"%PY%" -u tools/run_multiseed.py --config experiments_si_ablations.yaml --seeds 0 1 2 --skip-existing > logs\full_si_ablations.log 2>&1

echo [%date% %time%] PHASE 3: Option B DINOv2 FlatTemporal (A12) >> logs\full_si_program.log
if exist data\bdsl_si_dino\train_data.npy (
  "%PY%" -u tools/run_multiseed.py --single config/bdsl_dino_temporal_si.yaml --seeds 0 1 2 --skip-existing > logs\full_si_dino.log 2>&1
) else (
  echo A12 SKIPPED: data\bdsl_si_dino\train_data.npy missing - Stage B.1 not finished; run separately later >> logs\full_si_program.log
)

echo [%date% %time%] PHASE 4: B2 RGB (S3D seed2) + I3D >> logs\full_si_program.log
call scripts\run_b2_rgb_seed2.bat
call scripts\run_i3d_bdsl_si.bat

echo [%date% %time%] PHASE 5: B3 co-training (3 seeds) >> logs\full_si_program.log
call scripts\run_b3_cotrain_seeds.bat

del logs\full_si.lock 2>nul
echo done > logs\full_si_done.marker
echo [%date% %time%] DONE full SI program >> logs\full_si_program.log
