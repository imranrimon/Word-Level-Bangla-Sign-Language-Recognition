@echo off
REM B2: RGB S3D baseline. First recovers seed 0's missing test-set row from
REM its saved best checkpoint (eval-only), then trains seeds 1-2 fresh.
REM Detached-safe; writes logs\rgb_s3d_seed<N>.log, failures to
REM logs\rgb_failures.log, and logs\rgb_done.marker at the end.
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\python.exe
if exist logs\rgb_done.marker del logs\rgb_done.marker

"%PY%" -u -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml --seed 0 -Experiment_name rgb_s3d_bdsl_si_seed0 --eval-only > logs\rgb_s3d_seed0_evalonly.log 2>&1
if errorlevel 1 echo SEED 0 EVAL-ONLY FAILED >> logs\rgb_failures.log

for %%s in (1 2) do (
  "%PY%" -u -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml --seed %%s -Experiment_name rgb_s3d_bdsl_si_seed%%s > logs\rgb_s3d_seed%%s.log 2>&1
  if errorlevel 1 echo SEED %%s FAILED >> logs\rgb_failures.log
)
echo done > logs\rgb_done.marker
