@echo off
REM I3D RGB row (ASL-Citizen parity), seed 0. Self-sequencing: waits for the
REM S3D pipeline to finish (logs\rgb_done.marker) so only one decode-bound
REM trainer runs at a time, and exits gracefully if the user has not yet
REM downloaded the I3D checkpoint. Extend the `for` list for more seeds.
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE

:waitloop
if not exist logs\rgb_done.marker (
  ping -n 601 127.0.0.1 >nul
  goto waitloop
)

if not exist path4_rgb_baseline\weights\rgb_imagenet.pt (
  echo I3D SKIPPED: weights missing at path4_rgb_baseline\weights\rgb_imagenet.pt >> logs\rgb_failures.log
  exit /b 1
)

for %%s in (0) do (
  "%PY%" -u -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/i3d_bdsl_si.yaml --seed %%s -Experiment_name rgb_i3d_bdsl_si_seed%%s > logs\rgb_i3d_seed%%s.log 2>&1
  if errorlevel 1 echo I3D SEED %%s FAILED >> logs\rgb_failures.log
)
echo done > logs\i3d_done.marker
