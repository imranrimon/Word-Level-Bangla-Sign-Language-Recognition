@echo off
REM B2 recovery: seeds 0 and 1 already completed with test rows; this runs
REM only the remaining seed 2 fresh, then writes logs\rgb_done.marker so the
REM queued I3D runner (run_i3d_bdsl_si.bat) fires afterward.
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
if exist logs\rgb_done.marker del logs\rgb_done.marker

"%PY%" -u -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml --seed 2 -Experiment_name rgb_s3d_bdsl_si_seed2 > logs\rgb_s3d_seed2.log 2>&1
if errorlevel 1 echo SEED 2 FAILED >> logs\rgb_failures.log

echo done > logs\rgb_done.marker
