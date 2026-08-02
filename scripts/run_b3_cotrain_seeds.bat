@echo off
REM B3: co-training ablation, seeds 0-2 sequentially. Detached-safe (run via
REM Task Scheduler or a double-click); writes logs\cotrain_seed<N>.log,
REM failure lines to logs\cotrain_failures.log, and logs\cotrain_done.marker
REM at the end.
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\python.exe
REM Prevent the Intel Fortran runtime (via numpy/MKL) from aborting training
REM with "forrtl: error (200): program aborting due to window-CLOSE event"
REM when the console window is closed / the launching session is torn down.
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
if exist logs\cotrain_done.marker del logs\cotrain_done.marker
for %%s in (0 1 2) do (
  "%PY%" -u main_cotrain.py --config config/bdsl_block_gcn_cotrain_si.yaml --seed %%s -Experiment_name bdsl_block_gcn_cotrain_si_seed%%s > logs\cotrain_seed%%s.log 2>&1
  if errorlevel 1 echo SEED %%s FAILED >> logs\cotrain_failures.log
)
echo done > logs\cotrain_done.marker
