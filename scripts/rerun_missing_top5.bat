@echo off
cd /d "%~dp0.."

echo ===================================================
echo RE-RUNNING EXPERIMENTS FOR TOP-5 LOGGING
echo ===================================================
echo.

call conda activate bdsl_graph

python -u tools/run_experiments.py --config experiments_rerun.yaml

if errorlevel 1 goto error

echo.
echo ===================================================
echo RE-RUNS COMPLETED
echo ===================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo RE-RUN FAILED
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
