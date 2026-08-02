@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer vs Attention GNN
echo Running ALL Experiments sequentially
echo ===================================================


call conda run -n bdsl_graph --no-capture-output python -u tools/run_experiments.py --config experiments.yaml

if errorlevel 1 goto error

echo.
echo ===================================================
echo ALL EXPERIMENTS COMPLETED
echo See results_final.csv for summary.
echo ===================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo EXPERIMENT SEQUENCE FAILED
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
