@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer vs Attention GNN
echo Running ALL Experiments sequentially
echo ===================================================


"C:\Users\rimon\anaconda3\envs\bdsl_graph\python.exe" -u tools/run_experiments.py --config experiments.yaml

if errorlevel 1 goto error

echo.
echo ===================================================
echo ALL EXPERIMENTS COMPLETED
echo See results.csv for summary.
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
