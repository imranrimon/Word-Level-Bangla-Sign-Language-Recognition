@echo off
cd /d "%~dp0.."

echo ===================================================
echo RESUME REMAINING GNN EXPERIMENTS
echo (Experiments 9-14: GNN Variants)
echo ===================================================
echo.

REM Activate conda environment
call conda activate bdsl_graph

REM Run the remaining experiments using the resume config
echo Starting from Attention GNN experiments...
python -u tools/run_experiments.py --config experiments_resume.yaml

if errorlevel 1 goto error

echo.
echo ===================================================
echo ALL REMAINING EXPERIMENTS COMPLETED
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
