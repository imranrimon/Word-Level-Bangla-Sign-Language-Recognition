@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer BdSL Video TRAINING ONLY
echo ===================================================
echo Preprocessing skipped (Data exists).
echo.
echo Starting Training...
call conda activate bdsl_graph
python -u main.py --config config/bdsl.yaml

echo.
echo Training Complete.
pause
