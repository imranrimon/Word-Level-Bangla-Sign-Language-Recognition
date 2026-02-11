@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer BdSL Image TRAINING ONLY
echo ===================================================
echo Preprocessing skipped (Data exists).
echo.
echo Starting Training (Images)...
call conda activate bdsl_graph
python -u main.py --config config/bdsl_img.yaml

echo.
echo Training Complete.
pause
