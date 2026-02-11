@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer BONE STREAM TRAINING
echo ===================================================
echo.
echo Launching Bone Video Training (250 Epochs) in new window...
start "SLGTformer - BONE Video Training" cmd /k "call conda activate bdsl_graph && python -u main.py --config config/bdsl_bone.yaml"

echo.
echo Launching Bone Image Training (100 Epochs) in new window...
start "SLGTformer - BONE Image Training" cmd /k "call conda activate bdsl_graph && python -u main.py --config config/bdsl_img_bone.yaml"

echo.
echo ===================================================
echo Done! Please check the two new windows.
echo ===================================================
pause
