@echo off
cd /d "%~dp0.."
echo ===================================================
echo LAUNCHING FULL PARALLEL TRAINING
echo ===================================================
echo.
echo Launching Video Training (250 Epochs) in new window...
start "SLGTformer - Video Training" cmd /k "scripts\train_bdsl_only.bat"

echo.
echo Launching Image Training (100 Epochs) in new window...
start "SLGTformer - Image Training" cmd /k "scripts\train_bdsl_images_only.bat"

echo.
echo ===================================================
echo Done! Please check the two new windows.
echo ===================================================
pause
