@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer VERIFICATION RUN (Sequential)
echo ===================================================

echo [1/2] Running Video Training (3 Epochs)...
call scripts\train_bdsl_only.bat
if errorlevel 1 (
    echo Video Training Failed!
    exit /b 1
)

echo.
echo [2/2] Running Image Training (3 Epochs)...
call scripts\train_bdsl_images_only.bat
if errorlevel 1 (
    echo Image Training Failed!
    exit /b 1
)

echo.
echo ===================================================
echo Verification Complete! Verify results with:
echo python tools/visualize_results.py --work_dir work_dir/bdsl_img
echo ===================================================
pause
