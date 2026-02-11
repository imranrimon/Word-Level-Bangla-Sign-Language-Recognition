@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer BdSL Full Training Pipeline
echo ===================================================

echo [1/3] Checking GPU Availability...
call conda run -n bdsl_graph python tools/check_gpu.py

echo.
echo [2/3] Running Preprocessing...
echo Found ~9300 videos. This process will take several hours (approx 3-6 hours depending on CPU).
echo Please do not close this window.
echo.
echo Data Root: Word_level_Bangla_Sign_Language_Dataset/BdSLW30

:: Use --no-capture-output to force real-time streaming
call conda run -n bdsl_graph --no-capture-output python -u preprocessing/preprocess_bdsl.py --data_root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" --output_dir "data/bdsl"

if errorlevel 1 (
    echo ERROR: Preprocessing failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Training...
call conda run -n bdsl_graph --no-capture-output python -u main.py --config config/bdsl.yaml

echo.
echo Pipeline Complete.
pause
