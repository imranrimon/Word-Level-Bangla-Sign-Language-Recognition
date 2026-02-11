@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer BdSL IMAGE DATASET Pipeline
echo ===================================================

echo [1/3] Checking GPU Availability...
call conda run -n bdsl_graph python tools/check_gpu.py

echo.
echo [2/3] Running Image Preprocessing...
echo Data Root: Word_level_Bangla_Sign_Language_Dataset/Bangla Sign Language Dataset/RESIZED_DATASET
call conda run -n bdsl_graph --no-capture-output python -u preprocessing/preprocess_bdsl_images.py --data_root "Word_level_Bangla_Sign_Language_Dataset/Bangla Sign Language Dataset/RESIZED_DATASET" --output_dir "data/bdsl_img"

if errorlevel 1 (
    echo ERROR: Preprocessing failed. Exiting.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Training (Images)...
call conda run -n bdsl_graph --no-capture-output python -u main.py --config config/bdsl_img.yaml

echo.
echo Pipeline Complete.
pause
