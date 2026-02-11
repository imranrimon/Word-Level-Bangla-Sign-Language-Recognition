@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer COMPLETE TRAINING PIPELINE
echo (Joint Stream + Bone Stream)
echo ===================================================

echo.
echo [1/5] Training JOINT Video Model (250 Epochs)...
call scripts\train_bdsl_only.bat
if errorlevel 1 goto error

echo.
echo [2/5] Training JOINT Image Model (100 Epochs)...
call scripts\train_bdsl_images_only.bat
if errorlevel 1 goto error

echo.
echo [3/5] Training BONE Video Model (250 Epochs)...
call conda activate bdsl_graph
python -u main.py --config config/bdsl_bone.yaml
if errorlevel 1 goto error

echo.
echo [4/5] Training BONE Image Model (100 Epochs)...
call conda activate bdsl_graph
python -u main.py --config config/bdsl_img_bone.yaml
if errorlevel 1 goto error

echo.
echo [5/5] Fusing Scores...
echo Fusing Video Scores...
python tools/fuse_scores.py --joint_dir work_dir/bdsl_graph --bone_dir work_dir/bdsl_bone
echo Fusing Image Scores...
python tools/fuse_scores.py --joint_dir work_dir/bdsl_img --bone_dir work_dir/bdsl_img_bone

echo.
echo ===================================================
echo ALL TRAINING COMPLETE
echo ===================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ERROR ENCOUNTERED. EXECUTION HALTED.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
