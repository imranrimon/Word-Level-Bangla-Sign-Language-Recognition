@echo off
REM Stage B.1 - DINOv2 hand/face-crop feature extraction for the SI split.
REM CPU-bound (MediaPipe Holistic + video decode dominate; DINOv2-small barely
REM touches the GPU), so this runs CONCURRENTLY with the GPU training program.
REM Per-clip .npz caching in data/bdsl_dino_cache makes it fully resumable, so
REM a reboot mid-run costs nothing already computed. Launch via Task Scheduler.
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\envs\bdsl_graph\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
REM Reboot-resilient: already-complete -> do nothing; otherwise the per-clip
REM .npz cache lets extraction resume from where it stopped.
if exist logs\stage_b1_done.marker exit /b 0

"%PY%" -u preprocessing/extract_dinov2_features.py ^
  --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" ^
  --output-dir data/bdsl_si_dino ^
  --cache-dir data/bdsl_dino_cache ^
  --splits train val test ^
  --device cuda ^
  --model vit_small_patch14_dinov2.lvd142m ^
  --batch-size 64 > logs\stage_b1_dino.log 2>&1

if errorlevel 1 (
  echo STAGE B.1 FAILED >> logs\stage_b1_failures.log
  echo failed > logs\stage_b1_failed.marker
  exit /b 1
)
echo done > logs\stage_b1_done.marker
