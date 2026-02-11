@echo off
cd /d "%~dp0.."

echo ===================================================
echo RUNNING ABLATION STUDY (TABLE 1 REPRODUCTION)
echo ===================================================
echo.

call conda activate bdsl_graph

REM echo [1/6] Running SLGTFormer w/o LGRPE (Video)...
REM python -u main.py --config config/bdsl_no_lgrpe.yaml
REM if errorlevel 1 echo Error in w/o LGRPE && goto error

REM echo [2/6] Running SLGTFormer w/o TTSA (Video)...
REM python -u main.py --config config/bdsl_no_ttsa.yaml
REM if errorlevel 1 echo Error in w/o TTSA && goto error

echo [3/6] Running SLGTFormer w/o PAF (Video)...
python -u main.py --config config/bdsl_no_paf.yaml
if errorlevel 1 echo Error in w/o PAF && goto error

echo [4/6] Running SLGTFormer Image w/o LGRPE...
python -u main.py --config config/bdsl_img_no_lgrpe.yaml
if errorlevel 1 echo Error in Image w/o LGRPE && goto error

echo [5/6] Running SLGTFormer Image w/o TTSA...
python -u main.py --config config/bdsl_img_no_ttsa.yaml
if errorlevel 1 echo Error in Image w/o TTSA && goto error

echo [6/6] Running SLGTFormer Image w/o PAF...
python -u main.py --config config/bdsl_img_no_paf.yaml
if errorlevel 1 echo Error in Image w/o PAF && goto error

echo.
echo ===================================================
echo ABLATION STUDY COMPLETED SUCCESSFULLY
echo ===================================================
pause
exit /b 0

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo ABLATION STUDY FAILED
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pause
exit /b 1
