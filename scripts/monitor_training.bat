@echo off
cd /d "%~dp0.."

REM Activate conda environment
call conda activate bdsl_graph

REM Run the monitoring script
python tools/monitor_training.py --interval 10

pause
