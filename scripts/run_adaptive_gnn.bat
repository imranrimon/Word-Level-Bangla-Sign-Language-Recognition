@echo off
cd /d "%~dp0.."
call conda activate bdsl_graph
python -u main.py --config config/bdsl_adaptive_gnn.yaml
pause
