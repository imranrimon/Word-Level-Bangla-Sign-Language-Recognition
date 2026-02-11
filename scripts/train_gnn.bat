@echo off
cd /d "%~dp0.."
echo ===================================================
echo SLGTformer Comparison Study
echo Training Attention GNN (ST-GCN + Dilated TCN)
echo ===================================================

call conda activate bdsl_graph
python -u main.py --config config/bdsl_gnn.yaml

pause
