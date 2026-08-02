@echo off
REM ==========================================================================
REM Option C (T5) - SSL fine-tune. The masked-SSL BlockGCN backbones are ALREADY
REM pretrained (work_dir\bdsl_shubert_pretrain{,_bdsl_only}\pretrained_epoch30.pt,
REM 30 epochs each). This runs the only remaining step: fine-tune each backbone
REM on BdSLW60-SI x 3 seeds, using the SAME recipe as the from-scratch BlockGCN
REM baseline (fair comparison; only the init differs).
REM
REM   cross-lingual  : backbone pretrained on BdSL+ASL pool (75,589 clips)
REM   monolingual    : backbone pretrained on BdSL-only pool (54,506 clips)  [ablation]
REM
REM GPU-bound -> waits for the main SI program (logs\full_si_done.marker) so the
REM two never contend for the single RTX 8000. Launch via Task Scheduler.
REM ==========================================================================
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\envs\bdsl_graph\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
if exist logs\optionc_done.marker exit /b 0
if exist logs\optionc.lock exit /b 0
echo %date% %time% > logs\optionc.lock

:waitloop
if not exist logs\full_si_done.marker (
  REM ping-based sleep (works without a console-stdin, unlike `timeout`)
  ping -n 901 127.0.0.1 >nul
  goto waitloop
)

echo [%date% %time%] Option C SSL fine-tune START/RESUME >> logs\optionc_program.log

REM --- Cross-lingual (BdSL+ASL) backbone: the headline SSL row ---
for %%s in (0 1 2) do (
  if not exist work_dir\bdsl_block_gcn_shubert_seed%%s\eval_results (
    "%PY%" -u main.py --config config/bdsl_block_gcn_si.yaml --seed %%s ^
      -Experiment_name bdsl_block_gcn_shubert_seed%%s ^
      --weights work_dir/bdsl_shubert_pretrain/pretrained_epoch30.pt ^
      --ignore-weights fc.weight fc.bias >> logs\optionc_shubert.log 2>&1
    if errorlevel 1 echo SHUBERT SEED %%s FAILED >> logs\optionc_failures.log
  )
)

REM --- Monolingual (BdSL-only) backbone: cross-lingual ablation ---
for %%s in (0 1 2) do (
  if not exist work_dir\bdsl_block_gcn_shubert_bdsl_only_seed%%s\eval_results (
    "%PY%" -u main.py --config config/bdsl_block_gcn_si.yaml --seed %%s ^
      -Experiment_name bdsl_block_gcn_shubert_bdsl_only_seed%%s ^
      --weights work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt ^
      --ignore-weights fc.weight fc.bias >> logs\optionc_shubert_bdsl_only.log 2>&1
    if errorlevel 1 echo SHUBERT_BDSL_ONLY SEED %%s FAILED >> logs\optionc_failures.log
  )
)

del logs\optionc.lock 2>nul
echo done > logs\optionc_done.marker
echo [%date% %time%] Option C SSL fine-tune DONE >> logs\optionc_program.log
