@echo off
REM ==========================================================================
REM LOSO headline-variance sweep (audit fix #2, "largest single credibility
REM lift"). Leave-one-signer-out over all 11 full-vocab signers for the headline
REM BlockGCN backbone -> report mean +/- std ACROSS FOLDS (signer noise) as the
REM headline variance, not seed noise. 11 folds x 1 seed (the "smart sweep";
REM extend --seeds to 0 1 2 for the full version). Each fold's data is
REM regenerated once from data\bdsl_cache and cached under data\bdsl_si_loso\.
REM
REM GPU-bound: waits for Option C (logs\optionc_done.marker) so it runs LAST,
REM after the baseline table + interventions + SSL fine-tunes. Launched detached
REM (scripts\detach_run.py) so console-control events can't kill it.
REM ==========================================================================
cd /d "%~dp0.."
set PY=C:\Users\rimon\anaconda3\envs\bdsl_graph\python.exe
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
set KMP_DUPLICATE_LIB_OK=TRUE
if exist logs\loso_done.marker exit /b 0
if exist logs\loso.lock exit /b 0
echo %date% %time% > logs\loso.lock

:waitloop
if not exist logs\optionc_done.marker (
  ping -n 901 127.0.0.1 >nul
  goto waitloop
)

echo [%date% %time%] LOSO headline sweep START/RESUME >> logs\loso_program.log
"%PY%" -u tools/run_loso.py --single config/bdsl_block_gcn_si.yaml ^
  --test-signers 1 4 5 6 8 9 11 12 2 13 15 --seeds 0 --skip-existing ^
  > logs\loso_block_gcn.log 2>&1
if errorlevel 1 echo LOSO block_gcn FAILED >> logs\loso_failures.log

del logs\loso.lock 2>nul
echo done > logs\loso_done.marker
echo [%date% %time%] LOSO headline sweep DONE >> logs\loso_program.log
