@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base). Safe to rerun after an interruption: finished steps are skipped.
REM Stage 7E: class weighting off for TGAE, VGRNN and Static VGAE (independent of 7A). About 60 GPU runs.
python run_scope.py --project-root .. --dataset deakin_fold2 --setting noweight --out ..\scope_deakin_fold2_noweight || exit /b 1
python run_scope.py --project-root .. --dataset deakin --setting noweight --out ..\scope_deakin_noweight || exit /b 1
python run_scope.py --project-root .. --dataset unsw_fold2 --setting noweight --out ..\scope_unsw_fold2_noweight || exit /b 1
python run_scope.py --project-root .. --dataset unsw --setting noweight --out ..\scope_unsw_noweight || exit /b 1
echo Stage 7E finished.
