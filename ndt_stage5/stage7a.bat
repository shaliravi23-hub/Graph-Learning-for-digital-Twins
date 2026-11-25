@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base). Safe to rerun after an interruption: finished steps are skipped.
REM Stage 7A: tuning parity (about 200 GPU runs), then the default and tuned comparisons (CPU).
python tune_models.py --project-root .. --dataset deakin_fold2 --existing-runs ..\seeds_deakin_fold2 --out ..\tuning_deakin_fold2 || exit /b 1
python tune_models.py --project-root .. --dataset deakin --existing-runs ..\seeds_deakin ..\seeds_deakin_decoder --out ..\tuning_deakin || exit /b 1
python tune_models.py --project-root .. --dataset unsw_fold2 --existing-runs ..\seeds_unsw_fold2 --out ..\tuning_unsw_fold2 || exit /b 1
python tune_models.py --project-root .. --dataset unsw --existing-runs ..\seeds_unsw ..\seeds_unsw_decoder --out ..\tuning_unsw || exit /b 1
if not exist ..\stage7\default_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2 --runs ..\seeds_deakin_fold2 --history-lr ..\history_lr_deakin_fold2 --label default --out ..\stage7\default_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\tuned_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2 --runs ..\tuning_deakin_fold2\tuned --history-lr ..\history_lr_deakin_fold2 --label tuned --out ..\stage7\tuned_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\default_deakin.json (
  python compare_best.py --project-root .. --dataset deakin --runs ..\seeds_deakin ..\seeds_deakin_decoder --history-lr ..\history_lr_deakin --label default --out ..\stage7\default_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\tuned_deakin.json (
  python compare_best.py --project-root .. --dataset deakin --runs ..\tuning_deakin\tuned --history-lr ..\history_lr_deakin --label tuned --out ..\stage7\tuned_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\default_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2 --runs ..\seeds_unsw_fold2 --history-lr ..\history_lr_unsw_fold2 --label default --out ..\stage7\default_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\tuned_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2 --runs ..\tuning_unsw_fold2\tuned --history-lr ..\history_lr_unsw_fold2 --label tuned --out ..\stage7\tuned_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\default_unsw.json (
  python compare_best.py --project-root .. --dataset unsw --runs ..\seeds_unsw ..\seeds_unsw_decoder --history-lr ..\history_lr_unsw --label default --out ..\stage7\default_unsw.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\stage7\tuned_unsw.json (
  python compare_best.py --project-root .. --dataset unsw --runs ..\tuning_unsw\tuned --history-lr ..\history_lr_unsw --label tuned --out ..\stage7\tuned_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting default --files ..\stage7\default_unsw.json ..\stage7\default_deakin.json ..\stage7\default_deakin_fold2.json ..\stage7\default_unsw_fold2.json || exit /b 1
python summarize_stage7.py --setting tuned --files ..\stage7\tuned_unsw.json ..\stage7\tuned_deakin.json ..\stage7\tuned_deakin_fold2.json ..\stage7\tuned_unsw_fold2.json || exit /b 1
echo Stage 7A finished.
