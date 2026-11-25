@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base). Safe to rerun after an interruption: finished steps are skipped.
REM Stage 7D: history length 1 and 2 (needs stage 7A). About 120 GPU runs.
python run_scope.py --project-root .. --dataset deakin_fold2 --setting history1 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_history1 || exit /b 1
if not exist ..\stage7\history1_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2 --runs ..\scope_deakin_fold2_history1 --history-lr ..\history_lr_deakin_fold2 --candidates tgae tgae_bilinear vrdag --label history1 --out ..\stage7\history1_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting history1 --tuning ..\tuning_deakin --out ..\scope_deakin_history1 || exit /b 1
if not exist ..\stage7\history1_deakin.json (
  python compare_best.py --project-root .. --dataset deakin --runs ..\scope_deakin_history1 --history-lr ..\history_lr_deakin --candidates tgae tgae_bilinear vrdag --label history1 --out ..\stage7\history1_deakin.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting history1 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_history1 || exit /b 1
if not exist ..\stage7\history1_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2 --runs ..\scope_unsw_fold2_history1 --history-lr ..\history_lr_unsw_fold2 --candidates tgae tgae_bilinear vrdag --label history1 --out ..\stage7\history1_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting history1 --tuning ..\tuning_unsw --out ..\scope_unsw_history1 || exit /b 1
if not exist ..\stage7\history1_unsw.json (
  python compare_best.py --project-root .. --dataset unsw --runs ..\scope_unsw_history1 --history-lr ..\history_lr_unsw --candidates tgae tgae_bilinear vrdag --label history1 --out ..\stage7\history1_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting history1 --files ..\stage7\history1_unsw.json ..\stage7\history1_deakin.json ..\stage7\history1_deakin_fold2.json ..\stage7\history1_unsw_fold2.json || exit /b 1
python run_scope.py --project-root .. --dataset deakin_fold2 --setting history2 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_history2 || exit /b 1
if not exist ..\stage7\history2_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2 --runs ..\scope_deakin_fold2_history2 --history-lr ..\history_lr_deakin_fold2 --candidates tgae tgae_bilinear vrdag --label history2 --out ..\stage7\history2_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting history2 --tuning ..\tuning_deakin --out ..\scope_deakin_history2 || exit /b 1
if not exist ..\stage7\history2_deakin.json (
  python compare_best.py --project-root .. --dataset deakin --runs ..\scope_deakin_history2 --history-lr ..\history_lr_deakin --candidates tgae tgae_bilinear vrdag --label history2 --out ..\stage7\history2_deakin.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting history2 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_history2 || exit /b 1
if not exist ..\stage7\history2_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2 --runs ..\scope_unsw_fold2_history2 --history-lr ..\history_lr_unsw_fold2 --candidates tgae tgae_bilinear vrdag --label history2 --out ..\stage7\history2_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting history2 --tuning ..\tuning_unsw --out ..\scope_unsw_history2 || exit /b 1
if not exist ..\stage7\history2_unsw.json (
  python compare_best.py --project-root .. --dataset unsw --runs ..\scope_unsw_history2 --history-lr ..\history_lr_unsw --candidates tgae tgae_bilinear vrdag --label history2 --out ..\stage7\history2_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting history2 --files ..\stage7\history2_unsw.json ..\stage7\history2_deakin.json ..\stage7\history2_deakin_fold2.json ..\stage7\history2_unsw_fold2.json || exit /b 1
echo Stage 7D finished.
