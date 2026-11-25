@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base). Safe to rerun after an interruption: finished steps are skipped.
REM Stage 7B: 600 s and 900 s windows (needs stage 7A). About 120 GPU runs.
if not exist ..\dataset_ready_deakin_fold2\600s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset deakin_fold2 --seconds 600 || exit /b 1
)
if not exist ..\history_lr_deakin_fold2_600s\fit.json (
  python history_lr.py --project-root .. --dataset deakin_fold2_600s --out ..\history_lr_deakin_fold2_600s || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin_fold2 --setting window600 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_window600 || exit /b 1
if not exist ..\stage7\window600_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2_600s --runs ..\scope_deakin_fold2_window600 --history-lr ..\history_lr_deakin_fold2_600s --candidates tgae tgae_bilinear vrdag --label window600 --out ..\stage7\window600_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_deakin_chronological\600s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset deakin --seconds 600 || exit /b 1
)
if not exist ..\history_lr_deakin_600s\fit.json (
  python history_lr.py --project-root .. --dataset deakin_600s --out ..\history_lr_deakin_600s || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting window600 --tuning ..\tuning_deakin --out ..\scope_deakin_window600 || exit /b 1
if not exist ..\stage7\window600_deakin.json (
  python compare_best.py --project-root .. --dataset deakin_600s --runs ..\scope_deakin_window600 --history-lr ..\history_lr_deakin_600s --candidates tgae tgae_bilinear vrdag --label window600 --out ..\stage7\window600_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_unsw_fold2\600s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset unsw_fold2 --seconds 600 || exit /b 1
)
if not exist ..\history_lr_unsw_fold2_600s\fit.json (
  python history_lr.py --project-root .. --dataset unsw_fold2_600s --out ..\history_lr_unsw_fold2_600s || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting window600 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_window600 || exit /b 1
if not exist ..\stage7\window600_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2_600s --runs ..\scope_unsw_fold2_window600 --history-lr ..\history_lr_unsw_fold2_600s --candidates tgae tgae_bilinear vrdag --label window600 --out ..\stage7\window600_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready\600s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset unsw --seconds 600 || exit /b 1
)
if not exist ..\history_lr_unsw_600s\fit.json (
  python history_lr.py --project-root .. --dataset unsw_600s --out ..\history_lr_unsw_600s || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting window600 --tuning ..\tuning_unsw --out ..\scope_unsw_window600 || exit /b 1
if not exist ..\stage7\window600_unsw.json (
  python compare_best.py --project-root .. --dataset unsw_600s --runs ..\scope_unsw_window600 --history-lr ..\history_lr_unsw_600s --candidates tgae tgae_bilinear vrdag --label window600 --out ..\stage7\window600_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting window600 --files ..\stage7\window600_unsw.json ..\stage7\window600_deakin.json ..\stage7\window600_deakin_fold2.json ..\stage7\window600_unsw_fold2.json || exit /b 1
if not exist ..\dataset_ready_deakin_fold2\900s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset deakin_fold2 --seconds 900 || exit /b 1
)
if not exist ..\history_lr_deakin_fold2_900s\fit.json (
  python history_lr.py --project-root .. --dataset deakin_fold2_900s --out ..\history_lr_deakin_fold2_900s || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin_fold2 --setting window900 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_window900 || exit /b 1
if not exist ..\stage7\window900_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2_900s --runs ..\scope_deakin_fold2_window900 --history-lr ..\history_lr_deakin_fold2_900s --candidates tgae tgae_bilinear vrdag --label window900 --out ..\stage7\window900_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_deakin_chronological\900s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset deakin --seconds 900 || exit /b 1
)
if not exist ..\history_lr_deakin_900s\fit.json (
  python history_lr.py --project-root .. --dataset deakin_900s --out ..\history_lr_deakin_900s || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting window900 --tuning ..\tuning_deakin --out ..\scope_deakin_window900 || exit /b 1
if not exist ..\stage7\window900_deakin.json (
  python compare_best.py --project-root .. --dataset deakin_900s --runs ..\scope_deakin_window900 --history-lr ..\history_lr_deakin_900s --candidates tgae tgae_bilinear vrdag --label window900 --out ..\stage7\window900_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_unsw_fold2\900s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset unsw_fold2 --seconds 900 || exit /b 1
)
if not exist ..\history_lr_unsw_fold2_900s\fit.json (
  python history_lr.py --project-root .. --dataset unsw_fold2_900s --out ..\history_lr_unsw_fold2_900s || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting window900 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_window900 || exit /b 1
if not exist ..\stage7\window900_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2_900s --runs ..\scope_unsw_fold2_window900 --history-lr ..\history_lr_unsw_fold2_900s --candidates tgae tgae_bilinear vrdag --label window900 --out ..\stage7\window900_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready\900s\metadata.json (
  python make_merged_windows.py --project-root .. --dataset unsw --seconds 900 || exit /b 1
)
if not exist ..\history_lr_unsw_900s\fit.json (
  python history_lr.py --project-root .. --dataset unsw_900s --out ..\history_lr_unsw_900s || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting window900 --tuning ..\tuning_unsw --out ..\scope_unsw_window900 || exit /b 1
if not exist ..\stage7\window900_unsw.json (
  python compare_best.py --project-root .. --dataset unsw_900s --runs ..\scope_unsw_window900 --history-lr ..\history_lr_unsw_900s --candidates tgae tgae_bilinear vrdag --label window900 --out ..\stage7\window900_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting window900 --files ..\stage7\window900_unsw.json ..\stage7\window900_deakin.json ..\stage7\window900_deakin_fold2.json ..\stage7\window900_unsw_fold2.json || exit /b 1
echo Stage 7B finished.
