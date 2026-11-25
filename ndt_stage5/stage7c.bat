@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base). Safe to rerun after an interruption: finished steps are skipped.
REM Stage 7C: predicting 2 and 4 windows ahead (needs stage 7A). About 120 GPU runs.
if not exist ..\dataset_ready_deakin_fold2\300s_h2\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset deakin_fold2 --horizon 2 || exit /b 1
)
if not exist ..\history_lr_deakin_fold2_h2\fit.json (
  python history_lr.py --project-root .. --dataset deakin_fold2_h2 --out ..\history_lr_deakin_fold2_h2 || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin_fold2 --setting horizon2 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_horizon2 || exit /b 1
if not exist ..\stage7\horizon2_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2_h2 --runs ..\scope_deakin_fold2_horizon2 --history-lr ..\history_lr_deakin_fold2_h2 --candidates tgae tgae_bilinear vrdag --label horizon2 --out ..\stage7\horizon2_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_deakin_chronological\300s_h2\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset deakin --horizon 2 || exit /b 1
)
if not exist ..\history_lr_deakin_h2\fit.json (
  python history_lr.py --project-root .. --dataset deakin_h2 --out ..\history_lr_deakin_h2 || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting horizon2 --tuning ..\tuning_deakin --out ..\scope_deakin_horizon2 || exit /b 1
if not exist ..\stage7\horizon2_deakin.json (
  python compare_best.py --project-root .. --dataset deakin_h2 --runs ..\scope_deakin_horizon2 --history-lr ..\history_lr_deakin_h2 --candidates tgae tgae_bilinear vrdag --label horizon2 --out ..\stage7\horizon2_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_unsw_fold2\300s_h2\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset unsw_fold2 --horizon 2 || exit /b 1
)
if not exist ..\history_lr_unsw_fold2_h2\fit.json (
  python history_lr.py --project-root .. --dataset unsw_fold2_h2 --out ..\history_lr_unsw_fold2_h2 || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting horizon2 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_horizon2 || exit /b 1
if not exist ..\stage7\horizon2_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2_h2 --runs ..\scope_unsw_fold2_horizon2 --history-lr ..\history_lr_unsw_fold2_h2 --candidates tgae tgae_bilinear vrdag --label horizon2 --out ..\stage7\horizon2_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready\300s_h2\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset unsw --horizon 2 || exit /b 1
)
if not exist ..\history_lr_unsw_h2\fit.json (
  python history_lr.py --project-root .. --dataset unsw_h2 --out ..\history_lr_unsw_h2 || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting horizon2 --tuning ..\tuning_unsw --out ..\scope_unsw_horizon2 || exit /b 1
if not exist ..\stage7\horizon2_unsw.json (
  python compare_best.py --project-root .. --dataset unsw_h2 --runs ..\scope_unsw_horizon2 --history-lr ..\history_lr_unsw_h2 --candidates tgae tgae_bilinear vrdag --label horizon2 --out ..\stage7\horizon2_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting horizon2 --files ..\stage7\horizon2_unsw.json ..\stage7\horizon2_deakin.json ..\stage7\horizon2_deakin_fold2.json ..\stage7\horizon2_unsw_fold2.json || exit /b 1
if not exist ..\dataset_ready_deakin_fold2\300s_h4\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset deakin_fold2 --horizon 4 || exit /b 1
)
if not exist ..\history_lr_deakin_fold2_h4\fit.json (
  python history_lr.py --project-root .. --dataset deakin_fold2_h4 --out ..\history_lr_deakin_fold2_h4 || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin_fold2 --setting horizon4 --tuning ..\tuning_deakin_fold2 --out ..\scope_deakin_fold2_horizon4 || exit /b 1
if not exist ..\stage7\horizon4_deakin_fold2.json (
  python compare_best.py --project-root .. --dataset deakin_fold2_h4 --runs ..\scope_deakin_fold2_horizon4 --history-lr ..\history_lr_deakin_fold2_h4 --candidates tgae tgae_bilinear vrdag --label horizon4 --out ..\stage7\horizon4_deakin_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_deakin_chronological\300s_h4\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset deakin --horizon 4 || exit /b 1
)
if not exist ..\history_lr_deakin_h4\fit.json (
  python history_lr.py --project-root .. --dataset deakin_h4 --out ..\history_lr_deakin_h4 || exit /b 1
)
python run_scope.py --project-root .. --dataset deakin --setting horizon4 --tuning ..\tuning_deakin --out ..\scope_deakin_horizon4 || exit /b 1
if not exist ..\stage7\horizon4_deakin.json (
  python compare_best.py --project-root .. --dataset deakin_h4 --runs ..\scope_deakin_horizon4 --history-lr ..\history_lr_deakin_h4 --candidates tgae tgae_bilinear vrdag --label horizon4 --out ..\stage7\horizon4_deakin.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready_unsw_fold2\300s_h4\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset unsw_fold2 --horizon 4 || exit /b 1
)
if not exist ..\history_lr_unsw_fold2_h4\fit.json (
  python history_lr.py --project-root .. --dataset unsw_fold2_h4 --out ..\history_lr_unsw_fold2_h4 || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw_fold2 --setting horizon4 --tuning ..\tuning_unsw_fold2 --out ..\scope_unsw_fold2_horizon4 || exit /b 1
if not exist ..\stage7\horizon4_unsw_fold2.json (
  python compare_best.py --project-root .. --dataset unsw_fold2_h4 --runs ..\scope_unsw_fold2_horizon4 --history-lr ..\history_lr_unsw_fold2_h4 --candidates tgae tgae_bilinear vrdag --label horizon4 --out ..\stage7\horizon4_unsw_fold2.json --plan preregistration_stage7.md || exit /b 1
)
if not exist ..\dataset_ready\300s_h4\metadata.json (
  python make_horizon_dataset.py --project-root .. --dataset unsw --horizon 4 || exit /b 1
)
if not exist ..\history_lr_unsw_h4\fit.json (
  python history_lr.py --project-root .. --dataset unsw_h4 --out ..\history_lr_unsw_h4 || exit /b 1
)
python run_scope.py --project-root .. --dataset unsw --setting horizon4 --tuning ..\tuning_unsw --out ..\scope_unsw_horizon4 || exit /b 1
if not exist ..\stage7\horizon4_unsw.json (
  python compare_best.py --project-root .. --dataset unsw_h4 --runs ..\scope_unsw_horizon4 --history-lr ..\history_lr_unsw_h4 --candidates tgae tgae_bilinear vrdag --label horizon4 --out ..\stage7\horizon4_unsw.json --plan preregistration_stage7.md || exit /b 1
)
python summarize_stage7.py --setting horizon4 --files ..\stage7\horizon4_unsw.json ..\stage7\horizon4_deakin.json ..\stage7\horizon4_deakin_fold2.json ..\stage7\horizon4_unsw_fold2.json || exit /b 1
echo Stage 7C finished.
