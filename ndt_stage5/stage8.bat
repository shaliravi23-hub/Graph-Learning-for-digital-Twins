@echo off
REM Run from ndt_stage5 in Anaconda Prompt (base), after process_unsw_newdays.bat has processed all 12 new days.
REM Safe to rerun after an interruption: finished steps are skipped. No model is retrained.
REM 1. Replay on the main test days (the values the predictions are based on; must match the plan). About 5 min, CPU.
REM 2. Build the new-day dataset (records the plan's SHA-256). About 2 min.
REM 3. Score the 40 checkpoints on the new days on the GPU, with guard G1. About 5 min.
REM 4. Simple methods, guard G2, comparisons, predictions. About 15 min, CPU.
set GRAPHS=D:\unsw\outputs\graphs_300s
set PLAN=preregistration_stage8_newdays.md
for %%d in (16-09-28 16-09-29 16-09-30 16-10-01 16-10-02 16-10-03 16-10-04 16-10-05 16-10-06 16-10-07 16-10-11 16-10-12) do (
  if not exist %GRAPHS%\%%d\graph_snapshots.pkl (
    echo %%d has not been processed yet. Finish process_unsw_newdays.bat first.
    exit /b 1
  )
)
if not exist ..\stage8_main_replay\confirmation.json (
  python confirm_newdays.py --replay-main --project-root .. --plan %PLAN% --scores ..\seeds_unsw ..\seeds_unsw_decoder --main-runs ..\seeds_unsw --history-lr ..\history_lr_unsw --stage7 ..\stage7 --out ..\stage8_main_replay || exit /b 1
)
if not exist ..\dataset_ready_unsw_newdays\300s\metadata.json (
  python make_unsw_newdays.py --project-root .. --graph-root %GRAPHS% --plan %PLAN% || exit /b 1
)
if not exist ..\scores_unsw_newdays\scoring_report.json (
  python score_newdays.py --project-root .. --runs ..\seeds_unsw ..\seeds_unsw_decoder --out ..\scores_unsw_newdays || exit /b 1
)
if not exist ..\stage8\confirmation.json (
  python confirm_newdays.py --project-root .. --plan %PLAN% --scores ..\scores_unsw_newdays --main-runs ..\seeds_unsw --history-lr ..\history_lr_unsw --stage7 ..\stage7 --out ..\stage8 || exit /b 1
)
echo Stage 8 finished. Results in ..\stage8 (main-day replay in ..\stage8_main_replay).
