@echo off
REM Stage 10 steps (preregistration_stage10_random_search.md). Run through run_stage10.bat. Finished steps are skipped.
set PLAN=preregistration_stage10_random_search.md
set ARGS=--project-root .. --plan %PLAN% --out ..\stage10_rs --sampler random
python -c "import optuna" 2>nul || pip install optuna==5.1.0 || exit /b 1
python bo_search.py %ARGS% --dataset unsw --step both || exit /b 1
python bo_search.py %ARGS% --dataset deakin --step both || exit /b 1
if not exist ..\stage10_rs\unsw\newday_scores\scoring_report.json (
  python score_newdays.py --project-root .. --runs ..\stage10_rs\unsw\final --out ..\stage10_rs\unsw\newday_scores || exit /b 1
)
if not exist ..\stage10_rs\capacity_check\summary.csv (
  python capacity_check.py --project-root .. --out ..\stage10_rs\capacity_check --runs unsw=stage10_rs/unsw/final deakin=stage10_rs/deakin/final || exit /b 1
)
if not exist ..\stage10\stage9_results.json (
  python bo_analysis.py --project-root .. --plan %PLAN% --bo ..\stage10_rs --newday-scores ..\stage10_rs\unsw\newday_scores --stage7 ..\stage7 --out ..\stage10 || exit /b 1
)
if not exist ..\stage10\comparison\comparison.json (
  python rs_compare.py --plan %PLAN% --tpe ..\stage9 --rs ..\stage10 --out ..\stage10\comparison || exit /b 1
)
echo Stage 10 finished. Results in ..\stage10
