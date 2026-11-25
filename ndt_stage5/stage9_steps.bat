@echo off
REM Stage 9 steps (preregistration_stage9_optimisation.md). Run through run_stage9.bat. Finished steps are skipped.
set PLAN=preregistration_stage9_optimisation.md
set ARGS=--project-root .. --plan %PLAN% --out ..\stage9_bo
python -c "import optuna" 2>nul || pip install optuna==5.1.0 || exit /b 1
if not exist ..\stage9_bo\self_test_done.txt (
  python bo_search.py --project-root .. --dataset unsw --step self-test --models tgae --out ..\stage9_bo || exit /b 1
  echo passed > ..\stage9_bo\self_test_done.txt
)
python bo_search.py %ARGS% --dataset unsw --step both || exit /b 1
python bo_search.py %ARGS% --dataset deakin --step both || exit /b 1
if not exist ..\stage9_bo\unsw\newday_scores\scoring_report.json (
  python score_newdays.py --project-root .. --runs ..\stage9_bo\unsw\final --out ..\stage9_bo\unsw\newday_scores || exit /b 1
)
if not exist ..\stage9_bo\capacity_check\summary.csv (
  python capacity_check.py --project-root .. --out ..\stage9_bo\capacity_check --runs unsw=stage9_bo/unsw/final deakin=stage9_bo/deakin/final || exit /b 1
)
if not exist ..\stage9\stage9_results.json (
  python bo_analysis.py --project-root .. --plan %PLAN% --bo ..\stage9_bo --newday-scores ..\stage9_bo\unsw\newday_scores --stage7 ..\stage7 --out ..\stage9 || exit /b 1
)
echo Stage 9 finished. Results in ..\stage9
