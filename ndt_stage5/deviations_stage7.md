# Stage 7: deviations from `preregistration_stage7.md` (plan SHA-256 bf038249…f048f)

**D7C-1 (15 November 2026): five scripts changed to add the prediction horizon.**
The changed scripts are `train_seeds.py`, `run_analysis_seeds.py`, `history_lr.py`, `compare_best.py` and `run_scope.py`. The details, the old and new hashes, and the tests are in `preregistration_stage7_addendum_7C.md`. With one-step data the changed scripts give identical results.

**D7-2 (recorded 16 November 2026): run order.**
- The plan's order was 7A, 7B, 7C, 7E, 7D.
- The actual order was 7A, then 7C (15 Nov, about 15:40–21:46), then 7B (15 Nov, about 22:27, to 16 Nov, 00:39).
- 7B therefore ran with the D7C-1 versions of the five scripts. Every 7B result file records those hashes.
- Each setting is evaluated on its own, and one-step behaviour is unchanged, so the order affects no result.

**D7-3 (recorded 16 November 2026): an exploratory analysis ran between 7C and 7B.**
- What ran: `periodicity_check.py` (SHA-256 2f020aaf…112c), under `periodicity_check_plan.md` (SHA-256 0d299956…3c95).
- It was motivated by the 7C results and was specified before it ran.
- It is not part of the pre-registered predictions, and it trains nothing.
- It changed no stage-7 script, candidate set or rule.
- Its outcomes were E1 held, E2 held and E3 not held. They are reported in full.
