# Stage 7 addendum (15 November 2026): implementation of the prediction-horizon setting (7C)

This addendum belongs to `preregistration_stage7.md` (SHA-256 `bf0382497655eafe816c4db6f6418911296cacdad8f110876b6866924f0f048f`), which is not edited. The plan fixed the 7C procedure and prediction (P3a, P3b) and said that the implementing script would be added, with its hash, before any horizon run. This is that addition.

**Timing.** Written after the stage 7A results had been seen and before any horizon run on the real data. Nothing in P3a, P3b, the classification rule or the evaluation changes.

## 1. What was added
- `horizon_support.py`: builds the history/target items for horizon h, fits the h-step Transition table, and validates the horizon test files.
- `make_horizon_dataset.py`: writes `<dataset folder>\300s_h2` and `300s_h4`.
- `stage7c.bat`: runs 7C on the four test sets.

## 2. Deviation D7C-1: five scripts changed after the plan was fixed
These scripts changed only to add the horizon option. Each new branch runs only for the datasets `<test set>_h2` and `<test set>_h4`.

| Script | Hash in the plan | New hash | Change |
|---|---|---|---|
| `train_seeds.py` | `99877568…df1d` | `dca800c1…f797` | horizon datasets added to the registry; data setup calls `validate_h` and `validation_items_h` |
| `run_analysis_seeds.py` | `de0e3cd8…7e3e` | `776b3915…bc38` | horizon datasets added to the registry |
| `history_lr.py` | `22d2eff9…8f87` | `f8855dee…b3fa` | h-step transition feature; `validate_h`, `validation_items_h` |
| `compare_best.py` | `5949916e…1ce3` | `6800c25b…cadd` | h-step Transition; `validate_h`, `validation_items_h` |
| `run_scope.py` | `706acffb…6e75` | `19cf1d63…c0ae` | settings `horizon2` and `horizon4` |

**One-step behaviour is unchanged.** With h = 1 each new function calls the original function it replaces. On synthetic data:
- History-LR gave identical scores;
- `compare_best.py` gave an identical result file, apart from the recorded source hashes;
- `train_seeds.py` gave identical scores for TGAE and DynGEM, and identical protocol records apart from the runner hash.

Stages 7A, 7B, 7D and 7E use only one-step datasets, so either version of these scripts gives the same results for them.

## 3. How the fixed procedure is implemented

**History and target.**
- History windows are t−4 … t−1 and the target is window t+h−1.
- Items are built day by day, so no history or target crosses a day.
- This gives 285 − h targets per day: 283 for h = 2 and 281 for h = 4, against 284 for one step. Each day's first target is h − 1 windows later than in the one-step setting.
- The static windows and their feature scaling are copied unchanged from `300s`.

**Learned models (TGAE, asymmetric TGAE, VRDAG).**
- They keep their 7A settings and decoders.
- They are trained directly on h-step targets: history t−4 … t−1 is mapped to window t+h−1.
- The training loss, its feature-reconstruction term and VRDAG's training posterior refer to the target window t+h−1. Scoring uses the history only.
- The positive class weight is recomputed from the h-step training targets with the same formula.

*This is the one choice the plan did not spell out.* Direct training was chosen because the simple methods are also fitted directly to h-step transitions, and because it needs no change to the models. Iterated one-step prediction is not run.

**Simple methods.**
- Persistence uses window t−1.
- Recent frequency and EWMA (L = 4) use the same four history windows.
- EWMA's α is re-tuned on validation, as before.
- EWMA (expanding) advances by one history window per target, so it uses windows up to t−1 only and never sees windows t … t+h−2.

**Transition and History-LR.**
- The Transition table counts h-step pairs: the state at window w and the edge at window w+h, both in the same training day. It uses the same pooled pseudo-count as before.
- History-LR keeps the same 15 features. Its transition feature comes from the h-step counts. For a training row, leave-one-out removes the row's own (t−1 → t+h−1) pair.
- History-LR is refitted on the h-step training targets, and C is re-selected on validation.

**Validation and evaluation.**
- The threshold and the edge budget come from the h-step validation targets. The budget is the rounded mean edge count of those targets.
- New and disappearing edges are defined relative to window t−1.
- The descriptive new-edge categories use only windows observed before the history.
- `compare_best.py`, `summarize_stage7.py`, the 0.5-edge margin and the Holm adjustment over 8 contrasts are unchanged.

**Checks built into the scripts.**
- Before writing, `make_horizon_dataset.py` checks that the same builder with h = 1 reproduces the existing `temporal_test.pkl` exactly. This check runs on the real data.
- `validate_h` refuses a dataset whose metadata gives a different horizon, or whose targets are not h windows after the history.
- `validate_h` applies the stage-5 test checks (consecutive targets, overlapping histories, binary matrices with a zero diagonal) to a one-step view of the same items.

## 4. Tests on synthetic data (before release)
- **Two-step build.** Targets lie 2 windows after the last history window, and each target matrix is the static window it names.
- **Transition table.**
  - The 2-step table equals a brute-force count with the same pseudo-count, and differs from the 1-step table.
  - With h = 1 the table equals the stage-5 table.
- **History-LR.** Its transition feature uses the 2-step table. The leave-one-out version removes exactly the row's own pair.
- **Validator.** It refused metadata with the wrong horizon, and it refused a one-step test file offered as a two-step one.
- **End to end.** On the split-2 UNSW layout, the full pipeline ran with 1 training epoch: dataset, History-LR, three models × five seeds, and `compare_best.py`.
- **Interrupted build.** An interrupted build is rebuilt, and a finished one is never overwritten.

## 5. Script hashes (SHA-256)
```
e8b418194e4fe0e9e6905cf9aeec6bc561a4da45d9a960cc0ea66df0f7d838bf  horizon_support.py (new)
c8fda89334872329f9f7b931d88526a77e7688e89d6d4cf4e7b845f56f10a6d9  make_horizon_dataset.py (new)
3cfd6f5cb4b5ed1808ef5fc73103519e175b3c5659ea36b29f7df0fecd6fece8  stage7c.bat (new)
dca800c1be98b36765fd7f1f7c5ae05d67e91bfefa0b0745a8dfe876d455f797  train_seeds.py (changed, D7C-1)
776b3915785ae89f788ee2333311dc3f1d8a4a69cea4e0fd70c5b842b859bc38  run_analysis_seeds.py (changed, D7C-1)
f8855dee9bf08891ea223dea0d1a4e6c782740e8e4b0b1d612f6b572b98eb3fa  history_lr.py (changed, D7C-1)
6800c25bff3c63a3528d826371e3118af41dda39a176424a824614472f94cadd  compare_best.py (changed, D7C-1)
19cf1d634f01f5929087635bfe66e7b0c3103ece3329b22af7c2bc20fcf9c0ae  run_scope.py (changed, D7C-1)
f95e662f54f97043f4db011c1e08e1f76782a1932c881d74959ab4b120fa82d5  summarize_stage7.py (unchanged)
594c5e0e34d51d2472a9a749bb4e1d4082ed5cc9d4b574d1a98341b33aea2e4b  stage6_analysis.py (unchanged)
3ce69bcfb6edb832d77303b8e1ba8ad78dcc6a21753f36603e1b510e21c56e75  tune_models.py (unchanged)
2b0ccfe749358b0eb68c5bd1054fee68a6082207109eb69aded6499cec9d470f  make_merged_windows.py (unchanged)
c3888690d42485688b1fcf7fb9060e073bd7a60a155a027279cf02d3c748b2b7  evaluate_pilot.py (unchanged)
64e65639b68e72cfb9c7bebf8e2d9631b5bf86cc4689825e6f3c53eca0b0a7cd  evaluate_temporal.py (unchanged)
```
