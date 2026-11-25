# Deviations from preregistration_unsw_fold2.md

The plan file is unchanged (SHA-256 c36e1e8d9cd2babe8217ca5ea6d7c819500e24c5c47dd61f72b308d59814b28f). Each deviation is listed with its time, its reason and what was known when it was made.

## D1. Source of the split (13 November 2026, before any data were read or any model was trained)
- **What happened:** the first command, `make_unsw_fold2.py`, stopped at its file check because `outputs\graphs_300s\16-09-23\graph_snapshots.pkl` does not exist. That folder holds only `edges.csv`, `nodes.csv` and `graph_stats.csv`. No data had been read, no split had been built, no model had been trained and no result had been produced.
- **Change:** `make_unsw_fold2.py` now builds the split from the main UNSW dataset, `dataset_ready\300s\static_train.pkl`, which already holds all five days (23–27 September) as built for the paper. It uses the same procedure as `make_deakin_fold2.py`:
  1. undo the main feature scaling with the stored mean and SD;
  2. keep only nodes active on at least one of the five days;
  3. refit the scaling on 23–25 September only;
  4. build history sequences with the `build_graph_dataset.py` functions.

  The adjacency matrices are exactly those the main analysis used for these days. Dates, history length, the 288-window completeness check and everything else in the plan are unchanged.
- **Check:** on synthetic data the new builder reproduces the graph-based build exactly. Adjacency, node order and window keys are identical, and node features differ by at most 1e-6 (float32 rounding).
- **Script hash:** `make_unsw_fold2.py` changes from 2add5d70…04fa to `6ca9ccc9605694d447ce2804334a626f4561dae54253b57825479f9d20382e66`. No other script changed.
