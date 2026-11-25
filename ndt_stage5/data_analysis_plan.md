# Exploratory data and representation analysis (data_analysis.py)

**Written:** 19 November 2026, after the stage-9 results were known. This analysis is exploratory and descriptive: it tests no prediction and adjusts nothing. It answers the supervisor's question of whether the remaining gap lies in the data relationships or in the representations.

## What is computed
- **D1, pair-history table and its ceiling.** Each ordered device pair in a target window has an 8-bit state: its own presence in the last four windows and that of the reverse pair. A table of P(edge | state), pooled over all pairs (256 entries), is fitted on the training days (a usable method) and, separately, on the evaluated days themselves (an in-sample ceiling for any method that treats all pairs alike given these 8 bits). Pairs are ranked by the table; within the boundary state of the edge budget, metrics are expected values under a random choice among tied pairs. Compared at the validation edge budget with the optimised and published asymmetric TGAE (five seeds) and with EWMA (L = 4), seasonal copy, order-2 Transition and History-LR. Sets: main UNSW test days, main Deakin test day, the 12 further UNSW days.
- **D2, traces.** Target edges and appearing edges are split by the trace they leave in the pair's 8-bit history (previous window; earlier in the last four windows; reverse pair only; no trace), and each method's edges right are split the same way.
- **D3, inputs used.** The optimised and published asymmetric TGAE are re-scored with the node features set to zero, with the node feature rows permuted (fixed permutation, seed 0), and with the adjacency history set to zero.

## Order of work
D1, D2 and the two feature variants of D3 were defined before the first run. The zero-history variant of D3 was added after that run had shown that removing the node features collapses the model; it was added to see how much the model predicts from features alone.
