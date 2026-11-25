# Exploratory check: a two-window on/off pattern (15 November 2026)

**Status.** This check is exploratory. It is not part of the pre-registered stage-7 predictions. The 7C results suggested it and had been seen before it was written. It is specified here before it is run, and every output will be reported whatever it shows.

## What suggested it
Persistence (copying window t−1) gets more edges right when predicting 2 windows ahead than 1 window ahead, on all four test sets. The number of edges that change is also lower at lag 2 than at lag 1:

| Test set | Lag 1 | Lag 2 | Lag 4 |
|---|---|---|---|
| UNSW | 9.25 | 4.56 | 5.14 |
| UNSW split 2 | 12.75 | 6.73 | 7.38 |
| Deakin | 5.71 | 4.80 | 5.61 |
| Deakin split 2 | 4.51 | 3.60 | 4.36 |

## Hypothesis
Many connections are active in every other 5-minute window, that is, on a cycle of about 10 minutes. The learned models' one clear win (UNSW, one step ahead: +2.36 changes recovered per window against EWMA, L = 4) comes from predicting the edges that return after one window off.

## What is computed (`periodicity_check.py`)
**Part A: lag profile.**
- The mean number of edges that differ between windows t and t−k, for k = 1 … 8, within each day, on the 300 s train, validation and test windows.
- Among one-step new edges on the test windows, the share that were present at t−2.
- Among one-step disappearances, the share that were absent at t−2.

**Part B: one step ahead.** This uses the same cohorts, edge budget and tuned learned model (M\*, seeds 42–46) as the stage 7A comparison.
- *Seasonal naive* copies window t−2.
- *Order-2 Transition* is the per-pair probability of an edge at t given its states at t−2 and t−1, counted on training days with Transition's pooled pseudo-count.
- M\* − Order-2 Transition is tested with the stage-7 rule: two-level bootstrap with 10,000 replicates and blocks of 8, Holm over 8 contrasts, margin 0.5.
- As a guard, the script stops unless it reproduces the stage 7A numbers for Persistence, Transition and M\*.

## Expectations, fixed before running
- **E1.** The test windows of UNSW and UNSW split 2 show a zig-zag. Fewer edges change at lag 2 than at lags 1 and 3, and fewer at lag 4 than at lags 3 and 5.
- **E2.** On UNSW, M\* − Order-2 Transition in changes recovered is below +0.5 per window. Against EWMA (L = 4) it was +2.36.
- **E3.** With the stage-7 rule, M\* is clearly better than Order-2 Transition on 0 of the 4 test sets.

## How the outcomes will be read, fixed before running
- **E1 fails:** the pattern is not a two-window cycle, and the explanation is dropped.
- **E1 and E2 hold:** an order-2 counting rule reproduces the UNSW change recovery. This is reported as an explanation of the one learned-model win, and labelled exploratory.
- **E1 holds and E2 fails:** the cycle exists, but the learned models capture more than an order-2 rule. This is reported as such.
- **E3** is reported as found. The best simple method re-selected with the two new methods is descriptive only.

## Script hash (SHA-256)
```
2f020aaf329ff02f97b1ec1be28bd30d3215d4f9bd0420ea6fde39219f24112c  periodicity_check.py
```
