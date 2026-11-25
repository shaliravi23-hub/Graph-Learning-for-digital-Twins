# Stage 10: correction to the plan's background section

**Written on:** 21 November 2026 (Asia/Kuala_Lumpur), after the random-search results.

The plan `preregistration_stage10_random_search.md` (SHA-256 7205eaebd0c4ba20…) is unchanged. This note corrects one statement in its section "What stage 9 found".

## The statement
"In 13 of the 14 searches, the last 20 trials improved the best validation F1 by 0.006 or less."

## The correct figure
Recounted from the stage-9 study files (`stage9_bo/{unsw,deakin}/studies/*.db`), the right number is 10 of the 14 searches. For each search, the gain is the best validation F1 over trials 0–39 minus the best over trials 0–19, counting completed trials only. The four exceptions are:

| Dataset | Model | Gain over the last 20 trials |
|---|---|---|
| UNSW | VGRNN | 0.016 |
| Deakin | VGRNN | 0.013 |
| Deakin | Static VGAE (previous) | 0.010 |
| UNSW | Static VGAE (previous) | 0.007 |

All four gains are within the seed-to-seed SD of the final runs, which ranged from 0.001 to 0.019.

## Effect
- The design, the predictions R1–R4 and the analyses are unchanged.
- The statement was part of the stated basis of R2, a margin of 0.02. The other part of that basis, the seed SD of up to 0.019, is correct.
- The error was found while preparing the manuscript, after the random-search results were known. Both are reported.
