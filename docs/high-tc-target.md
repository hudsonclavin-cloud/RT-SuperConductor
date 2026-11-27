# A "Tc of at least 10 K" target for the ranking: plan, fixed before any score

**Main idea:** the ranking is trained and judged on "superconducts at all".
Because our nulls mean "not superconducting above about 1–2 K", that task
asks it to separate a 3 K superconductor from a 1 K null, and composition
may not decide that. For the room-temperature goal the question is
different: which compounds have a high Tc?

This test asks two things, on rows held out by group, as in every
comparison in this project:
1. How well does the current ranking already answer "Tc of at least 10 K"?
2. Does a model trained directly on that question answer it better?

Written 2026-11-27, before any model has been trained or scored on this
target. Owner approved the test the same day (`docs/training-data-sources.md`).
Script: `scripts/high_tc_target.py`.

## Rows and target

- **Rows:** the development rows, out of fold, with the same five folds and
  early-stopping groups as cycle 6. Locked rows are not touched.
- **Main reading: non-cuprates.** 8,942 rows:
  - 1,901 superconductors with Tc of at least 10 K (929 other, 870
    iron-based, 102 non-copper oxides);
  - 4,741 superconductors below 10 K;
  - 1,956 nulls.
- **Target:** 1 if a superconductor has Tc of at least 10 K, else 0 (lower-Tc
  superconductors and nulls).
  - Every non-cuprate superconductor in these rows has a Tc, so none is set
    aside.
  - The same at 20 K (760 positives) is reported as a second reading.

## The scores compared

| Score | What it is |
| --- | --- |
| **Incumbent, p** | The ranking of record's out-of-fold chance of superconducting: the 50/50 blend of cycle 6 and TabICLv2, from the committed out-of-fold files |
| **Incumbent, expected Tc** | The blend's probability times its predicted Tc (log-averaged) |
| **New** | Cycle 6's boosting classifier, same features and settings, trained on the 10 K target. Out of fold |

The incumbent's better score of the two is fixed first, on this target, and
the new model is compared with it.

## Reading, fixed now

- **The comparison:** the new model minus the better incumbent, in
  non-cuprate AUC for "Tc of at least 10 K".
- **The interval:** groups are resampled 1,000 times.

| Result | What follows |
| --- | --- |
| Median gain of at least +0.01 and the interval above zero | The new model becomes a second ranking, the **high-Tc ranking**, used for the room-temperature track and for choosing Hazel candidates. The ranking of record is unchanged |
| Anything else | The incumbent's better score serves the high-Tc question. No new model |

- **Not a reason to change the ranking of record.** The incumbent's AUC on
  this target, whatever it is, is reported as an answer to question 1.
- **Not comparable with 0.79.** AUCs on this target measure a different
  task, so they are not compared with the current 0.79 as if one had
  improved on the other.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| The incumbent's better score reaches non-cuprate AUC of at least 0.85 on the 10 K target | 70% |
| Expected Tc beats p as the incumbent's better score | 75% |
| The new model meets the adoption rule | 30% |
| At 20 K, the incumbent's better score reaches at least 0.88 | 55% |
