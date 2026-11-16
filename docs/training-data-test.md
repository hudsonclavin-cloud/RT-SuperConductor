# Which training data: the project's table, the NIMS file, or both?

**Main idea:** in the blind test, a model trained on the NIMS file beat the
ranking of record by 0.126 on the same compounds, with the same learner and
inputs. That was unplanned and rested on 18 nulls. This is the planned test of
what it suggests: that the ranking of record would do better with the NIMS
data in its training.

Plan written 2026-11-16 18:35 UTC, before any model in this test was trained. The owner
approved it.

## The two training sets

| | Ranking of record's table | NIMS file |
| --- | --- | --- |
| Compositions | 14,010 usable (development and train-only rows) | 15,496 |
| What a negative is | Mixed: database flags and reviewed negative evidence | A sample measured and found not superconducting |
| Dates | None | Publication year on every record |

- The record table's 1,950 locked test rows are not touched: not trained on,
  not scored.

## The test

- **Three models, one learner** (the ranking of record's boosting settings,
  155 formula features): trained on the record table only, on NIMS only, and
  on both together.
- **Five folds by element system.** For each fold, every composition whose
  element system falls in it is removed from all training data, whichever
  table it came from. So no model has seen the chemistry it is scored on.
- **Scored on each table's own labels,** intermetallic ("Other") rows only,
  the kind of compound rung 4 picks from: once on the held-out record-table
  rows, once on the held-out NIMS rows.
- Intervals from 1,000 resamples of element systems.
- Where a composition is in both tables with different labels, both rows are
  kept and the disagreement rate is reported.

## What would count

| Outcome | Reading |
| --- | --- |
| "Both" beats "record only" on the NIMS rows with a 99% interval above zero, **and** is no more than 0.01 below it on the record table's own rows | Adding the NIMS data helps where the record table is weak and does not hurt at home. It becomes a candidate change to the ranking of record. That change alters training labels, so it goes to the owner as its own pull request under the learning-cycle guards. |
| "Both" helps on the NIMS rows but hurts on the record table's rows | The two tables mean different things by "negative". No change; the difference itself is the finding. |
| "Both" does not help on the NIMS rows | The blind-test difference was chance. Drop it. |

- [Certain] The first half of the first row is nearly automatic: a model
  given NIMS data will do better on NIMS rows. The informative parts are the
  size of the gain and whether home performance holds.
- **Also reported, for information:** each single-table model on the other
  table's rows. That shows which data transfers better.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| "Both" beats "record only" on NIMS rows, 99% interval above zero | 90% |
| "Both" is within 0.01 of "record only" on the record table's rows | 60% |
| "Record only" scores below 0.65 on the NIMS rows | 55% |
| "NIMS only" scores below 0.65 on the record table's rows | 50% |
| The two tables disagree on more than 5% of the compositions they share | 50% |

## Limits

- Neither table is a neutral judge. Each is scored on its own labels, and
  there is no third set of experimental labels left that neither has seen.
- A change to the ranking of record is not made here. This test only says
  whether to propose one.
