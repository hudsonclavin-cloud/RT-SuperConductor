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

## Result, 2026-11-16 18:43 UTC

**Main result:** adding the NIMS data helps a little on NIMS rows (+0.018)
and does not hurt on the record table's own rows (+0.002). By the rule above
that makes it a candidate change. It is far smaller than the blind test
suggested: the two tables turn out to be mostly the same compounds.

Numbers are from `docs/training-data-test.json` (command:
`python scripts/training_data_test.py`). Intermetallic rows, held-out
element systems, AUC:

| Trained on | Scored on record rows (6,637; 1,532 nulls) | Scored on NIMS rows (8,403; 1,731 nulls) |
| --- | --- | --- |
| Record table only | 0.8348 | 0.8255 |
| NIMS only | 0.8231 | 0.8470 |
| Both | 0.8367 | 0.8435 |

| Difference | Value | 95% interval | 99% interval |
| --- | --- | --- | --- |
| Both − record only, on NIMS rows | +0.0180 | +0.0108 to +0.0260 | +0.0085 to +0.0275 |
| Both − record only, on record rows | +0.0019 | −0.0044 to +0.0091 | −0.0064 to +0.0114 |
| Both − NIMS only, on NIMS rows | −0.0035 | −0.0087 to +0.0016 | −0.0108 to +0.0026 |
| Both − NIMS only, on record rows | +0.0136 | +0.0047 to +0.0255 | +0.0019 to +0.0292 |

**Against the rule.** Both conditions of the first row hold: the 99% interval
on NIMS rows is above zero, and "both" is not below "record only" at home.
Reading: candidate change to the ranking of record, to be proposed to the
owner. Nothing is changed here.

**What the test also showed.**

- [Certain] The tables overlap heavily: 12,461 compositions are in both
  (91% of the record table's 13,644 labelled rows). They disagree on the
  label of 55 of those (0.44%): 35 the record table calls superconductors and
  NIMS calls nulls, 20 the other way.
- [Certain] Each single-table model transfers well to the other table
  (0.8255 and 0.8231). They are not different kinds of data.
- [Likely] So the +0.126 seen in the blind test was mostly not a
  training-data effect. Here the same comparison (NIMS-trained against
  record-trained, on NIMS rows) is +0.0215. The rest is most likely chance
  from 18 nulls, as that test's own 99% interval allowed.
- [Likely] What "both" adds is the 3,035 compositions that only NIMS has.
  The shared compositions are counted twice in "both", which was the plan's
  stated design and is a limit: part of any difference could be that
  re-weighting.

**Predictions.**

| Outcome | Chance given | Result |
| --- | --- | --- |
| "Both" beats "record only" on NIMS rows, 99% interval above zero | 90% | Yes |
| "Both" is within 0.01 of "record only" on the record table's rows | 60% | Yes (+0.0019) |
| "Record only" scores below 0.65 on the NIMS rows | 55% | No (0.8255) |
| "NIMS only" scores below 0.65 on the record table's rows | 50% | No (0.8231) |
| The two tables disagree on more than 5% of shared compositions | 50% | No (0.44%) |

- Three of five wrong, all in the same direction: I assumed the two tables
  were much more different than they are. I had not counted the overlap
  before writing the plan.

**Correction to the plan's table (labelled, original text kept).** The plan
says "14,010 usable" record-table compositions. 14,010 is the number of
development and train-only rows; 366 of the development rows have no label,
so 13,644 were used. The locked test rows were not read.

**What a change would be, if the owner wants it.** Add the 3,035 NIMS-only
compositions to the ranking of record's training table, and look at the 55
disagreements by hand. It changes training labels, so it needs its own plan
and pull request under the learning-cycle guards. Its expected size is about
+0.02 AUC on intermetallics, and there is no unused locked test set left to
confirm it on.
