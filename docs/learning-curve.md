# What more training data is worth: the learning curve, 2026-11-27

**Main idea:** each doubling of the training rows has raised the ranking
learner's AUC by about +0.03, and the curve has not started to level off.
More data of the same kind is worth having, but the steps are large: one
more doubling means about 12,000 more labelled compositions that the table
does not hold.

This is a measurement, not a planned test. It decides no model and changes
no ranking. Command: `python scripts/learning_curve.py`, written to
`docs/learning-curve.json`.

## How it was measured

- Cycle 6's boosting classifier, with its own settings, folds and
  early-stopping rows. Each development fold's training rows were thinned
  at random to a fraction, keeping the share of superconductors.
- The held-out folds' predictions were pooled into one non-cuprate AUC, as
  in every comparison in this project.
- Each fraction below 1 was repeated 3 times with different thinning.
- **Check:** at the full fraction the result is cycle 6's own development
  AUC to the last digit (0.78817), so the set-up is the real one.

| Share of training rows | Labelled training rows per fold | Non-cuprate AUC (mean of 3) | Spread (SD) |
| --- | --- | --- | --- |
| 1/8 | 1,447 | 0.696 | 0.007 |
| 1/4 | 2,895 | 0.732 | 0.002 |
| 1/2 | 5,789 | 0.759 | 0.004 |
| 3/4 | 8,685 | 0.775 | 0.002 |
| All | 11,580 | 0.788 | — |

| Doubling | AUC gain |
| --- | --- |
| 1/8 → 1/4 | +0.036 |
| 1/4 → 1/2 | +0.027 |
| 1/2 → all | +0.029 |
| Fitted slope, all points | +0.030 per doubling |

## What it means

- [Certain, on these rows] The user's instinct holds: more rows of this kind
  have helped steadily, and the last doubling helped as much as the one
  before it.
- [Likely] It explains why the project's earlier additions looked so small:
  - cycle 4 added 851 rows, about 7% more;
  - that is a tenth of a doubling, worth about +0.003 by this curve;
  - cycle 4 measured −0.004 (−0.010 to +0.002) at home, which is
    consistent with that within noise.
  - Small additions cannot show up; only large ones can.
- [Speculative] If the trend held for one more doubling, about 23,000
  labelled compositions would give a development AUC near 0.82. Curves like
  this usually bend down eventually, so treat that as an upper guess.
- [Likely] The new rows count only if they are new compositions with
  trustworthy labels:
  - a composition the table already holds adds nothing;
  - the thinning here kept the table's mix of measured nulls and database
    zeros, so the curve says nothing about which of the two is worth more.

## What the sources could add, by this curve [Speculative]

| Source | New labelled compositions | AUC gain if the curve holds |
| --- | --- | --- |
| NEMAD superconductor database (Itani et al. 2026), the authors' own count | about 14,000 "not in SuperCon" | about +0.03 |
| The same, by this project's stand-in check on SuperCon2 (23% new element systems, 30% more near our parents; perhaps half pass a check) | about 3,000 to 5,500 | about +0.01 to +0.02 |
| ElementsClaw preprint's verified superconductors missing from the structure database | 66 | none measurable |
| One more year of literature harvesting by the project's own readers | hundreds | none measurable |

- The large source is the only one that could move the AUC by itself. The
  small ones are still worth collecting as blind test data.
