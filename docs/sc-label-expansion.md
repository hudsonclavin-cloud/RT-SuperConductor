# More labels: the wider Materials Project window

> **Update 2026-11-07.** Measured before negatives were graded by evidence (`docs/negative-labels.md`). The production fit now uses the wide window's labels only where the negative passes the evidence tiers: 3,429 training rows instead of 3,664.

**Main idea:** 1,420 extra labelled materials, including 707 compositions not in the stable snapshot, slightly improve the tree models when each composition's polymorphs share one unit of weight. Without that weighting they hurt the neural network. The production classifier now trains on all 3,664 labels with this weighting.

## Where the extra labels come from

`scripts/fetch_mp_snapshot.py --hull-max 0.1 --max-sites 40` fetches 75,714 Materials Project materials: up to 0.1 eV/atom above the hull and up to 40 sites, versus 0.025 and 20 for the scoring pool. `build_mp_candidates.py` matches them to SuperCon and 3DSC by atomic fraction. That gives 3,664 labelled materials (1,923 superconductors, 1,741 tested non-superconductors), against 2,244 in the stable snapshot. The labelled rows are committed as `data/candidates/mp-wide-labelled.csv.gz`.

Many extra rows are other polymorphs of compositions already labelled. A literature report usually does not say which polymorph was measured, so the label is shared, and each polymorph is a copy of one piece of evidence.

## Test

`python scripts/eval_label_expansion.py`. Outer folds are StratifiedGroupKFold over the element systems of core + extra labels (5 folds × 3 shuffles). Every condition is scored on the same core test materials only.

| Model | Core labels only | + extra labels | + extra, polymorph-weighted |
| --- | --- | --- | --- |
| Gradient boosting | 0.815 | 0.818 (+0.004, wins 2 of 3) | **0.819** (+0.004, wins 2 of 3) |
| Random forest | 0.802 | 0.808 (+0.006, wins 2 of 3) | 0.810 (+0.007, wins 3 of 3) |
| PyTorch network | 0.767 | 0.747 (−0.021, wins 0 of 3) | 0.767 (−0.001, wins 1 of 3) |

## What this means

- **For gradient boosting the gain is within noise.** +0.004 on 2 of 3 shuffles. It is no worse and possibly slightly better, and the extra compositions widen what the model has seen, so the production model uses them.
- **Duplicated evidence hurts the neural network.** Unweighted polymorph copies cost it 0.02 AUC, and weighting removes the damage. This is the reason the production fit weights each row by 1 / (polymorphs of its composition), as the external review recommended.
- The model's reported held-out AUC (`docs/mp-sc-classifier.json`) is measured with core labels only. The table above is the evidence for the expanded training.
