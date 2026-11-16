# A steadier ranking of the made metals

## Why

Cycles 1 and 3 (`docs/learning-cycles.md`) showed that any change of two to
four training labels, real or random, reshuffles the made-metals order. Across
eight perturbed runs the top 50 kept only 36 to 40 of its 50 places, and
Zr6Ga2Ni ranged from 3rd to 21st. A rank quoted to the place carries no
information beyond its range.

The idea: train many models on resamples of the training data, average their
probabilities for the ranking, and report each candidate's rank as a range
across the models.

## The method, fixed before any result

Written 2026-11-16 05:50 UTC (before any bagged model was trained).

- **Models.** Round 2's boosting settings, unchanged
  (`docs/scratchnet-round2.json`). Nothing is retuned.
- **Bags.** For each of the five development folds, 8 bags. Each bag
  resamples the *groups* (`ds.arrays.group`, the near-duplicate groups the
  split already uses) of the fold's fit rows with replacement, keeps all rows
  of each drawn group (with repeats), and fits the classifier and the Tc
  regressor. The fold's validation rows stay as they are and stop each model
  early. Bag `b` of fold `f` uses seed `1000 * b + f`. That is 40 final models.
- **Ranking of record for the bag.** The mean probability over the 40 models
  and the geometric-style mean Tc the single-model ranking already uses, passed
  through the unchanged `made_metals_order`.
- **Rank range.** Each of the 40 models also ranks the made metals alone. A
  candidate's range is the 10th to 90th percentile of its 40 ranks.
- **Cross-validation.** The same folds and settings as `run_learning_cycle.py`,
  predicting each test fold by the mean of its 8 bags. This gives the bag's AUC
  and Tc error on the development rows. The locked rows are neither trained on
  nor scored.

## The test

**Table.** The current one (after cycle 3: 11,825 superconductors, 3,702
negatives, sha `961fa702...`).

**Single-model reference.** Cycle 3's saved ranking
(`data/processed/learning-cycles/cycle3-made-metals.json.gz`).

**Perturbations.** Five placebos, seeds 41 to 45. Each marks four random
development non-cuprate negatives as superconductors at 0.9 K, in memory only
(`run_learning_cycle.placebo`). The same five perturbations go to both methods:
- single model: `run_learning_cycle.py --name stab-single-<n> --previous cycle3
  --placebo-flips 4 --placebo-seed 4<n>` (seeds 41 to 45);
- bag: `scripts/run_bagged_ranking.py --name stab-bag-<n> --previous
  stab-bag-base --placebo-flips 4 --placebo-seed 4<n>`.

**Primary measure.** For each method, the number of the unperturbed top 50 that
is still in the top 50 after the perturbation (the single model against
cycle 3's order, the bag against its own unperturbed order). Mean over the five
placebos.

**Decision rule.**
- The bag is **supported** if its mean top-50 kept is at least **5 more** than
  the single model's mean over the same five placebos, **and** both guards hold:
  - the bag's cross-validated AUC (all development rows) is at least
    **0.785** (cycle 3's 0.7902 minus 0.005);
  - the bag's cross-validated Tc log-MAE is at most **0.396** (cycle 3's
    0.3863 plus 0.010).
- If the primary measure is met and a guard fails: **steadier but worse**. No
  adoption.
- If the primary measure is not met: **not supported**. No adoption.
- Adopting the bag as the ranking of record is a separate pull request and the
  owner's decision. This test only reports.

**Reported, not decisive.** Top-10 and top-200 kept; rank correlation; the
ranks of Zr6Ga2Ni, Ti3SnH and the other watched compounds across the five
placebos, single against bag; each top-50 candidate's rank range; how many of
the top 50 keep their whole 10-to-90 range inside the top 100.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| Primary measure met (bag at least 5 more kept than the single model) | 55% |
| Bag's mean top-50 kept is at least 42 | 50% |
| AUC guard holds (at least 0.785) | 85% |
| Tc guard holds (at most 0.396) | 85% |
| Bag's AUC is at least 0.002 above the single model's | 35% |
| Zr6Ga2Ni's rank spread across the five placebos is narrower for the bag | 65% |

The single model's top-50 kept has run from 36 to 40 in nine runs, so I expect
about 38 for it here.

## What this test cannot show

- Whether the steadier order is *more right*. Stability is not accuracy: a
  constant ranking is perfectly stable. The AUC and Tc guards catch a large
  loss, not a small one.
- Whether the gain comes from the resampling or from averaging more models.
  The boosting models are deterministic, so 40 identical-data models would
  all be the same; the resampling is what makes the members differ. A
  no-resampling average of different-settings models is not tested.
- Anything about the real candidates. This is a stability test on the
  training pipeline, with five random perturbations.
- Five placebos have a wide margin: the primary threshold of 5 is about three
  times the spread of the single model's kept counts, so a real difference of
  that size should show, but a smaller one will not.
