# Do crystal-structure features improve the ranking? Plan, fixed before any training

**Main idea:** add three structure features to cycle 6's recipe and compare
the two models on the development rows that have a structure, held out by
group. The features are space-group number, crystal system and volume per
atom.

Written 2026-11-27, before any model with these features has been trained or
scored. Owner approved (c) the same day. Features:
`scripts/structure_features.py`. Test: `scripts/structure_cycle.py`.

## Rows and features

- **Where the structures come from:**
  - 3DSC, which matches SuperCon to Materials Project structures: 5,610 of
    the table's 15,960 compositions;
  - this project's verified matching (`docs/structure-match.md`), high and
    medium confidence only: 887 more.
  - All other rows are missing.
- **Comparison rows:** non-cuprate development rows with a label and a
  structure. That is 4,968 rows (3,636 superconductors, 1,332 nulls) in
  1,677 groups.
- **Why only rows with a structure.** Every Materials Project candidate the
  ranking scores has a structure. Whether a training row has one may
  correlate with its label: 3DSC is a superconductor database's matches. A
  model could learn "has a structure, so superconductor". Comparing only
  rows with a structure measures what the structure says, not whether it
  exists.

## Models

| | Features | Otherwise |
| --- | --- | --- |
| **A** | Cycle 6's 155 formula features | Cycle 6's recipe: boosting settings, seed offset 0, the 851 NIMS added rows, the same folds and early-stopping groups |
| **B** | The same 155, plus space-group number, crystal system (7 indicator columns) and volume per atom; missing where there is no structure | Identical to A |

- Both are trained in the same run, so the only difference is the
  structure features.
- The added NIMS rows have no structure features.

## Reading, fixed now

- **E1, the decision.** B minus A, non-cuprate AUC (chance of
  superconducting) on the 4,968 rows. Groups are resampled 1,000 times.
- **E2, at home.** B minus A on all 8,598 labelled non-cuprate development
  rows.
- **E3, reported only.** B minus A in AUC for "Tc ≥ 10 K", by expected Tc, on
  the 4,968 rows.

| Result | What follows |
| --- | --- |
| E1 median at least +0.005 and E1's interval above zero, and E2's median not below −0.005 | Structure features enter the ranking as cycle 8. The bundle is rebuilt for TabICLv2. Candidates are scored with their own Materials Project structures |
| Anything else | No change. The structure features are kept for the physics screen and for Hazel inputs |

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| E1's median is above zero | 70% |
| The adoption rule is met | 35% |
| E3's median is above zero | 60% |

**Why not higher.**
- [Likely] Three coarse features carry little the formula does not already
  imply.
- [Likely] Some 3DSC matches are the wrong polymorph. Their matching was not
  verified here.
- Much of the gain the literature reports for structure models comes from
  random splits.
