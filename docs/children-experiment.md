# The five-children experiment: two generations, judged on later discoveries

Owner's idea (2026-11-16): give several models the same chemistry and physics
knowledge and the literature up to a cutoff, have them predict superconductors
reported after it, let the best one produce the next generation, and repeat.

Written 2026-11-16 07:46 UTC, before any child was trained or scored. The
counts in "Feasibility" were made while writing, from the data alone: no model
had seen the pool or the hits.

## Feasibility: what dated data exists

- **The training table has no publication years.** Its two SuperCon snapshots
  (the 2018 export and the 2024 release behind SuperCon-MTG) hold almost the
  same compositions: only 46 are in the later one alone. A snapshot difference
  cannot serve as "the future".
- **SuperCon2 has years.** It is 40,324 records extracted automatically from
  arXiv papers, each with a DOI and a publication year (extracted 2022-12-03;
  `docs/data-sources.md`). It is not committed; `scripts/children.py data`
  downloads it to `data/raw/supercon2/`.
- **It is noisy.** Extraction puts Curie temperatures and other transitions in
  the Tc column (it lists KCuF3 at 38 K and TiAu at 90 K), misses pressure, and
  includes calculated values.
- **Counted against this project's candidate pool:** of the 25,136 Materials
  Project compositions that are not within 0.02 of any composition in the
  2018-era table, SuperCon2 has a Tc record for 217. By first year: 148 before
  2016, 34 in 2016 to 2018, and 35 in 2019 to 2021.

**So the test is small.** About 70 raw later discoveries, perhaps 40 real. It
can show a large difference between models and nothing finer. A
well-powered version needs the original NIMS SuperCon datasheet with its
reference years (`mdr.nims.go.jp`, which this environment cannot reach).

## Design

**Knowledge given to every child (the "cutoff").**
- Labels from the 2018 SuperCon export and 3DSC (built from that export)
  only. SuperCon-MTG (with its Hosono and SuperMat rows) and this project's
  own corrections are left out, because they carry later knowledge. Locked
  rows are left out, as always.
- Chemistry: the element-property features the ranking of record uses.
- The effective cutoff is about 2017 (the export is "by 2018"), not 2016.

**Pool.** The 25,136 Materials Project compositions above. A child ranks all
of them by its chance of superconducting.

**Hits.** A pool composition is a hit when SuperCon2 has a record for it
(within 0.0015 on every element) with a Tc above 0, at least one record with
no applied pressure, and a hydrogen fraction below 0.5.
- **Selection slice:** hits whose first SuperCon2 year is before 2016. These
  are not "future": they are superconductors the 2018 export happens to lack.
  The children never see them, so they can be used to choose between children.
- **Final slice:** hits first reported in 2016 to 2021. Scored **once**, at the
  end, for two models only: the champion and the baseline.
  - Before any scoring, one reader checks each final-slice hit against its
    paper's title and abstract: does it report superconductivity in that
    compound at ambient pressure? The primary result uses the confirmed hits;
    the result on all raw hits is reported beside it.

**Score.** The AUC of hits against the rest of the pool, which is the mean
percentile at which the hits are ranked. Also reported: hits in the top 1%
and top 5% of the pool. Pool compounds that are not hits count as negatives
though some are undiscovered superconductors, so every score is a lower bound.

**Generation 1: five children that differ in kind.**

| Child | What it is |
| --- | --- |
| `boost` | The ranking of record's method: gradient boosting, round 2's classifier settings. **The baseline.** |
| `bag` | Eight boosting models, each on a resample of the groups, averaged |
| `knn` | Nearest neighbours (15, distance-weighted) on standardised features: pure memory of similar known compounds |
| `mlp` | A small neural network (128 and 64 units) on standardised features |
| `physics` | `boost` plus one extra input: a score from a model trained only on Bercx et al.'s electron-phonon calculations, cross-fitted so no compound's own calculation is used |

**Generation 2: offspring of the generation 1 winner** (W; runner-up R).

| Child | What it is |
| --- | --- |
| `elite` | W unchanged |
| `variant_a`, `variant_b` | W with one setting changed each way (table in `scripts/children.py`, fixed before the run) |
| `cross2` | Average of the pool ranks of W and R |
| `cross3` | Average of the pool ranks of generation 1's top three |

**Promotion rule.** The generation 2 winner replaces W as champion only if its
selection-slice AUC is higher and the paired bootstrap interval of the
difference (1,000 resamples of the hits) lies above zero. Otherwise the
champion is W.

**Final test, once.** Champion against `boost` on the final slice: AUC, top 1%
and 5% hits, paired interval.

**How it explains itself.** For the champion: permutation importance of each
input group on the selection slice, and the nearest training compounds of its
ten best-ranked pool compounds. These are reported as descriptions. They are
not used to choose anything.

**Cost.** Plain code on this machine's CPU. No AI credits, apart from one
low-tier reader for the final-slice check.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| `boost` ranks the selection hits at AUC 0.70 or more | 60% |
| The generation 1 winner is `boost` or `bag` | 55% |
| `knn` is within 0.03 of `boost` on the selection slice | 40% |
| `physics` beats `boost` on the selection slice | 30% |
| Generation 2 promotes a child over its parent | 20% |
| The champion beats `boost` on the final slice with an interval above zero | 10% |
| `boost` ranks the confirmed final-slice hits at AUC 0.65 or more | 55% |

## What this cannot show

- **Fine differences.** With a few dozen hits the standard error of an AUC is
  about 0.04 to 0.06. Picking the best of five by a noisy score also inflates
  the winner's selection score; the final slice is the only honest number.
- **Foresight in the strict sense.** The cutoff is a database export, not a
  date. Materials Project lists a compound because someone made it or
  calculated it, some after 2018, so the pool itself carries a little later
  knowledge. Every child sees the same pool, so comparisons between children
  are fair.
- **Why a model is right.** Importance scores describe a model; they are not
  a mechanism.

**Correction (written 2026-11-16 07:48 UTC).** The line "Written 2026-11-16
07:46 UTC" above is later than the commit that added this file (07:45:56 UTC).
It was written at 07:45 UTC. Nothing else changes.

## Data assembled, 2026-11-16 07:48 UTC (no child trained yet)

`python scripts/children.py data` reproduces the feasibility counts:
25,136 pool compositions, 217 with a SuperCon2 record
(`docs/children/data.json`, `data/benchmarks/children/hits.csv`).

| Slice | Hits | Left out: hydrogen-rich | Left out: pressure-only |
| --- | --- | --- | --- |
| Selection (first year before 2016) | 137 | 6 | 5 |
| Final (2016 to 2021) | 54 | 7 | 8 |

The training table has 13,562 compositions (10,210 superconductors), labelled
from the 2018 export and 3DSC only, locked rows left out.
