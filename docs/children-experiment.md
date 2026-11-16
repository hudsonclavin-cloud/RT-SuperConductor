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

## Amendment to the selection slice, written 2026-11-16 07:51 UTC

**What was seen first.** Generation 1 was running. Four children had been
scored on the raw selection slice, all at or below chance (AUC 0.44 to 0.50).
I had not looked at any child's score for any single compound.

**What is wrong with the slice.** A look at 30 of the 137 selection hits shows
most are not superconductors at all. They are substrates and reagents (MgO,
SiO2, LaAlO3, XeF2) and magnets whose Curie temperature was extracted as a Tc
(SrRuO3 at 165 K). [Likely] A compound with a pre-2016 arXiv record that
NIMS's curated export still lacks is, more often than not, an extraction
error. I should have foreseen that selection effect in the design.

**The change.** The selection slice gets the same reader check the plan
already gives the final slice: one low-tier reader per hit, from the paper's
title and abstract, blind to every model score. **Children are chosen on the
confirmed selection hits.** The raw-slice scores are reported beside them.
Nothing else in the design changes, and the final slice is still unscored.

## Results, written 2026-11-16 07:58 UTC

Files: `docs/children/gen1-raw-selection.json`, `gen1.json`, `gen2.json`,
`final.json`, `explain.json`; reader checks in `data/benchmarks/children/`.

**Reader checks** (one low-tier reader per hit, from the title and abstract,
blind to model scores). "Not confirmed" means the checked paper does not
report superconductivity in that compound at ambient pressure. Some of those
compounds are real superconductors measured in another paper (the reader's
notes name CaSn3, CaBi2 and HfV2Ga4 as "treated as known"). They are left out
of the hits, and they are not counted as negatives either.

| Slice | Raw hits | Confirmed | Not confirmed | Unclear |
| --- | --- | --- | --- | --- |
| Selection (before 2016) | 137 | **7** | 123 | 7 |
| Final (2016 to 2021) | 54 | **11** | 35 | 8 |

**Generation 1** (AUC = mean percentile of the hits in the 25,136-compound pool):

| Child | On the 7 confirmed selection hits | On all 137 raw hits |
| --- | --- | --- |
| `boost` (baseline) | **0.812** | 0.453 |
| `physics` | 0.755 | 0.442 |
| `bag` | 0.716 | 0.439 |
| `mlp` | 0.712 | 0.497 |
| `knn` (memory) | 0.547 | 0.469 |

Winner `boost`, runner-up `physics`.

**Generation 2** (offspring of `boost`, same 7 hits):

| Child | AUC | Against the parent |
| --- | --- | --- |
| `elite` (boost unchanged) | 0.812 | |
| `variant_a` (smaller trees, 15 leaves) | 0.791 | −0.021 (−0.070 to +0.019) |
| `variant_b` (larger trees, 63 leaves) | **0.857** | **+0.045 (+0.023 to +0.066)** |
| `cross2` (boost + physics) | 0.789 | −0.023 (−0.062 to +0.007) |
| `cross3` (boost + physics + bag) | 0.770 | −0.042 (−0.069 to −0.013) |

By the promotion rule, `variant_b` is the champion. A bootstrap over 7 hits
understates the uncertainty, so I would not trust this promotion on its own.

**Final test, scored once** (later discoveries, 2016 to 2021):

| | Champion (`variant_b`) | Baseline (`boost`) | Champion minus baseline |
| --- | --- | --- | --- |
| 11 confirmed hits: AUC | 0.784 | **0.766** | +0.018 (−0.007 to +0.047) |
| 11 confirmed hits: in the top 5% of the pool | 4 | 4 | |
| 11 confirmed hits: in the top 1% | 1 | 1 | |
| All 54 raw hits: AUC | 0.680 | 0.662 | +0.019 (−0.002 to +0.040) |

- The baseline's AUC is above chance by 0.27 (interval 0.08 to 0.40).
- By the baseline's percentile, the confirmed hits rank: ThCoC2 99.9,
  Nb5Pt3O 98.3, NaSn2As2 97.3, PbTaS2 97.3, SnTaS2 94.7, BaPtSb 91.0,
  LaIr3 83.5, CaSb2 78.2, TbPdBi 48.6, Sr3SnO 32.8, UTe2 20.8.

**Against the predictions:**

| Prediction | Chance | Result |
| --- | --- | --- |
| `boost` ranks the selection hits at 0.70 or more | 60% | yes on confirmed hits (0.812); no on the raw slice (0.453), the one the prediction was written for |
| Generation 1 winner is `boost` or `bag` | 55% | yes (`boost`) |
| `knn` within 0.03 of `boost` | 40% | no (0.547 against 0.812) |
| `physics` beats `boost` | 30% | no |
| Generation 2 promotes a child | 20% | yes |
| Champion beats `boost` on the final slice, interval above zero | 10% | no |
| `boost` ranks the confirmed final hits at 0.65 or more | 55% | yes (0.766) |

**What it shows.**
- [Likely] **The baseline does rank real later discoveries.** Trained only on
  what a 2018 database held, it put 11 confirmed superconductors first
  reported in 2016 to 2021 at a median of the 91st percentile of 25,136
  candidates. Four of the 11 were in its top 5%; chance gives about half of
  one.
  - It is not only telling metals from insulators. Against the 16,415
    metallic pool compounds alone, the AUC is 0.782 (exploratory; computed
    after the final score).
  - Eleven hits is a small test. The interval for the AUC is roughly 0.58 to
    0.90.
- **Its misses are the unconventional ones.** UTe2 (a spin-triplet uranium
  compound), Sr3SnO and TbPdBi rank at or below the middle. [Likely] A model
  that learned from mostly conventional superconductors does not see those
  coming.
- **Two generations of selection found nothing that holds.** The promoted
  child is +0.018 on the final slice, with an interval that includes zero.
  With 7 selection hits, selection is choosing on noise.
- **Pure memory does poorly** (`knn`, 0.547 on the confirmed selection hits).
  [Speculative] The boosted model is doing more than finding the nearest
  known compound.
- **The physics input did not help** (0.755 against 0.812), in line with rung
  4's result that the calculated set covers too little chemistry.

**How the champion works, as description only** (`explain.json`, 7 hits, so
rough):
- Shuffling the count of unfilled f-electron states costs the most (AUC
  −0.21), then melting temperature (−0.10) and valence-electron count (−0.06).
- Its ten best-ranked pool compounds each sit next to known superconductors of
  a familiar family: SrCaPb6 next to CaPb3, V6GaGe next to V3Ga-type alloys,
  TaB2Mo3C2 next to Mo2BC, Zr2SN2 next to ZrNCl. It proposes relatives of what
  it knows.

**Where I went wrong.**
- The selection slice as first designed was mostly extraction errors. I found
  that only after generation 1 had run on it, and amended the design.
- The amendment left 7 selection hits, too few to choose between children.
- The readers were single, low-tier and strict. The "not confirmed" pile holds
  real superconductors, so the hit lists are clean but incomplete.

**What a real version needs.**
- The NIMS SuperCon datasheet with reference years (`mdr.nims.go.jp`). It
  would give thousands of dated, curated first reports and a true date cutoff
  with three time slices.
- A second reader on the "not confirmed" pile here would roughly double the
  confirmed hits at small cost.

## Generations 3 to 5 on dated NIMS data: plan, written 2026-11-16 08:13 UTC, before any model was trained on it

**The owner's instruction (2026-11-16):** run three more generations, because
"some AIs take a very long time to bloom". He also approved access to the NIMS
host.

**The new data.** The MDR SuperCon Datasheet, version 240322
(doi:10.48505/nims.4487, CC BY 4.0, National Institute for Materials Science).
It has 33,459 curated records, each with a publication year, a Tc or a "not
superconducting down to" temperature. It is not committed;
`scripts/children_nims.py data` downloads it to `data/raw/nims-supercon/`.

Counted while writing this, from the data alone (no model):
- 24,409 records are usable (a year, a parseable formula of known elements,
  a Tc or a measured null, no applied pressure): 15,496 compositions.
- A composition is a superconductor if any record gives a Tc above 0, and a
  non-superconductor if its records only say "not superconducting down to".

| Slice | Trained on | New compositions, in element systems the training years never held | Superconductors | Not | Element systems |
| --- | --- | --- | --- | --- | --- |
| Selection | first seen up to 2010 | first seen 2011 to 2015: 1,130 | 932 | 198 | 401 |
| Final | first seen up to 2015 | first seen 2016 to 2020: 524 | 393 | 131 | 217 |

This is a true date cutoff with measured negatives, about a hundred times the
confirmed hits the first two generations had.

**Why this is "generations 3 to 5" and also a fresh start.** Generations 1
and 2 chose on 7 hits. Their one robust result (the baseline ranks later
discoveries) does not depend on that choice. Here every lineage is retrained
on the dated data, and the earlier choice only sets where the boost lineage
starts.

**Design: five lineages, so that a slow starter can bloom.** The first
experiment let only the winner have offspring. The owner's hypothesis is that
a kind of model that starts behind may overtake with more generations. So
each of the five kinds keeps its own lineage:
- Starting member: `boost` starts from generation 2's promoted setting (63
  leaves); `bag`, `knn`, `mlp` and `physics` start from their generation 1
  settings.
- Each generation (3, 4, 5), every lineage's current member has **two
  mutants**. A mutant moves one setting, chosen at random, one step up or
  down a fixed ladder (`LADDERS` in the script; the random seed is fixed by
  lineage, generation and mutant number).
- The best of parent and two mutants on the selection slice survives.
- Selection score: AUC of superconductors against measured
  non-superconductors, on the selection slice, model trained on compositions
  first seen up to 2010.

**Final test, once.** Each lineage's generation 5 member, and the baseline
(`boost` with generation 1's settings), are retrained on compositions first
seen up to 2015 and scored on the final slice. Six models, one scoring.
- Primary: each lineage's AUC minus the baseline's, with an interval from
  1,000 resamples of whole element systems. With five comparisons, a
  difference counts only if its 99% interval lies above zero.
- "Bloomed late" means: a lineage that was below `boost` at the start of
  generation 3 beats the baseline on the final slice by that standard.
- Also reported: AUC by family (iron-based, oxides, other), and each
  lineage's selection AUC by generation.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Baseline's final AUC is 0.70 or more | 55% |
| `boost` or `bag` has the best final AUC | 65% |
| Any lineage beats the baseline on the final slice by the standard above | 15% |
| A lineage that started behind "blooms late" by the definition above | 7% |
| Every lineage's selection AUC rises from generation 3 to 5 (it cannot fall, by construction), by a median of at least 0.01 | 50% |
| `knn` final AUC is at least 0.10 below the baseline | 45% |

**My view, stated before the run.** More generations add no information; more
data does. Three generations of two mutants each is a small search around
each starting point, and the selection score is noisy even with 1,130
compositions. I expect small gains on the selection slice that mostly do not
carry to the final slice. The NIMS data is the real improvement here.

**What this cannot show.**
- "First seen" is the year of the reference NIMS entered, which can lag the
  discovery. A compound first seen in 2012 may have been known in 2009.
- New compositions in new element systems still include members of known
  families (a new iron arsenide in 2012). The by-family table shows how much
  of the score comes from those.
- The physics input comes from calculations published in 2025 on structures
  known by then, so the `physics` lineage sees a little of the future.
