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

## Generations 3 to 5 and the second reader: results, written 2026-11-16 08:23 UTC

Files: `docs/children/nims-data.json`, `nims-evolve.json`, `nims-final.json`,
`second-look.json`; `data/benchmarks/children/second-read.csv`.

**The data came out as counted in the plan:** 1,130 selection compositions and
524 final ones, all in element systems the training years never held.

**Selection slice by generation** (AUC, superconductors against measured
non-superconductors, first seen 2011 to 2015; model trained on data up to 2010):

| Lineage | Start | Gen 3 | Gen 4 | Gen 5 | What changed |
| --- | --- | --- | --- | --- | --- |
| `boost` | 0.592 | 0.600 | 0.602 | 0.614 | faster learning rate, back to 31 leaves, no L2 penalty |
| `bag` | 0.600 | 0.616 | 0.617 | 0.617 | 63 leaves, 16 bags |
| `knn` | 0.566 | 0.566 | 0.582 | 0.582 | 10 neighbours, city-block distance |
| `mlp` | 0.595 | 0.605 | 0.605 | 0.605 | slower learning rate |
| `physics` | 0.597 | 0.615 | 0.618 | 0.623 | faster learning rate, larger leaves, stronger L2 penalty |

The baseline (generation 1's `boost` settings) scores 0.598 on this slice.

**Final test, scored once** (524 compositions first seen 2016 to 2020, 393
superconductors, 217 element systems; models trained on data up to 2015):

| Model | AUC (95% interval) | Minus baseline (95%; 99%) | Beats baseline by the rule |
| --- | --- | --- | --- |
| Baseline | 0.685 (0.605 to 0.768) | | |
| `boost` | 0.704 (0.628 to 0.773) | +0.019 (−0.016 to +0.050; −0.029 to +0.058) | no |
| `bag` | **0.720** (0.649 to 0.792) | **+0.035** (+0.008 to +0.062; +0.001 to +0.070) | **yes, barely** |
| `knn` | 0.560 (0.477 to 0.649) | −0.125 (−0.215 to −0.043; −0.241 to −0.025) | no (worse) |
| `mlp` | 0.689 (0.612 to 0.763) | +0.005 (−0.045 to +0.047; −0.063 to +0.068) | no |
| `physics` | **0.720** (0.638 to 0.798) | +0.036 (−0.007 to +0.073; −0.021 to +0.082) | no |

**Final AUC by family:**

| Family | Compositions | Superconductors | Baseline | `boost` | `bag` | `knn` | `mlp` | `physics` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Other (mostly intermetallics) | 386 | 285 | 0.756 | 0.759 | 0.773 | 0.615 | 0.759 | **0.800** |
| Iron-based | 76 | 58 | 0.399 | 0.543 | 0.523 | 0.399 | 0.466 | 0.475 |
| Oxides | 43 | 33 | 0.521 | 0.479 | 0.549 | 0.311 | 0.491 | 0.412 |
| Cuprates | 19 | 17 | 0.677 | 0.677 | 0.765 | 0.677 | 0.853 | 0.735 |

**Against the predictions:**

| Prediction | Chance | Result |
| --- | --- | --- |
| Baseline's final AUC is 0.70 or more | 55% | no (0.685) |
| `boost` or `bag` has the best final AUC | 65% | no, by 0.0008: `physics` 0.7204, `bag` 0.7196 |
| Any lineage beats the baseline by the rule | 15% | **yes** (`bag`) |
| A lineage that started behind blooms late | 7% | no |
| Selection AUC rises by a median of at least 0.01 | 50% | yes (median +0.017 from start to generation 5) |
| `knn` is at least 0.10 below the baseline | 45% | yes (−0.125) |

**What it shows.**
- [Certain, for this test] **On truly dated data the models are much weaker
  than cross-validation suggested.** The baseline scores 0.685 on compounds
  first reported after its training years, against 0.79 in grouped
  cross-validation.
- **Nearly all of that skill is in conventional intermetallics.** In "Other"
  the baseline scores 0.756. For iron-based compounds and oxides it is at or
  below chance. [Likely] A formula model cannot tell which new iron arsenide
  or oxide will superconduct; that depends on doping and structure it does
  not see.
- **Averaging many models helped a little.** `bag` is +0.035 over the
  baseline, and its 99% interval just clears zero.
  - This is the one comparison of five that passed. With five tries, one
    marginal pass is weak evidence.
  - It does not show that evolution helped. The starting `bag` was never
    scored on the final slice, so the gain may come from bagging itself.
- **No late bloomer.** `knn` started last and finished last; `mlp` stayed
  level with the baseline. `physics` improved most on the selection slice and
  ties for best on the final one, but its interval includes zero.
  - [Speculative] `physics` at 0.800 on the intermetallics is the most
    interesting number here. It was picked after seeing the table, so it needs
    its own test.
- **Three generations bought about +0.02 on the selection slice.** On the
  final slice the evolved `boost` is +0.019 over the baseline, not
  distinguishable from zero. My view before the run stands: the data mattered,
  the generations did not much.

**The second reader on the first experiment's unconfirmed records.** Four
low-tier readers searched the literature for the 173 unconfirmed SuperCon2
compounds. They confirmed 12 more as bulk, ambient-pressure superconductors,
each with a Crossref-verified DOI: 6 first reported in 2016 or later, and 6
earlier.
- A labelled second look, with no model chosen on it: on the 17 confirmed
  2016-to-2021 discoveries the baseline's AUC is 0.789 (was 0.766 on 11), with
  6 in the top 5% of the pool.
- Two of the twelve look doubtful to me and are kept as the readers gave them:
  EuFe2As2 (superconducting only in samples made under pressure, per the
  note) and FeB4 (disputed).

**How the two experiments fit together.** They ask different questions.
- The pool test (first experiment): among 25,000 mostly untested compounds,
  do later-discovered superconductors rank high? Yes, about the 80th to 90th
  percentile. That is the question that matters for choosing what to study.
- The dated test (this one): of new compounds that somebody did measure, can
  the model tell which ones superconduct? Only moderately, 0.69 overall and
  0.76 for intermetallics. Chemists had already chosen those compounds as
  promising, so this is the harder question.

## Generations 6 and 7: children that differ in what they know. Plan, 2026-11-16 16:55 UTC, before any model is trained

**Main idea:** the earlier generations changed how each child learns and got
about +0.02. These two change what each child is told: the learner stays
fixed, and each child gets one extra block of physics or chemistry.

**A limit found first: no blind test is left.**
- [Certain] The NIMS file has 3 usable new-system compositions first reported
  after 2020. Its record stops there.
- [Certain] The 2016 to 2020 slice was the final test of generations 3 to 5
  and has been read.
- So nothing these generations produce can be a final result. Selection is
  the only honest evidence here. A blind test needs new data: superconductor
  papers from 2021 on, collected by readers, or a newer NIMS release.

**Selection folds** (same rules as before: test compositions are first seen in
the window, in an element system the training years never held).

| Fold | Trained on papers up to | Tested on first seen | Test compositions | Of them intermetallic ("Other") | Nulls among those |
| --- | --- | --- | --- | --- | --- |
| A | 1995 | 1996 to 2000 | 902 | 355 | 54 |
| B | 2000 | 2001 to 2005 | 835 | 491 | 79 |
| C | 2005 | 2006 to 2010 | 1,039 | 282 | 71 |
| D | 2010 | 2011 to 2015 | 1,130 | 487 | 104 |

- Folds A to C have never been scored. Fold D was the selection slice of
  generations 3 to 5, where only learner settings were chosen on it.
- These counts were made before writing this plan. No model was scored.

**Endpoint.** Primary: AUC on the intermetallic test compositions, averaged
over the four folds. Secondary: AUC on all test compositions, same average.
- This is a change from generations 3 to 5, which used all compositions. The
  reason: the rung 4 pool is copper-free, non-magnetic metals, all "Other".
- [Certain] I chose it knowing that on the spent 2016 to 2020 slice the
  baseline scored 0.756 on intermetallics and the physics child 0.800.

**Learner, fixed for every child:** gradient boosting with the ranking of
record's settings (learning rate 0.05, 31 leaves, minimum leaf 20, L2 1.0),
on the 155 formula features. That alone is the baseline.

**Generation 6: five children, one change each.**

| Child | What it is told, beyond the 155 features |
| --- | --- |
| `elements` | Measured properties of the elements, summarised over the formula (mean, minimum, maximum, range, spread): the element's own superconducting Tc, bulk modulus, sound velocity, molar volume, electrical resistivity, thermal conductivity, thermal expansion, and a Debye-temperature estimate from sound velocity and molar volume. Plus the mean of mass^(-1/2) (the isotope-effect term) and the distance of the valence-electron count per atom from Matthias's two peaks (4.7 and 6.5). |
| `proxy` | The calculated-coupling proxy trained on Bercx et al.'s Table B (the same input the `physics` child had). |
| `alloy` | Semi-empirical alloy chemistry from the formula: Miedema's formation enthalpies (intermetallic, solid solution, amorphous) and the Yang solid-solution parameters (atomic-size mismatch δ and Ω), from matminer. |
| `tc-size` | No new input. It also learns how high Tc was (a second model fitted to log(1 + Tc), nulls as zero) and ranks by the average of the two models' ranks. |
| `family` | The chemical family as an input, and a second model trained on the test composition's own family only (when that family has at least 200 training rows). Ranks by the average of the two models' ranks. |

**Generation 7: breeding.** Order generation 6's changes by the primary
endpoint and keep those above the baseline.

| Child | Built from |
| --- | --- |
| `best` | The top change alone. |
| `best-two` | The top two changes together. |
| `best-three` | The top three together. |
| `best-two-bagged` | `best-two`, averaged over 8 group-bootstrap bags. |
| `all` | All five changes together, whatever they scored. |

- If fewer changes beat the baseline than a child needs, that child uses the
  ones that did. If none did, generation 7 is `all` and a bagged baseline.

**Rule for a champion.** The generation 7 child with the highest primary
endpoint is the champion only if its gain over the baseline has a 99%
interval above zero. The interval comes from 1,000 bootstrap resamples of
element systems within each fold, the four folds averaged in each resample.
Otherwise there is no champion and the baseline stands.

**Second look, not a test.** The champion (or, with no champion, the best
generation 7 child) and the baseline are scored once on the spent 2016 to
2020 slice and reported as a labelled second look. It decides nothing.

**What this does not change.** The ranking of record. A winning block would
still need its own test under the learning-cycle guards before it is adopted.

**Predictions [Speculative].**

| Outcome | Chance |
| --- | --- |
| `elements` beats the baseline on the primary endpoint (any margin) | 55% |
| `proxy` beats the baseline on the primary endpoint | 55% |
| `alloy` beats the baseline on the primary endpoint | 45% |
| A champion is declared | 30% |
| The best generation 7 child is above the baseline on the second look's intermetallics | 60% |

## Generations 6 and 7: results, 2026-11-16 17:16 UTC

**Main result:** the child given all five changes at once is the champion. On
intermetallics it scores 0.752 against the baseline's 0.696, averaged over the
four selection folds, and its 99% interval is above zero. This is selection
evidence, not a blind test; no blind test exists (see the plan above).

**Generation 6: one change each** (mean over the four folds).

| Child | Intermetallics AUC | Gain | 99% interval of the gain | All compositions AUC | Gain |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.696 | | | 0.655 | |
| `tc-size` | 0.723 | +0.027 | +0.007 to +0.048 | 0.661 | +0.006 |
| `elements` | 0.711 | +0.014 | −0.015 to +0.047 | 0.665 | +0.011 |
| `proxy` | 0.704 | +0.008 | −0.015 to +0.029 | 0.662 | +0.007 |
| `family` | 0.703 | +0.007 | −0.015 to +0.028 | 0.659 | +0.004 |
| `alloy` | 0.702 | +0.005 | −0.016 to +0.026 | 0.660 | +0.006 |

- All five are above the baseline, so all five went into the breeding.
- Only `tc-size` is clearly above it on its own. It adds no physics: it lets
  the model learn how high Tc was, not only whether there was one.

**Generation 7: breeding.**

| Child | Changes | Intermetallics AUC | Gain | 99% interval | All compositions: gain | 99% interval |
| --- | --- | --- | --- | --- | --- | --- |
| `best` | tc-size | 0.723 | +0.027 | +0.007 to +0.048 | +0.006 | −0.011 to +0.023 |
| `best-two` | tc-size, elements | 0.740 | +0.043 | +0.012 to +0.079 | +0.019 | −0.007 to +0.037 |
| `best-three` | tc-size, elements, proxy | 0.745 | +0.048 | +0.014 to +0.083 | +0.019 | −0.005 to +0.039 |
| `best-two-bagged` | tc-size, elements, 8 bags | 0.724 | +0.028 | −0.002 to +0.058 | +0.001 | −0.024 to +0.022 |
| **`all`** | all five | **0.752** | **+0.055** | **+0.022 to +0.091** | **+0.023** | **+0.001 to +0.043** |

- **Champion: `all`,** by the rule (highest primary endpoint, 99% interval
  above zero). It is ahead of the baseline in every fold.
- The gains add up: each change alone is small, and together they are about
  twice the best single one.
- Bagging did not help here. The bagged child is below the same child
  unbagged, the opposite of generations 3 to 5.

**By fold, intermetallics AUC.**

| Fold (tested on) | Baseline | `all` |
| --- | --- | --- |
| A (1996 to 2000) | 0.793 | 0.874 |
| B (2001 to 2005) | 0.618 | 0.680 |
| C (2006 to 2010) | 0.713 | 0.755 |
| D (2011 to 2015) | 0.661 | 0.698 |

**Second look at 2016 to 2020 (not blind; taken once, after the champion was
named and committed).**

| | Baseline | `all` | Gain | 99% interval |
| --- | --- | --- | --- | --- |
| Intermetallics (386) | 0.756 | 0.844 | +0.088 | +0.031 to +0.165 |
| All compositions (524) | 0.685 | 0.763 | +0.078 | +0.025 to +0.149 |

**Predictions against results.**

| Prediction | Chance given | Result |
| --- | --- | --- |
| `elements` beats the baseline on the primary endpoint | 55% | yes (+0.014) |
| `proxy` beats the baseline | 55% | yes (+0.008) |
| `alloy` beats the baseline | 45% | yes (+0.005) |
| A champion is declared | 30% | yes |
| The best generation 7 child is above the baseline on the second look's intermetallics | 60% | yes (+0.088) |

I gave a champion 30% and was too pessimistic.

**What I take from it.**

- [Likely] Telling the model more works better than tuning it. Three
  generations of tuning gave about +0.02; two generations of added knowledge
  gave +0.055 on the same kind of test.
- [Likely] The improvement is real and not an artifact of picking a winner.
  - `all` was defined before generation 6 ran and did not depend on its
    results.
  - It was the best of only five children, it wins in all four folds, and it
    wins again, by more, on a slice it was not chosen on.
- [Certain] That last slice is not blind. I designed these children knowing
  that a physics input had helped on it. The honest status is "strong
  selection evidence, awaiting a blind test".
- [Speculative] Which of the five changes carries the gain is not settled.
  `tc-size` is the largest alone, but the breeding did not test each change's
  removal from `all`. That is one more run.
- [Certain] The absolute level is still modest. 0.75 on intermetallics means
  the model orders a random superconductor above a random measured null three
  times in four.

**What it does not change, yet.** The ranking of record is the cycle 3
boosting model. `all` was trained on NIMS data, which the ranking of record
does not use. Adopting it needs the learning-cycle guards and, better, a blind
test.

**Next, in order of value.**
1. A blind test: superconductor papers from 2021 on, read by low-tier readers
   into a dated table (measured Tc, and measured nulls). This is the one thing
   that turns "selection evidence" into a result.
2. Leave-one-out on `all`: which changes can be dropped.
3. Score the rung 4 pool with `all` as a second opinion beside the ranking of
   record.
