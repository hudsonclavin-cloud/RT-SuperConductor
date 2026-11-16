# Blind test of the generation 7 champion on papers from 2021 on

**Main idea:** the champion model has only selection evidence behind it,
because the dated database stops in 2020. This test collects compounds that
papers from 2021 on measured for superconductivity, freezes that table, and
only then scores the frozen models on it, once.

Plan written 2026-11-16 17:49 UTC, before any paper was read for it. The owner approved
building it.

## What is frozen before any paper is read

- **The two models.** The baseline (the ranking of record's boosting
  settings on the 155 formula features) and the champion (`all`: the same
  learner with all five generation 6 changes), as defined in
  `docs/children/knowledge-gen7.json`. Both are trained on every NIMS
  composition (15,496), with the script at commit `14bccf5`.
- **A check that they cannot drift.** Refitting the champion and re-scoring
  the 571 rung 4 survivors reproduces
  `data/benchmarks/rung4/shortlist2-second-opinion.csv` exactly (sha256
  `4a63bf25…`, checked today). The scoring command repeats that check and
  stops if it fails.
- **No model score is computed for any collected compound until the table is
  committed.** The readers are not told a model exists.

## What counts as a record

| Field | Rule |
| --- | --- |
| Paper | Published in 2021 or later, with a DOI or arXiv number that a search tool returned. |
| Sample | Bulk (polycrystal or single crystal) at ambient pressure. No films, interfaces, monolayers, gated or pressurised samples. Experiment, not calculation. |
| Composition | Fully stated. A doped series counts only at compositions the abstract states. |
| Superconductor | A Tc above 0 from resistivity, susceptibility or heat capacity. |
| Null | The abstract says no superconductivity was found, with the lowest temperature reached, and that temperature is 2 K or lower. |

## From records to the test set

1. Every record goes to a second low-tier reader, who sees only the paper's
   identifier and the claim and must confirm it from the abstract. Only
   confirmed records are kept.
2. One row per composition. A composition with any superconducting record is
   a superconductor (the NIMS rule).
3. Keep a composition only if its set of elements appears nowhere in the NIMS
   compositions. This is the rule the selection folds used, and it means
   neither model has seen that chemistry.
4. Commit the table. Then score.

## How the papers are found

- **A fixed grid, set here, so no one chooses papers by taste:** 6 publication
  years (2021 to 2026) × 5 themes = 30 low-tier readers.

  | Theme | What the reader searches for |
  | --- | --- |
  | 1 | Newly reported superconductors among intermetallic and ternary compounds |
  | 2 | Superconductivity in named structure families: noncentrosymmetric, Laves, A15, Heusler, kagome; and borides, carbides, nitrides, silicides, germanides |
  | 3 | Alloy superconductors, including high-entropy and medium-entropy alloys |
  | 4 | Nulls stated outright: "no superconductivity down to", "absence of superconductivity" |
  | 5 | Nulls inside property papers on new metallic compounds: measured to 2 K or below with no transition |

- Each reader returns up to 25 records with the abstract sentence it relied
  on. Readers work from titles and abstracts.
- I check a random 20 identifiers against a reference tool myself.

## The one scoring run

- **Primary endpoint:** AUC on the intermetallic ("Other") compositions,
  champion minus baseline, with 95% and 99% intervals from 1,000 resamples of
  element systems.
- **Secondary:** the same on all compositions; and each model's own AUC.
- **Also reported, with a warning:** the ranking of record (cycle 3). Its
  training table has no dates, so it may contain some of these compounds.
- **Sensitivity checks, fixed here:** nulls measured to 0.5 K or lower only;
  and superconductors whose paper claims a first report only.
- The command refuses a second run.

**What would count as a result.**

| Outcome | Reading |
| --- | --- |
| Champion's gain has a 95% interval above zero | The gain is confirmed on new data. The blocks go to the learning-cycle guards. |
| Gain positive, interval includes zero | Consistent with the selection evidence, not confirmed. |
| Gain zero or negative | The selection gain did not carry over. The champion is not adopted. |

**Power rule.** If the table has fewer than 80 intermetallic compositions, or
fewer than 25 nulls among them, the test is underpowered. It is still scored
once and reported, and no claim is made either way.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| The table meets the power rule | 55% |
| The champion's gain on intermetallics is above zero | 70% |
| Its 95% interval is above zero | 30% |
| The champion's own AUC on intermetallics is 0.70 or higher | 45% |
| The second reader rejects more than 15% of records | 50% |

## What can go wrong

- **Biggest risk: too few nulls.** Papers rarely report that a compound does
  not superconduct. The nulls that are published are compounds someone
  expected to superconduct, so they are hard cases, and AUC will look lower
  than on a fair sample.
- **Most likely failure:** the table falls short of the power rule and the
  answer is "no claim".
- **Abstract-only reading.** A Tc or a null stated only in the full text is
  missed, and a low-tier reader can misread one. The second reader is the
  guard; it is not a full-text check.
- **Not every compound is new.** A 2022 paper can re-measure a compound known
  for decades. The element-system rule keeps the models honest (they have not
  seen it), and the first-report sensitivity check shows whether it matters.
- **Opportunity cost:** about 40 low-tier reader runs.

## Amendment: a second harvest for nulls, 2026-11-16 18:05 UTC, before any score

**Where the first harvest ended.**

| Step | Count |
| --- | --- |
| Lines the 30 readers wrote | 265 |
| Records after checks and removing duplicates | 211 (175 superconductors, 36 nulls), from 168 papers |
| Confirmed by the second reader | 198 (6% rejected) |
| Compositions | 176 |
| In an element system the NIMS data never held | 99 |
| Of those, intermetallic | 95: 77 superconductors, **18 nulls** |

- **The power rule is not met:** 18 nulls against the 25 required.
- The second readers rejected 13 records. Three were papers whose first
  preprint appeared in late 2020, which the first readers had missed.
- My own checks: 20 random identifiers are all registered with Crossref under
  titles naming the right compound, and I read 4 confirmed nulls' abstracts
  myself; all 4 say what the readers recorded.

**The change.** Twelve more low-tier readers, two new null themes for each of
the six years, then second readers on their records as before.

| Theme | What the reader searches for |
| --- | --- |
| 6 | A paper that reports superconductivity in one compound and says a sister compound does not superconduct down to a stated temperature |
| 7 | Searches for superconductivity that came up empty: heavy-fermion, Kondo and magnetic intermetallics, and new metallic compounds, measured to 2 K or below |

**Why this is allowed, and what it risks.**
- [Certain] No model has scored any collected compound. I decided this from
  the table's size alone.
- Scoring 18 nulls would use the test up and, by the plan's own rule, allow no
  claim.
- [Likely] The new themes pull in a different kind of null: sister compounds
  of superconductors (hard cases) and f-electron magnets (easy cases, and far
  from the rung 4 pool). So the first-harvest table is also scored as a third
  sensitivity check, fixed here: intermetallics from themes 1 to 5 only.
- Nothing else changes: the record rules, the element-system rule, the
  endpoint, the intervals, and one scoring run.
- If the table still has fewer than 25 intermetallic nulls after this, there
  is no third harvest. It is scored once and reported as underpowered.

**Prediction [Speculative].** The power rule is met after the second harvest:
55%.
