# Second blind table: compounds a model predicted and experiment refuted

**Main idea:** the first blind test had 18 nulls and could not decide
anything. Abstracts almost never say that a compound does not superconduct,
so a second harvest of the same kind would fall short again. One public
dataset does hold many measured failures: the samples made and measured in
Pogue et al., "Closed-loop superconducting materials discovery" (npj
Computational Materials 9, 181, 2023). This table is built from it.

Plan written 2026-11-17 17:33 UTC, before any model in this project has scored any
compound in it. The owner supplied the paper and approved the download.

## Why another harvest of abstracts would not work

| Check | Result |
| --- | --- |
| Papers from 2021 on whose abstract contains "no superconductivity down to" (OpenAlex) | 1 |
| The same for six more null phrases | 2 to 44 each, many of them films, pressure or theory |
| Intermetallic nulls the first test's 12 extra null readers added in new element systems | 0 |
| Confirmed records of the first test that were never scored and are absent from both training tables | 26 intermetallic compositions, 10 of them nulls |

## The source

- **What it is.** The raw magnetic-susceptibility files of every sample the
  authors measured (dataset doi:10.34863/w3xm-w506, PARADIM Data
  Collective): 324 files, each named with the sample's composition. All were
  measured down to 2 K.
- **What the authors say.** Of 190 predictions, 65 were judged worth making.
  Five known superconductors were rediscovered and one new one was found in
  the Zr-In-Ni system. The rest did not superconduct above 2 K, and were fed
  back to their model as negatives.
- **What it lacks.** A table of results. The label of each sample is read
  here from its raw curve, by the rule below.
- The raw files stay under `data/raw/pogue-2023/` (git ignores it). What is
  committed is the derived table: composition, label, the numbers the rule
  used, and the file names.

## The null rule, fixed before any curve is analysed

| Step | Rule |
| --- | --- |
| Scans used | Susceptibility against temperature in a field of 100 Oe or less. Field sweeps are not used. |
| Signal | The real part of the AC susceptibility per gram (or the DC moment over field per gram where a file has no AC column), smoothed with a 3-point running median. |
| Shielding fraction | f = −(lowest smoothed value) × 4π × 7 g/cm³. A full superconductor gives about 1. The density is fixed at 7 because the true one is not known for every sample; f is therefore right to within about a factor of two. |
| **Null** | Every usable scan of the composition reaches 2.1 K or lower and has f below 0.02. |
| **Signal seen** | Any scan has f of 0.02 or more. The composition is left out. It is not counted as a superconductor, because a small signal is often an impurity (indium, tin, or the tantalum foil the authors mention). |
| No mass | If a file gives no sample mass, the mass in the file name is used; with neither, the file is not used. |

**Left out by rule, whatever the curve shows:**

- Compositions in the three systems the authors flag as unresolved: Zr-In-Ni,
  Zr-In-Cu and Zr-Fe-Sn, with or without added oxygen, nitrogen or indium.
- Mixtures of several targets in one sample, films, samples named only by a
  laboratory code, and samples named "reduced" (their oxygen content is not
  the named one).
- Any composition present in the NIMS file, in the ranking of record's
  table, or in the rows cycle 4 added. This also removes the known
  superconductors the authors made as a check.

## Superconductors for the table

The dataset supplies nulls only. Superconductors come from papers of 2021
on, by the first test's record rules (bulk, ambient pressure, a stated Tc,
confirmed by a second reader):

1. The 16 intermetallic superconductors of the first test's confirmed
   records that were never scored (new compositions in element systems the
   training data already holds), and its 10 unscored nulls.
2. A new harvest, found by a fixed search and not by taste: OpenAlex
   abstracts from 2021 on that contain "superconduct" and one of "new
   superconductor", "discovery of superconductivity" or "superconductivity
   in" in the title. Low-tier readers extract records from the abstracts I
   hand them; a second low-tier reader confirms each. Papers used in the
   first test are left out.

## From records to the table

- One row per composition. A composition is kept only if it is absent from
  both training tables and the added rows, and is not a near twin of one
  (within 0.02 on every element), so no model has seen it or a doped
  variant of it.
- **Unlike the first test, the element system may be known.** Rung 4 picks
  new compositions in known families, so this is the closer test of how the
  ranking is used. It is an easier setting for every model than the first
  test's.
- The table is committed with its checksum. Then it is scored, once.

## The one scoring run

| Comparison | Models | Why |
| --- | --- | --- |
| A (primary) | Cycle 4 against cycle 3, the ranking of record's own models | The owner accepted cycle 4 on a narrow rule. Neither has scored these rows. |
| B | The lockbox champion (`elements` + `proxy` + `alloy` + `tc-size`) against the baseline, both trained on all NIMS data | The champion is "the model to beat" and this is its next blind data. |

- **Endpoint:** AUC on the intermetallic ("Other") rows, difference within
  each pair, with 95% and 99% intervals from 1,000 resamples of element
  systems. The same on all rows is secondary.
- **Sensitivity checks, fixed here:** nulls from the dataset only against
  all superconductors; and rows not containing iron, cobalt, nickel,
  manganese or chromium (so that "it is a magnet" cannot carry the result).
- The command refuses a second run.

**Power rule.** Fewer than 25 intermetallic nulls or fewer than 25
intermetallic superconductors: scored once, reported, no claim either way.

| Outcome (each comparison) | Reading |
| --- | --- |
| 95% interval above zero | The first model is better at telling real new superconductors from predicted ones that failed. |
| Interval includes zero | No difference shown. |
| 95% interval below zero | The second model is better. For comparison A that would be evidence against the cycle 4 decision, to put to the owner. |

## What this test is, and is not

- [Certain] The two classes come from different places: nulls from one
  group's machine-guided search, superconductors from the literature. A
  model could separate them by where they came from. So the AUC values
  themselves mean little. Only the difference between two models on the
  same rows is read.
- [Certain] The nulls are the hard kind for a formula model: another formula
  model ranked them in its top class and they failed.
- [Likely] Many of the nulls are iron-rich alloys and oxides, unlike the
  rung 4 pool. The no-magnetic-elements check addresses that.
- [Certain] "Null" means no transition above 2 K. A compound that
  superconducts below 2 K counts as a null here, as in the first test.
- [Certain] I read the labels from raw curves. The rule is fixed above, and
  every number it used is committed, so anyone can check a label.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| At least 25 intermetallic nulls survive the rules | 55% |
| The power rule is met on both sides | 45% |
| A: cycle 4 minus cycle 3 is above zero | 55% |
| A: its 95% interval excludes zero, either way | 15% |
| B: champion minus baseline is above zero | 60% |
| B: its 95% interval is above zero | 20% |

## Amendment: how the papers are chosen, 2026-11-17 17:36 UTC, before any abstract is read

The plan's search returns about 2,000 papers, far more than is worth
reading. Two fixed steps cut it down, set here before any abstract is read:

1. **A word filter.** Keep a paper only if its abstract gives a temperature
   in kelvin and neither title nor abstract contains any of: film,
   monolayer, bilayer, heterostructure, interface, pressure, GPa, hydride,
   twisted, graphene, gate, nanowire, first-principles, ab initio, density
   functional, predict, cuprate, nickelate, iron-based, pnictide, theor,
   ionic liquid, intercalat, proximity, Josephson, topological insulator.
2. **A fixed random sample.** If more than 400 papers pass, 400 are drawn
   with a fixed seed. They are read in batches of 25.

- The readers are handed the abstracts; they do not search. That is cheaper
  than the first test's readers and leaves them no choice of paper.
- [Likely] The filter throws away some good papers (any abstract that
  mentions pressure in passing). It cannot add a wrong record.

## Correction to the word filter, 2026-11-17 17:37 UTC, from counts only (no abstract read)

**What went wrong.** The filter kept 55 of 1,930 papers. The words were
matched as parts of words, so "investigate" counted as "gate" and "ambient
pressure" counted as "pressure". On a sample of 200 papers, "gate" hit 37
and "theor" 55.

**The corrected filter, fixed here.**

- Words are matched whole (or from the start of a word).
- "Ambient pressure" is removed before "pressure" is looked for.
- "Theor…", "predict…" and "calculat…" exclude a paper only when they are
  in its title. Experimental abstracts often compare with theory.
- Everything else is as in the amendment above: a temperature in kelvin is
  required, and at most 400 papers are drawn with the fixed seed.

- [Certain] I decided this from the counts. No abstract had been read by me
  or by a reader, and no record had been extracted.
- [Certain] 35 of the 200 sampled papers have no abstract in OpenAlex and
  cannot be used at all.

## Result, 2026-11-17 17:43 UTC

**Main result:** on compounds no model had seen, cycle 4 tells real new
superconductors from predicted-and-failed compounds better than cycle 3
does: +0.047 AUC on intermetallics, with a 95% interval above zero. The
lockbox champion is ahead of the baseline by +0.029, which is not
distinguishable from zero.

Numbers are from `docs/blind2/score.json` (command:
`python scripts/blind_test2.py score`, which refuses a second run). The
table was committed before scoring (commit `fd4bf46`, sha256
`98ffcf9e…`, full value in `docs/blind2/table.json`).

**How the table was built.**

| Step | Count |
| --- | --- |
| Raw susceptibility files downloaded (1 of 324 failed; it is a mixture) | 323 |
| Files with a usable temperature scan | 251 |
| Compositions measured | 162 |
| Left out: a system the authors flag as unresolved | 75 |
| Left out: a diamagnetic signal of 2% of full shielding or more | 32 |
| **Nulls by the rule** | **55** |
| Papers the search returned / passed the word filter | 1,930 / 105 |
| Records the first readers extracted (from 42 papers) | 47 |
| Confirmed by the second readers | 45 |
| Compositions from all three sources, before the rules | 300 |
| Dropped: scored in the first blind table | 114 |
| Dropped: in a training table | 95 |
| Dropped: a near twin of a training composition | 9 |
| **The table** | **82 compositions: 30 superconductors, 52 nulls** |
| **Intermetallics in it** | **65: 27 superconductors, 38 nulls** |

- The power rule (25 of each among intermetallics) is met.
- Of the 38 intermetallic nulls, 29 come from the dataset and 9 from the
  first test's unscored records. Of the 27 superconductors, 14 come from the
  first test's unscored records and 13 from the new harvest.
- The rule's sanity check on known compounds: every known superconductor the
  authors made as a check (Nb3Ge, V3Si, Nb3Al and others) came out as
  "signal seen", and known non-superconductors (Cu3Sn, Ta3Si) as nulls.
- My own check of the readers: 8 random records read against their
  abstracts; all 8 say what was recorded.

**Comparison A (primary): cycle 4 against cycle 3.**

| Rows | Cycle 3 AUC | Cycle 4 AUC | Difference | 95% interval | 99% interval |
| --- | --- | --- | --- | --- | --- |
| Intermetallics (65; 38 nulls) | 0.8197 | 0.8665 | +0.0468 | +0.0075 to +0.1013 | −0.0030 to +0.1248 |
| All compositions (82; 52 nulls) | 0.7545 | 0.8218 | +0.0673 | +0.0181 to +0.1253 | +0.0012 to +0.1481 |
| Intermetallics, dataset nulls only (56) | 0.8059 | 0.8621 | +0.0562 | +0.0111 to +0.1205 | +0.0036 to +0.1514 |
| Intermetallics without Fe, Co, Ni, Mn, Cr (41) | 0.7786 | 0.8238 | +0.0452 | −0.0127 to +0.1187 | −0.0313 to +0.1467 |

- Reading by the plan's table: **cycle 4 is better.**

**Comparison B: the lockbox champion against the baseline.**

| Rows | Baseline AUC | Champion AUC | Difference | 95% interval |
| --- | --- | --- | --- | --- |
| Intermetallics (65) | 0.8012 | 0.8304 | +0.0292 | −0.0445 to +0.1136 |
| All compositions (82) | 0.7397 | 0.7436 | +0.0039 | −0.0723 to +0.0801 |
| Intermetallics, dataset nulls only (56) | 0.7957 | 0.8570 | +0.0613 | −0.0196 to +0.1579 |
| Intermetallics without Fe, Co, Ni, Mn, Cr (41) | 0.8262 | 0.8536 | +0.0274 | −0.0682 to +0.1348 |

- Reading: **no difference shown.**

**What it shows.**

- [Likely] Adding the 851 NIMS rows made the ranking of record better at
  this task. It is the first evidence for the cycle 4 decision that does not
  come from NIMS's own rows. The owner accepted cycle 4 on a rule it passed
  narrowly; this supports that decision.
- [Certain] The evidence is moderate, not strong. The 99% interval on the
  primary rows just includes zero, and with magnetic elements removed the
  95% interval includes zero too (41 rows).
- [Likely] The champion's gain over the baseline is real but small. Three
  tests now point the same way with the same weakness: +0.043 on the lockbox
  (interval above zero), −0.012 on the first blind table and +0.029 here
  (both intervals include zero). A true gain of zero to +0.05 fits all three.
- [Certain] The AUC values themselves (0.80 to 0.87) should not be quoted as
  the model's skill. Nulls and superconductors come from different sources,
  as the plan said.

**The strongest argument against reading too much into A.**

- The nulls are mostly one group's choices: compounds a formula model
  ranked in its top class. Cycle 4 may simply be less fooled by that
  particular model's tastes.
- Two comparisons were run. A was named primary beforehand, so it is not a
  pick after the fact, but one result at the 95% level out of two looks is
  weaker than it sounds.
- The labels were read by me from raw curves. The rule was fixed first and
  every number is committed, but no second person has checked them.

**Predictions.**

| Outcome | Chance given | Result |
| --- | --- | --- |
| At least 25 intermetallic nulls survive the rules | 55% | Yes (38) |
| The power rule is met on both sides | 45% | Yes |
| A: cycle 4 minus cycle 3 is above zero | 55% | Yes (+0.047) |
| A: its 95% interval excludes zero, either way | 15% | Yes |
| B: champion minus baseline is above zero | 60% | Yes (+0.029) |
| B: its 95% interval is above zero | 20% | No |

- I gave a clear result on A only 15%. I expected 851 rows to matter less
  than they did, the same error I made on cycle 4 itself.

**What follows.**

- Cycle 4 stays the ranking of record, now with independent support.
- This table is spent. The 43 dataset nulls and the 45 new records could be
  added to training in a later cycle, under the owner's rules; that would
  need its own plan.
- One low-tier reader made a call outside its task (it listed sessions); it
  changed nothing. No other stray action occurred in ten reader runs.
