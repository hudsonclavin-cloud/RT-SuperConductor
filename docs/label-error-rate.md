# How wrong are the table's labels? A random-sample measurement: plan, fixed before any reading

**Main idea:** read the primary literature for 100 randomly chosen non-cuprate
rows: 50 labelled superconductors and 50 labelled nulls. The share found
wrong in each class estimates the table's label error rate. That rate sets
the ceiling on the AUC any model can reach, so it decides whether the owner's
goal of 0.9 can be met by better models or needs cleaner labels first.

Written 2026-11-27, before any row has been read. Input:
`docs/perplexity/label-reader-02.csv`, made by
`python scripts/perplexity_batches.py`.

## The sample

- **Population:** labelled non-cuprate rows in the development and train-only
  splits:
  - 6,642 superconductors and 1,957 nulls;
  - locked rows excluded.
- **Draw:** 50 of each, random (seed 20261127), shuffled together.
  - The file carries no label, Tc, score or reason, so the reader cannot
    tell which class a row came from.
  - `sample_02` in the script rebuilds the classes; no key file is kept.
- **What we already know** (`docs/negative-labels.md`, `docs/learning-cycles.md`):
  - **Random negatives:** a blinded check of 40 found 0 confirmed wrong
    (0 to 8.8%) and 1 probable. But 29 of the 40 came back "nothing found",
    mostly because old Physical Review papers could not be reached.
  - **Suspicious negatives:** among the 500 most suspicious, cycle 3 showed
    0.8% wrong, and about 3% wrong on the physics.
  - **Superconductor labels have never been checked.**

## How a row is read and counted

1. **Reader one:** ChatGPT Pro deep research or Perplexity, with the prompt
   in `docs/perplexity/README.md`, 20 rows per run.
2. **A row counts as wrong** only if:
   - reader one's verdict contradicts the label, and
   - it then passes the owner's correction rule: a second independent reader
     and a sceptical verifier.

   Bulk sample, ambient pressure, composition within 0.15 at.% (or a
   bracketing single-phase series within 2 at.%), with evidence from the
   paper.
3. **A row counts as right** if reader one, with a primary source, agrees
   with the label.
4. **Anything else is unresolved:** no source, abstract only with no clear
   answer, or a reading that contradicts the label but fails the rule.

## What is reported

- For each class:
  - wrong / right / unresolved counts;
  - the wrong rate among resolved rows, with a Wilson 95% interval.
- **Two bounds for the unresolved rows:**
  - all of them right;
  - all of them wrong at the resolved rate.
- **The implied AUC ceiling for a perfect ranker:**
  - define *a* = 1 − (wrong rate among labelled superconductors);
  - define *b* = 1 − (wrong rate among labelled nulls);
  - then ceiling ≈ *ab* + ½[*a*(1 − *b*) + (1 − *a*)*b*].
  - [Speculative] This assumes errors are spread at random. Real errors
    cluster in hard compounds, so read it as rough.
- **Results stay out of training.** Confirmed corrections go to
  `data/training/negative_evidence.csv` and enter training only through a
  learning cycle. Rows that are checked and unchanged go to
  `data/training/label-audit-reviewed.csv`.

## Predictions [Speculative]

| Quantity | Central | Range |
| --- | --- | --- |
| Wrong rate among labelled superconductors (resolved rows) | 4% | 1 to 10% |
| Wrong rate among labelled nulls (resolved rows) | 6% | 2 to 15% |
| Unresolved, superconductors | 30% | 15 to 50% |
| Unresolved, nulls | 60% | 40 to 80% |
| Implied ceiling | 0.95 | 0.90 to 0.98 |

- **Most likely outcome:** the ceiling is well above the current 0.79. Then
  label noise is not what holds the ranking back, and the levers are
  structures, physics inputs and data.
- **What would change that:** an error rate among labelled nulls above about
  15%. That would put the ceiling near 0.90, and label cleaning would come
  first.
- **The main weakness:** with 50 rows per class and many unresolved, the
  intervals will be wide, roughly ±10 points. A second sample of 100 can be
  drawn the same way if the first is inconclusive.

## Added before any reading (2026-11-27, later the same day): what backs the nulls

A data-source review warned that in SuperCon-MTG's source, a blank Tc was
often written as 0, and that 19% of the zeros MTG resolved by reading the
paper were superconductors. Checked against the raw NIMS datasheet
(`data/raw/nims-supercon/20240322_MDR_OAndM.txt`) for the 1,957 non-cuprate
nulls this sample is drawn from:

| Evidence in NIMS | Nulls |
| --- | --- |
| A measured floor: "not superconducting down to X K" | 1,691 (86%) |
| Tc written as 0, no floor | 22 (1%) |
| No NIMS null; the label comes from another source | 244 (12%) |

- **The warning does not carry over.** The review read an older pipeline
  file; [Likely] the current table's nulls are mostly measured floors.
- **What a null means here.** The floor's median is 1.2 K, and 88% are at or
  below 2 K. So "null" means "no superconductivity above about 1 to 2 K". A
  compound with Tc of 0.5 K counts as a null by that definition; the
  reading should record the floor so this is not scored as an error.
- **The predictions above are unchanged.** This was learned before any
  reading; it is recorded so the result can be read with it.
