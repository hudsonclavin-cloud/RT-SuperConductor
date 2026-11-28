# Mining measured nulls from full-text papers: pilot

**Main idea:** measured negatives from 2015–2026 are the scarcest labels.
- The source survey (`docs/training-data-sources.md`) found that
  OpenAlex's full-text search is the one open route to them.
- This pilot finds the sentences that state a null ("no superconductivity
  down to X K") in arXiv full texts.
- Stage 1 is a pattern match and costs no model reading.
- Stage 2, still to do, is a reader per sentence: which compound, bulk or
  not, ambient pressure or not. Its leads then go through the owner's
  correction rule.

Owner approved 2026-11-27. Script: `scripts/null_mining.py`. Summary:
`docs/null-mining.json`. Texts and sentences:
`data/processed/null-mining/` (not committed).

## Stage 1 (2026-11-27)

| Step | Count |
| --- | --- |
| OpenAlex works since 2010 matching 10 phrases in full text | 3,191 |
| With an arXiv version | 1,775 |
| Full texts fetched (pilot, "down to" phrases first) | 199 |
| Sentences stating a null with a floor | 102, from 74 papers |
| Floor, lower quartile / median / upper quartile | 0.5 / 1.8 / 2.0 K |

**A random look at five sentences:**
- **Three are measured nulls:**
  - LaPt4Ge12−xSbx at x = 3, none down to 0.4 K;
  - an antiferromagnet with no superconductivity down to 1.8 K;
  - Ni-doped crystals down to 2 K.
- **Two would be rejected by a reader:**
  - a titanium probe in a device, not the sample;
  - a null measured under pressure.

So stage 1 finds the right kind of sentence, and stage 2 is needed for
every one of them.

## Stage 2 (next)

1. **For each sentence**, a reader records:
   - the compound and its composition;
   - the sample form (bulk, film or device);
   - the pressure;
   - the floor;
   - a verbatim quote.
2. **Blind:** the reader is not told whether the compound is in the table,
   or with what label.
3. **The correction rule:** rows that would add a new null, or contradict a
   label, go through the owner's correction rule before entering
   `data/training/negative_evidence.csv`.
