# Published machine-learning picks that were made and measured

`cases.csv` holds 43 compounds that a published machine-learning or
calculation screen picked as likely superconductors and that a laboratory
then made and measured, from 14 arXiv papers (2018 to 2026). They are an
out-of-time test for this project's models: none of the results was in the
training table when it was built, except where the table says otherwise
(`scripts/score_external_cases.py` checks every row against the table).

## How the rows were made

1. A literature search (13 to 14 November 2026) collected candidate cases and
   read each source paper; this produced one row per measured compound.
2. A second reader opened each source paper again and checked every field of
   every row against the text, tables and figures (outcome, Tc, how Tc was
   defined, the lowest temperature measured, pressure, whether a model chose
   the compound before it was measured, whether it was already known). Where
   the two readers disagreed, a third read settled it; `second_reader = yes`
   marks the rows of the four papers that needed it.
3. All 43 rows were confirmed against their source. The changes made at that
   step were small: the Tc values of HfZr3Re8 and Hf3ZrRe8 swapped back to the
   paper's text where a figure table had them the wrong way round, lowest
   temperatures filled in or corrected, the Mg2IrH6 row keyed on the target
   instead of the product, and the outcome of the bcc-alloy samples of arXiv:2509.25186 put
   as "not formed as predicted".

`quote` and `where_in_paper` point to the place in the paper that supports the
row. The full notes of both readers are not stored here.

## Scoring rules (the `outcome` column)

The project's models see only the composition, so a row is scored on what a
sample of that overall composition did at ambient pressure:

- `superconductor`: superconductivity at ambient pressure in a sample whose
  superconducting phase is known. This includes the four samples of
  arXiv:2509.25186 that came out as one disordered bcc alloy at the nominal
  composition instead of the predicted ordered compound (`sample_phase` says
  so), since the composition is what the models score.
- `not_superconductor`: measured at ambient pressure with no superconductivity
  down to `lowest_measured_temperature_k`. Nine of the twelve come from one
  paper (arXiv:2509.25186) and were only measured down to 4.2 K; seven of
  those were not X-rayed. They are weak negatives: a 3 K superconductor would
  pass that test unseen.
- `excluded`, with the reason in `exclusion_reason`: superconductivity only
  under pressure (four rows) or a measurement only under pressure (one);
  multiphase samples where the paper does not say which phase superconducts
  (six); and Mg2IrH6, which did not form (the product was an insulator that
  no model had picked).

`reported_outcome` and `reported_tc_k` keep what the paper reported, before
these rules.

## Caveats

- 31 scored rows from 9 papers is a small test, and the cases are not
  independent: one paper supplies 13 of them. Intervals in the summary
  resample whole papers.
- These are picks that other groups' models already rated highly. They test
  whether a model can tell good picks from plausible-looking failures, not
  whether it can find superconductors in a random pool.
- YRu3B2 and LuRu3B2 are in the training table labelled as
  non-superconductors (an old measurement stopped at 1.2 K). PdBiTe is in it
  as a known superconductor. Those three are not counted as clean.
- LuRu3B2 was not itself a model pick: the screen had rejected it, and it was
  made as the neighbour of YRu3B2 (`chosen_by_model_before_measurement =
  unclear`).

## Checksum

| File | sha256 |
|---|---|
| cases.csv | 71df4c145c3fc1f8e796096468dd10c94984f205a58c7c96411f0a1a08e3186b |
