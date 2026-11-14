# Blinded literature check of the made-metals list

`made-metals.csv` holds 160 compounds from the 7,693 that qualify for
`docs/scratchnet-candidates-boosting-top-made-metals.csv` (made, calculated
metal, outside the copper-oxide and iron families, no copper):

- `top`: the first 60 rows of that list (boosting probability 0.89 and above);
- `random`: 60 drawn at random from the rest (seed 20261114);
- `bottom`: 40 drawn at random from the 400 the model rates least likely.

The readers saw only a case id and a formula, shuffled, so they did not know
which group a compound came from or what the model said about it. One reader
per five compounds searched the literature (13 to 14 November 2026) and gave a
verdict from a fixed list; every claim of superconductivity then went to a
second, sceptical reader, who changed three of them (`first_reader_verdict`
keeps the original). The model's numbers were joined back afterwards.

Verdicts: `superconductor_ambient`; `superconductor_only_pressure_doping_or_film`;
`measured_no_superconductivity` (measured down to about 2 K or lower);
`magnetic_or_insulating_no_superconductivity`; `no_low_temperature_data_found`.

## Limits

- **The search was incomplete for about half the rows.** The readers shared a
  limit on web searches, and for 91 of the 160 rows the note says the limit
  was reached, a source could not be opened, or the verdict rests on memory.
  For 80 of those the verdict is `no_low_temperature_data_found`, which there
  means "not found in the time allowed", not "never measured".
- `confidence` is the reader's own; 98 rows are `low`.
- Sources are those the readers could reach: many journal sites were blocked,
  so abstracts and citing papers were often used instead of the paper.

Summarise with:

    python scripts/literature_check_summary.py --cases data/benchmarks/literature-check/made-metals.csv \
        --compare top random --compare top bottom --compare random bottom --out docs/literature-check-made-metals.json

## Checksum

| File | sha256 |
| --- | --- |
| made-metals.csv | 4d9e7fabb8f2d246d5ef78f71a3df36cfbcadec5555de410aa4555b80f10fcdf |
| negatives-sample.csv | bf451ae901367fe617fe087b6938ad132f6833300b78721e7b2f24a3ec035687 |

# Blinded literature check of training negatives (added 14 November 2026)

`negatives-sample.csv` holds 80 compositions that the network's table labels
as non-superconductors. All are development rows (folds 0 to 4); none is from
the locked test set. The pool is the 1,226 development negatives with 2 to 4
elements, a whole-number formula (at most 30 atoms), family Other or Oxides,
and no copper.

- `suspect`: the 40 that out-of-fold gradient boosting rated most likely to
  superconduct (probability 0.932 to 0.995), at most two per element system.
- `random`: 40 drawn at random from the rest (seed 20261114). Their
  probabilities run from 0.054 to 0.889.

Blinding and second reading were as for the made-metals check. For alloys
written as whole-number formulas (Bi19Pb is Bi0.95Pb0.05), readers looked for
measurements within about 2 atomic percent. A second reader re-checked all 10
claims of superconductivity:

- 3 were confirmed as stated.
- 2 kept their verdict, but the cited source could not be opened.
- 5 were changed. Four alloys went to `no_low_temperature_data_found` because
  no reachable source gave a Tc, and RuSe2 went to insulating.

`second_reader_reason` gives the reason for each. Local file paths were
removed from that column.

Extra columns:

- `oof_boosting_p` and `oof_network_p`: the out-of-fold probabilities from
  `data/processed/scratchnet/round2-oof.npz`.
- `n_reports`, `n_zero` and `lineages`: the label's evidence, as in the table.

## Limits of the negatives check

- **The search was worse than for the made metals.** For 67 of the 80 rows the
  notes say the shared search limit ran out, or a source could not be opened.
  Physical Review, doi.org and several publishers were unreachable, and that
  is where most alloy measurements from the 1950s to 1970s are. 62 rows are
  `no_low_temperature_data_found`; here that means "not found", not
  "never measured".
- 54 of the 80 rows are `low` confidence.

Summarise with:

    python scripts/literature_check_summary.py --cases data/benchmarks/literature-check/negatives-sample.csv \
        --compare suspect random --out docs/literature-check-negatives.json
