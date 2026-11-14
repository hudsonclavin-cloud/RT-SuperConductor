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
