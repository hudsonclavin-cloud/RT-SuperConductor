# Data sources for pressure-aware research

## Immediate source: NIMS SuperCon2

[SuperCon2](https://mdr.nims.go.jp/datasets/2169a31f-ca73-4391-aceb-7acf15066f78)
is the strongest current source for the pressure-aware milestone. The supplied
export contains 40,324 records with extracted critical temperatures, applied
pressure text, measurement methods, synthesis-related fields, formulas, DOIs,
and publication metadata. The source is described as CC BY 4.0.

SuperCon2 is automatically extracted literature data. It can contain values
from unrelated magnetic or structural transitions, malformed pressure strings,
and duplicated passages from one paper. Scalar values are normalized
conservatively; ranges and ambiguous values remain unknown with their original
text in caveats. Every shortlist row must be checked against its cited paper.

The archive's observed main CSV has 23 columns, even though one supplied data
dictionary lists 28. Treat the observed export as versioned input and reconcile
the difference before training a model.

## Secondary pressure source: SuperMat 2.0

[SuperMat](https://github.com/lfoppiano/SuperMat) is the best next import for
the pressure-aware milestone. Its `SuperMat-2.0.csv` file has `material`,
`tcValue`, `pressure`, `me_method`, and source-paper filename fields. The source
repository states that the dataset is CC BY 4.0 and cites DOI
10.1080/27660400.2021.1918396.

Its pressure and transition values are free-text literature annotations. They
must be parsed conservatively and checked against the cited paper. It contains
no crystal structures, and many rows have no material annotation.

The weekly evidence loop reads the CSV from the public `lfoppiano/SuperMat`
repository and records the checked-out source revision and input checksum in
its artifact. The current upstream CSV is small (about 1,300 annotated rows),
not a massive training corpus. It has no DOI column; `filename` points toward
the paper record. The workflow creates a paper-review queue and never treats
these annotations as verified labels. The repository states CC BY 4.0 for the
dataset; retain attribution to the SuperMat authors and its cited publication.

## Sources already imported

- UCI Superconductivity Data: useful composition features and `Tc`, but no
  row-level pressure or measurement method.
- 3DSC_MP: `Tc` plus Materials Project structures, but no pressure field.
- Kaggle's [SUPERCON Dataset snapshot](https://www.kaggle.com/datasets/chaozhuang/supercon-dataset):
  a raw tab-separated SuperCon literature table with 26,321 rows, sample IDs,
  common names, structure labels, and journal citations. It does not provide
  row-level pressure or measurement-method fields, so it is useful for
  provenance and benchmark coverage rather than pressure screening. Its
  featurized file is retained separately and is not treated as new evidence.

These remain valuable for structure and baseline modeling, but neither can label
the ambient-versus-engineering-pressure target by itself.

## Supplementary sources to investigate

Recent search-result leads on mechanisms and tuning are catalogued in
[`recent-literature-leads.md`](recent-literature-leads.md). They are explicitly
unverified and are not part of the training corpus.

- NIMS SuperCon legacy catalog: a broad superconducting-material catalog that
  may provide additional records beyond the SuperCon2 extraction. Obtain an
  authorized export and inspect its current fields before assuming pressure
  coverage.
- Materials Project: computed structures and properties for enriching candidates;
  it is not an experimental pressure/Tc evidence source. API access requires a
  user key and its terms must be followed.
- Paper-specific high-pressure hydride datasets: useful for physics validation,
  but each dataset must be imported with its paper, pressure range, calculation
  method, and license rather than merged as if it were a universal catalog.

## Import order

1. Normalize SuperCon2 pressure, methods, synthesis text, and publication
   metadata; keep automatic-extraction caveats attached to each row.
2. Normalize SuperMat pressure and measurement fields and link each row to its
   source-paper filename.
3. Add the Kaggle SuperCon raw table for literature and citation coverage while
   keeping pressure unknown.
4. Resolve material names and formulas conservatively; preserve unresolved rows.
5. Link records to 3DSC or Materials Project structures where the match is
   reviewable.
