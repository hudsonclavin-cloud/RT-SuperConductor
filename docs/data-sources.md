# Data sources for pressure-aware research

## Immediate source: SuperMat 2.0

[SuperMat](https://github.com/lfoppiano/SuperMat) is the best next import for
the pressure-aware milestone. Its `SuperMat-2.0.csv` file has `material`,
`tcValue`, `pressure`, `me_method`, and source-paper filename fields. The source
repository states that the dataset is CC BY 4.0 and cites DOI
10.1080/27660400.2021.1918396.

Its pressure and transition values are free-text literature annotations. They
must be parsed conservatively and checked against the cited paper. It contains
no crystal structures, and many rows have no material annotation.

## Sources already imported

- UCI Superconductivity Data: useful composition features and `Tc`, but no
  row-level pressure or measurement method.
- 3DSC_MP: `Tc` plus Materials Project structures, but no pressure field.

These remain valuable for structure and baseline modeling, but neither can label
the ambient-versus-engineering-pressure target by itself.

## Supplementary sources to investigate

- NIMS SuperCon: a broad superconducting-material catalog. Obtain an authorized
  export and inspect its current fields before assuming pressure coverage.
- Materials Project: computed structures and properties for enriching candidates;
  it is not an experimental pressure/Tc evidence source. API access requires a
  user key and its terms must be followed.
- Paper-specific high-pressure hydride datasets: useful for physics validation,
  but each dataset must be imported with its paper, pressure range, calculation
  method, and license rather than merged as if it were a universal catalog.

## Import order

1. Normalize SuperMat pressure and measurement fields and link each row to its
   source-paper filename.
2. Resolve material names and formulas conservatively; preserve unresolved rows.
3. Link records to 3DSC or Materials Project structures where the match is
   reviewable.
4. Add NIMS SuperCon only after obtaining an authorized export and documenting
   its fields and terms.
