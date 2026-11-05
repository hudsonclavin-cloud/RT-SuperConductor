# Reviewed training data

## Phase 2 harmonized schema

The schema's current trust tiers and training boundary are described below.

`harmonized_supercon.csv` is the validated multi-source boundary for the
pressure-aware surrogate. It combines SuperCon2, 3DSC, SuperMat, and UCI rows.
The required columns are `record_id`, `source_db`, `formula`, `tc_kelvin`,
`applied_pressure_gpa`, `measurement_method`, `sample_form`, `doi`, and
`review_status`, `provenance`, and `is_synthetic`. `chemical_family` is required
by the grouped model, while `mp_id`, `energy_above_hull`, and
`spacegroup_number` are enrichment columns.
`applied_pressure_gpa` records pressure during the superconducting measurement.
The optional `synthesis_pressure_gpa` records pressure used to synthesize or
process the sample. Keep the two values separate: high-pressure synthesis does
not mean the superconducting transition was measured under high pressure. A
missing synthesis pressure remains blank/unknown.

Validate it with:

```bash
PYTHONPATH=src python scripts/validate_harmonized.py
```

Allowed review states are `pending`, `confirmed`, `curated_database`, and
`excluded`. Provenance is one of `literature`, `synthetic_demo`, or `curated`;
`is_synthetic` must agree with that value. The harmonized boundary permits
`0 <= tc_kelvin <= 300` so 3DSC non-superconducting controls are retained.

Production training admits only
`review_status in {confirmed, curated_database}` with `is_synthetic=0`.
Automatic SuperCon2 extraction is always excluded; the five human-reviewed
literature rows are allowed because they carry `provenance=literature` and a
confirmed review. Synthetic rows remain available for fixtures and smoke
tests, while excluded rows remain negative controls for evaluation.

Download and ingest the real database snapshot with:

```bash
python scripts/download_training_data.py --cif-limit 128
python scripts/ingest_3dsc.py
```

The importer maps `sc_class` to `chemical_family`, preserves `Tc=0`, copies
Materials Project IDs and hull energies, derives space-group numbers from
available CIFs, and records its ambient-pressure assumption in
`data/processed/3dsc-ingestion-report.json`. The 3DSC table and structures are
CC BY 4.0 source data; raw files stay under ignored `data/raw/`.

The five confirmed Y-cuprate rows in `reviewed/reviewed_records.csv` are
bridged idempotently into the harmonized table with authentic DOI references:

```bash
PYTHONPATH=src python scripts/bridge_reviewed_records.py
```

This directory is reserved for human-reviewed labels that are safe to use in
model training. Put confirmed records in `reviewed/reviewed_records.csv` (or
another CSV in that directory) using the header in the template. Every row
must include its source DOI or record identifier, reviewer, review date, and
measurement context.

Automatically extracted SuperCon2 rows do not belong here until a reviewer has
confirmed that the reported transition is superconducting, identified the
measurement method, and verified the pressure and sample context.

The file `review_queue_supercon2.csv` is a pending work list generated from
the NIMS export. Its rows are hypotheses for paper review, not labels. Use
`scripts/build_supercon2_review_queue.py` to regenerate it from a fresh export.

`review_queue_family_leads.csv` is a second pending queue for iron-based,
MgB2, and hydride records found through the public NIMS and SuperMat sources.
It includes calculated and extreme-pressure leads for coverage, but those must
remain separate from measured ambient-pressure evidence.

`harmonized_formula_descriptors.csv` is a derived descriptor table. It records
neutral-atom shell and orbital proxies, Lewis-dot outer-electron proxies, and
explicit status fields for structure-dependent bonding. It is not a substitute
for crystal-structure or electronic-density analysis.

Structure-dependent inputs are produced by the manual **Structure and density
inputs** workflow. Its artifact contains Materials Project CIF references and
an electronic-analysis manifest. Bader, ELF, COHP, and Wannier status remains
pending until a real DFT calculation is run and its outputs are attached.

The continuous-training workflow validates this directory and the harmonized
schema on every run. The active-learning cycle trains the UCI composition
baseline and the 3DSC pressure-aware model with family-held-out `GroupKFold`,
then emits a metrics report and ranked candidate CSV. Missing optional
structures or Materials Project credentials produce explicit blocked/degraded
reports rather than fabricated values.

The Phase 2 surrogate is trained from the harmonized file; the legacy
reviewed-record workflow remains available for paper-level curation.

Required review decisions:

- `evidence_status` must be `confirmed`.
- `measurement_method` must identify the evidence for the transition.
- `pressure_condition` must be `ambient`, `known`, `unknown`, or
  `not_reported`; use `known` only with a numeric `pressure_gpa`.
- `pressure_gpa` is measurement/applied pressure. The optional
  `synthesis_pressure_gpa` records synthesis pressure independently; when given,
  it must be a nonnegative numeric value in GPa. Do not infer one from the other.
- `source_doi`, `source_record_id`, `reviewer`, and `review_date` are required.
- Unknown pressure stays explicit and is not silently treated as ambient.

## Measurement pressure remains unknown when the source omits it

New 3DSC imports leave `applied_pressure_gpa` blank because the source table does not specify measurement pressure. The former 1-atm screening target is retained only in ingestion-report metadata, not stored as an observation. The harmonized schema round-trips blank measurement pressure as `None`; promotion checks exclude unknown pressure from both ambient and high-pressure benchmark subsets. Existing reviewed rows and their observed pressures are preserved. `synthesis_pressure_gpa` describes preparation and never fills missing measurement pressure. The physics-feature comparisons drop measurement pressure from their matrices and therefore remain pressure-agnostic Tc benchmarks, not validations of ambient-pressure performance.
