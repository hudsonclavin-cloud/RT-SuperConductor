# Reviewed training data

## Harmonized schema and trust tiers

`harmonized_supercon.csv` is the validated multi-source boundary for the
pressure-aware surrogate. It combines SuperCon2, 3DSC, SuperMat, and UCI rows.
The required columns are `record_id`, `source_db`, `formula`, `tc_kelvin`,
`applied_pressure_gpa`, `measurement_method`, `sample_form`, `doi`, and
`review_status`, `provenance`, and `is_synthetic`. `chemical_family` is required
by the grouped model, while `mp_id`, `energy_above_hull`, and
`spacegroup_number` are enrichment columns.

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

The continuous-training workflow validates this directory and the harmonized
schema on every run. The active-learning cycle trains the UCI composition
baseline and the 3DSC pressure-aware model with family-held-out `GroupKFold`,
then emits a metrics report and ranked candidate CSV. Missing optional
structures or Materials Project credentials produce explicit blocked/degraded
reports rather than fabricated values.

Required review decisions:

- `evidence_status` must be `confirmed`.
- `measurement_method` must identify the evidence for the transition.
- `pressure_condition` must be `ambient`, `known`, `unknown`, or
  `not_reported`; use `known` only with a numeric `pressure_gpa`.
- `source_doi`, `source_record_id`, `reviewer`, and `review_date` are required.
- Unknown pressure stays explicit and is not silently treated as ambient.
