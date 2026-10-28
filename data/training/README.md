# Reviewed training data

## Phase 2 harmonized schema

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

Allowed review states are `pending`, `confirmed`, and `excluded`. Provenance is
one of `literature`, `synthetic_demo`, or `curated`; `is_synthetic` must agree
with that value. Only rows with `review_status=confirmed` and
`is_synthetic=0` enter model fitting. Synthetic rows remain available for
fixtures and pipeline smoke tests, while excluded rows are retained as
negative controls for the promotion gate.

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
schema on every run. The Phase 2 surrogate is trained from the harmonized file;
the legacy reviewed-record workflow remains available for paper-level curation.

Required review decisions:

- `evidence_status` must be `confirmed`.
- `measurement_method` must identify the evidence for the transition.
- `pressure_condition` must be `ambient`, `known`, `unknown`, or
  `not_reported`; use `known` only with a numeric `pressure_gpa`.
- `source_doi`, `source_record_id`, `reviewer`, and `review_date` are required.
- Unknown pressure stays explicit and is not silently treated as ambient.
