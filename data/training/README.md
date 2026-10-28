# Reviewed training data

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

The continuous-training workflow validates this directory on every run and
reports the number of accepted records. It currently retrains the composition
baseline from the pinned UCI snapshot; a pressure-aware trainer will merge
these validated records into a separate exploratory prototype. The prototype
is saved with an explicit evaluation and promotion status; it does not replace
the benchmark model.

Required review decisions:

- `evidence_status` must be `confirmed`.
- `measurement_method` must identify the evidence for the transition.
- `pressure_condition` must be `ambient`, `known`, `unknown`, or
  `not_reported`; use `known` only with a numeric `pressure_gpa`.
- `source_doi`, `source_record_id`, `reviewer`, and `review_date` are required.
- Unknown pressure stays explicit and is not silently treated as ambient.
