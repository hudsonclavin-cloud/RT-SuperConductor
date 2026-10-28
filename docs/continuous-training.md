# Continuous training workflow

`.github/workflows/continuous-training.yml` runs the first AI benchmark on a
weekly schedule, on changes to the trainer or reviewed training data, and from
the GitHub Actions **Run workflow** button.

Each run:

1. checks out the code;
2. installs the optional machine-learning dependencies;
3. validates human-reviewed records under `data/training/reviewed/`;
4. bridges confirmed reviewed records into the provenance-aware harmonized CSV;
5. validates provenance and `is_synthetic` metadata;
6. downloads and SHA-256 verifies the pinned UCI snapshot;
7. trains the grouped composition baseline;
8. attempts the pressure-aware surrogate using only confirmed empirical rows;
9. writes metrics, predictions, and blocked-data reports; and
10. uploads them as a run-specific artifact and GitHub job summary.

The workflow deliberately does not replace a model in production or treat
automatically extracted literature as reviewed labels. New records should be
added under `data/training/` only after a human verifies the paper, transition,
pressure, method, and sample context. Synthetic demonstration rows are retained
for fixtures and screening smoke tests but are excluded from model weights and
cross-validation metrics. The reviewed-record validator and bridge run before
training so an incomplete or ambiguous label cannot silently enter the model.
The pressure-aware trainer emits a blocked report when empirical records do not
yet span at least two chemical families; this keeps CI informative without
silently falling back to synthetic data.
