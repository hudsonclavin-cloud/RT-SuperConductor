# Continuous training workflow

`.github/workflows/continuous-training.yml` runs the first AI benchmark on a
weekly schedule, on changes to the trainer or reviewed training data, and from
the GitHub Actions **Run workflow** button.

Each run:

1. checks out the code;
2. installs the optional machine-learning dependencies;
3. validates human-reviewed records under `data/training/reviewed/`;
4. downloads and SHA-256 verifies the pinned UCI snapshot;
5. trains the grouped composition baseline;
6. fits a separate pressure-aware prototype from the reviewed records;
7. writes metrics, predictions, and both model artifacts; and
8. uploads them as a run-specific artifact and GitHub job summary.

The workflow deliberately does not replace a model in production or treat
automatically extracted literature as reviewed labels. New records should be
added under `data/training/` only after a human verifies the paper, transition,
pressure, method, and sample context. The next training milestone is merging
those validated records and adding pressure-aware targets with a separate
evaluation split. The reviewed-record validator runs before training so an
incomplete or ambiguous label cannot silently enter the model. The pressure-
aware prototype withholds validation and promotion while the reviewed set has
fewer than 50 records across at least 10 composition families.
