# Continuous training workflow

`.github/workflows/continuous-training.yml` runs the first AI benchmark on a
weekly schedule, on changes to the trainer or reviewed training data, and from
the GitHub Actions **Run workflow** button.

Each run:

1. checks out the code;
2. installs the optional machine-learning dependencies;
3. downloads and SHA-256 verifies the pinned UCI snapshot;
4. trains the grouped composition baseline;
5. writes metrics, predictions, and the model; and
6. uploads them as a run-specific artifact and GitHub job summary.

The workflow deliberately does not replace a model in production or treat
automatically extracted literature as reviewed labels. New records should be
added under `data/training/` only after a human verifies the paper, transition,
pressure, method, and sample context. The next training milestone is merging
those reviewed records and adding pressure-aware targets with a separate
evaluation split.
