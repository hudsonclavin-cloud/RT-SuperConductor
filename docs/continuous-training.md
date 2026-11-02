# Continuous training workflow

`.github/workflows/continuous-training.yml` runs one active-learning cycle on a
weekly schedule, on changes to the trainer or training data, and from the
GitHub Actions **Run workflow** button. The same cycle is available locally:

```bash
python scripts/run_active_learning.py
```

Each run:

1. checks out the code;
2. installs the optional machine-learning dependencies;
3. downloads and SHA-256 verifies the pinned UCI archive and 3DSC table;
4. fetches a deterministic, stratified CIF subset from the pinned 3DSC commit;
5. bridges reviewed records and ingests the 3DSC table into the harmonized CSV;
6. trains the UCI composition baseline with family-held-out `GroupKFold`;
7. trains the pressure-aware 3DSC surrogate with curated `Tc=0` negatives;
8. compares Matminer structure features against composition-only when coverage
   supports two held-out families;
9. enumerates substitutions, applies the Materials Project hull filter when
   credentials exist, and writes `candidates/top_k.csv`;
10. writes metrics, dataset hashes, degraded/blocked reports, and model
    artifacts; and
11. uploads the complete run as a run-specific artifact and GitHub job summary.

The workflow deliberately does not replace a model in production or treat
automatically extracted literature as reviewed labels. New records should be
added under `data/training/` only after a human verifies the paper, transition,
pressure, method, and sample context. Synthetic demonstration rows are retained
for fixtures and screening smoke tests but are excluded from model weights and
cross-validation metrics. The reviewed-record validator and bridge run before
training so an incomplete or ambiguous label cannot silently enter the model.
The pressure-aware trainer emits a blocked report when empirical records do not
yet span at least two chemical families; candidate ranking then emits a header-only
shortlist rather than silently falling back to synthetic data. Missing
Materials Project credentials are a visible degraded state: hull energy stays
unknown and the shortlist says that stability review is required.


The workflow also refreshes up to 1,000 SuperCon2 and the full SuperMat paper-
review queues and prepares a small batch of prioritized review tasks. These are
literature leads only: they do not produce Tc predictions and are not training
labels until a human verifies the source paper, transition, measurement method,
pressure, and sample context. SuperMat records may need DOI resolution from
their paper filename.

A short run is expected for these modest CPU data and screening jobs. Re-running
unchanged data checks reproducibility; it does not create a new learning cycle.
The active-learning script downloads, ingests, fits and ranks candidates, but
no measurement or calculation result returns automatically unless a researcher
reviews and adds it as versioned evidence. Artificial delays add no science.
GitHub Actions prepares and evaluates inputs; it does not execute Quantum
ESPRESSO, phonons, EPW, Bader, ELF, or COHP calculations. Larger first-principles
jobs need an external workstation or HPC allocation, with outputs ingested and
provenance recorded before retraining.
