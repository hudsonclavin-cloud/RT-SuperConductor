# Modeling workflows

## Reproducible active-learning cycle

One command downloads the pinned UCI snapshot and the pinned 3DSC Materials
Project table, fetches a deterministic CIF subset, bridges reviewed evidence,
ingests 3DSC, trains both models, evaluates optional structure features, and
writes a reviewable candidate shortlist:

```bash
python -m pip install -e '.[ml]'
python scripts/run_active_learning.py
```

Raw data are kept under `data/raw/` and ignored by git. Use
`--all-cifs` for a complete local structure run; CI uses a stratified CIF
subset to keep the weekly job bounded. The run directory contains the dataset
hashes, JSON metrics, blocked reports, model artifacts, and
`candidates/top_k.csv`.

## Trust tiers and training boundary

The model does not treat every row as an observation:

| tier | schema values | production use |
| --- | --- | --- |
| demonstration | `provenance=synthetic_demo`, `is_synthetic=1` | fixtures and smoke tests only |
| reviewed literature | `provenance=literature`, `review_status=confirmed`, `is_synthetic=0` | eligible after human review |
| curated database | `provenance=curated`, `review_status=curated_database`, `is_synthetic=0` | eligible database evidence |
| automatic SuperCon2 extraction | pending or non-literature provenance | never model training |

`select_training_records` enforces this policy. The five reviewed Y-cuprate
records are literature evidence, not synthetic demonstrations; automatic
SuperCon2 extraction remains outside training even though the reviewed rows
retain their source identifier.

## Initial AI baseline

The composition-only `RandomForestRegressor` is trained on UCI `unique_m.csv`.
Because UCI does not provide chemical families, the script records a
deterministic family heuristic from the elemental columns and uses
`GroupKFold` to hold out each family. Each report contains MAE, RMSE, and R²
per held-out family, overall metrics, a mean predictor comparison, and ROC-AUC
for `Tc>0 K` and `Tc>77 K`. These family labels are a modeling convention, not
new source annotations.

## Pressure-aware surrogate

`data/training/harmonized_supercon.csv` is the training boundary for
`scripts/train_pressure_aware.py`. Every row carries `provenance` and
`is_synthetic`; the production filter is strictly
`review_status == "confirmed" and is_synthetic == 0`. Synthetic demonstration
rows remain available for fixtures, candidate-screen smoke tests, and gate
plumbing, but never enter model fitting or cross-validation metrics.

The pipeline uses Matminer's `ElementProperty.from_preset("magpie")` elemental
composition statistics, `applied_pressure_gpa`, and Materials Project
`energy_above_hull`. It fits a random-forest ensemble and reports strict
`GroupKFold` evaluation by `chemical_family`, with per-family regression
metrics, mean-predictor comparison, and the two classification ROC-AUC tasks.
All model artifacts carry the harmonized dataset SHA-256 in both the joblib
bundle and the JSON report.

```bash
python -m pip install -e '.[ml,mp]'
PYTHONPATH=src python scripts/train_pressure_aware.py
```

The five confirmed Y-cuprate records in
`data/training/reviewed/reviewed_records.csv` are bridged automatically by
`scripts/bridge_reviewed_records.py` with `provenance="literature"` and
`is_synthetic=0`. The 3DSC importer assigns `review_status=curated_database`,
keeps all `Tc=0` negatives, and records that its pressure is an explicit
ambient assumption. If an empirical set does not contain two independent
families, the trainer fails closed; the cycle writes a blocked report without
contaminating model weights with synthetic rows.

The output includes `pressure-aware-model.joblib`,
`grouped-oof-predictions.csv`, `candidate_predictions.csv`, and
`promotion_predictions.csv`, and `pressure-aware-report.json`. The full-fit
predictions are for queue screening; the promotion file combines
family-held-out predictions for eligible rows with fitted-model predictions
for excluded negative controls.

## Structure-aware comparison

`scripts/evaluate_structure_features.py` adds Matminer's
`DensityFeatures` and `GlobalSymmetryFeatures` to the composition features and
repeats the same family-held-out split on CIF-backed records. Its report says
whether the combined model beats composition-only by MAE. If the downloaded
CIF subset does not cover two empirical families, it writes
`blocked_insufficient_structures`; this is a coverage limitation, not a claim
that structure features help. ALIGNN/M3GNet is reserved as an optional second
stage and is reported as not attempted in the baseline cycle.

## Candidate generation and acquisition

`scripts/rank_candidates.py` uses pymatgen to make same-family elemental
substitutions, removes observed formulas, queries Materials Project when
`MP_API_KEY` is available, and rejects known candidates at or above
`0.05 eV/atom`. Without a key it degrades explicitly: hull energy remains
unknown, plausibility is zero, and every shortlist row carries a stability
review reason. The acquisition function caps the uncertainty reward and
multiplies it by plausibility, so uncertainty alone cannot promote an
implausible row.

`data/processed/candidates/top_k.csv` is a shortlist for human review. Its
prediction is a hypothesis from the full-fit ensemble, not a measurement or a
claim that any candidate superconducts.

`scripts/enrich_materials_project.py` appends the lowest-energy polymorph's
`mp_id`, `energy_above_hull`, and `spacegroup_number` using `MP_API_KEY`. The
promotion workflow then applies the ambient MAE, hydride pressure-trend, and
excluded-reference false-positive thresholds in `scripts/promotion_gate.py`.

The model is a prioritization tool, not evidence of superconductivity. A
candidate must still receive structure-aware relaxation and experimental or
first-principles validation.
