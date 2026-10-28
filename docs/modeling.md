# Modeling workflows

## Initial AI baseline

The first model is a composition-only `RandomForestRegressor` trained on the
UCI dataset's elemental composition features and critical-temperature labels.
A `DummyRegressor` mean predictor is reported beside it. Its grouped split is
useful as a historical baseline, but it does not model pressure or stability.

## Phase 2 pressure-aware surrogate

`data/training/harmonized_supercon.csv` is the training boundary for
`scripts/train_pressure_aware.py`. The pipeline uses Matminer Magpie elemental
composition statistics, `applied_pressure_gpa`, and Materials Project
`energy_above_hull`. It fits a random-forest surrogate and reports strict
`GroupKFold` evaluation by `chemical_family`, along with out-of-fold and full-fit
predictions.

```bash
python -m pip install -e '.[ml,mp]'
PYTHONPATH=src python scripts/train_pressure_aware.py
```

The output includes `pressure-aware-model.joblib`,
`grouped-oof-predictions.csv`, `candidate_predictions.csv`, and
`promotion_predictions.csv`, and `pressure-aware-report.json`. The full-fit
predictions are for queue screening; the promotion file combines
family-held-out predictions for confirmed rows with fitted-model predictions
for excluded negative controls.

`scripts/enrich_materials_project.py` appends the lowest-energy polymorph's
`mp_id`, `energy_above_hull`, and `spacegroup_number` using `MP_API_KEY`. The
promotion workflow then applies the ambient MAE, hydride pressure-trend, and
excluded-reference false-positive thresholds in `scripts/promotion_gate.py`.

The model is a prioritization tool, not evidence of superconductivity. A
candidate must still receive structure-aware relaxation and experimental or
first-principles validation.
