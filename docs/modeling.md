# Initial AI baseline

The first model is deliberately modest: a `RandomForestRegressor` trained on
the UCI dataset's elemental composition features and critical
temperature labels. A `DummyRegressor` mean predictor is reported beside it so
the model must demonstrate value over a trivial baseline.

The split uses `GroupShuffleSplit` on the elemental-system signature. This keeps
records with the same set of elements together and makes the first test harder
than a random row split. The output records the dataset checksum, feature list,
split, random seed, metrics, predictions, and tree-ensemble uncertainty.

Run it locally with:

```bash
python -m pip install -e '.[ml]'
PYTHONPATH=src python scripts/train_baseline.py
```

Outputs are written under `data/processed/baseline/`:

- `baseline-metrics.json`
- `baseline-predictions.csv`
- `baseline-model.joblib`

This model predicts critical temperature from composition only. It does not
predict pressure feasibility, crystal stability, synthesis success, magnetic
evidence, or room-temperature superconductivity. Those become separate tasks
after the baseline and its failure modes are understood.

## Pressure-aware reviewed-data prototype

The workflow also fits `scripts/train_pressure_aware.py` from confirmed rows in
`data/training/reviewed/`. It uses elemental fractions, applied pressure, and
measurement method. Its report includes source duplication, pressure coverage,
target-window coverage, and the concrete data gaps that block a credible
family-held-out evaluation. The current five records are only a pipeline
prototype: they span two composition families, four papers, and one pressure
level. Its training fit is diagnostic only and the model is never promoted.

Before evaluating this model, grow the reviewed set to at least 50 independent
records from at least 10 composition families, with multiple pressure levels
and explicit examples in the 273.15–298.15 K target window. Then add a
source-grouped and family-held-out evaluation before using its rankings to
choose computational work.
