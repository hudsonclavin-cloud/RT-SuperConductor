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
