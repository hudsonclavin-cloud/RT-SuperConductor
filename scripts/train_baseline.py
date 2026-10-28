#!/usr/bin/env python3
"""Train a reproducible composition-only Tc baseline with grouped evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def _load_uci(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("UCI dataset is empty")
    fieldnames = list(rows[0])
    required = {"critical_temp", "material"}
    if not required.issubset(fieldnames):
        raise ValueError(f"UCI dataset is missing columns: {sorted(required)}")
    feature_names = [name for name in fieldnames if name not in required]
    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([float(row["critical_temp"]) for row in rows], dtype=float)
    materials = [row["material"] for row in rows]
    # Hold out elemental-system groups so closely related chemistry does not
    # appear on both sides of the evaluation split.
    groups = [
        ":".join(name for name in feature_names if float(row[name]) != 0.0)
        for row in rows
    ]
    return X, y, feature_names, materials, groups


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_k": float(mean_absolute_error(y_true, prediction)),
        "rmse_k": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def train_baseline(*, input_path: Path, output_dir: Path, random_state: int = 42) -> dict[str, object]:
    X, y, feature_names, materials, groups = _load_uci(input_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    constant = DummyRegressor(strategy="mean")
    constant.fit(X[train_idx], y[train_idx])
    constant_prediction = constant.predict(X[test_idx])

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    tree_predictions = np.asarray([tree.predict(X[test_idx]) for tree in model.estimators_])
    prediction = tree_predictions.mean(axis=0)
    uncertainty = tree_predictions.std(axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "baseline-predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material", "actual_tc_k", "predicted_tc_k", "uncertainty_k"))
        writer.writerows(
            zip(
                (materials[index] for index in test_idx),
                y[test_idx],
                prediction,
                uncertainty,
            )
        )
    model_path = output_dir / "baseline-model.joblib"
    joblib.dump(model, model_path)

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    metrics = {
        "model": "RandomForestRegressor",
        "target": "critical_temp",
        "features": feature_names,
        "dataset": str(input_path),
        "dataset_sha256": digest,
        "rows": int(len(y)),
        "feature_count": int(X.shape[1]),
        "group_count": int(len(set(groups))),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "random_state": random_state,
        "split": "GroupShuffleSplit by elemental-system signature",
        "baseline_metrics": _metrics(y[test_idx], constant_prediction),
        "model_metrics": _metrics(y[test_idx], prediction),
        "mean_tree_uncertainty_k": float(uncertainty.mean()),
        "output_model": str(model_path),
        "output_predictions": str(predictions_path),
        "interpretation": (
            "Composition-only Tc baseline. It has no pressure feature and is not an ambient-pressure predictor."
        ),
    }
    (output_dir / "baseline-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/uci/extracted/unique_m.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/baseline"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train_baseline(input_path=args.input, output_dir=args.output_dir, random_state=args.random_state), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
