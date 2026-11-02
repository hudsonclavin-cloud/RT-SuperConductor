#!/usr/bin/env python3
"""Train a reproducible UCI composition baseline with family-held-out CV.

The UCI table has no source-provided chemical-family field. This script uses
an explicit, deterministic family mapping from the non-zero elemental columns,
then holds out each family with ``GroupKFold``. The report includes a mean
predictor, regression metrics, and ROC-AUC for both superconductivity labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def _family_from_row(row: dict[str, str], feature_names: list[str]) -> str:
    elements = {name for name in feature_names if float(row[name]) != 0.0}
    if "Cu" in elements and "O" in elements:
        return "Cuprates"
    if "Fe" in elements and elements.intersection({"As", "Se", "P", "Te"}):
        return "Iron-based"
    if "H" in elements and len(elements) > 1:
        return "Hydrides"
    if "C" in elements and len(elements) <= 3:
        return "Carbon-based"
    if len(elements) <= 2:
        return "Simple-elements"
    return "Other"


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
    families = [_family_from_row(row, feature_names) for row in rows]
    return X, y, feature_names, materials, families


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "mae_k": float(mean_absolute_error(y_true, prediction)),
        "rmse_k": float(np.sqrt(mean_squared_error(y_true, prediction))),
    }
    if len(y_true) >= 2:
        value = float(r2_score(y_true, prediction))
        result["r2"] = value if math.isfinite(value) else None
    else:
        result["r2"] = None
    return result


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    all_labels = (y_true > threshold).astype(int)
    labels = all_labels
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    auc = None
    if len(labels) and len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, scores))
    return {
        "threshold_k": threshold,
        "positive_count": int(all_labels.sum()),
        "row_count": int(len(all_labels)),
        "roc_auc": auc,
    }


def _classifier(X: np.ndarray, y: np.ndarray, threshold: float, random_state: int) -> RandomForestClassifier | None:
    labels = (y > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return None
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, labels)
    return model


def train_baseline(*, input_path: Path, output_dir: Path, random_state: int = 42, n_splits: int = 5) -> dict[str, object]:
    X, y, feature_names, materials, families = _load_uci(input_path)
    groups = np.asarray(families)
    unique_groups = sorted(set(families))
    if len(unique_groups) < 2:
        raise ValueError("UCI family GroupKFold requires at least two chemical families")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    n_splits = min(n_splits, len(unique_groups))

    oof = np.full(len(y), np.nan, dtype=float)
    oof_uncertainty = np.full(len(y), np.nan, dtype=float)
    mean_oof = np.full(len(y), np.nan, dtype=float)
    classification_oof = {
        0.0: np.full(len(y), np.nan, dtype=float),
        77.0: np.full(len(y), np.nan, dtype=float),
    }
    folds: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=random_state + fold,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        tree_predictions = np.asarray([tree.predict(X[test_idx]) for tree in model.estimators_])
        prediction = tree_predictions.mean(axis=0)
        oof[test_idx] = prediction
        oof_uncertainty[test_idx] = tree_predictions.std(axis=0)
        mean_oof[test_idx] = float(y[train_idx].mean())
        classification: dict[str, object] = {}
        classification_mean: dict[str, object] = {}
        for threshold in classification_oof:
            classifier = _classifier(X[train_idx], y[train_idx], threshold, random_state + fold)
            scores = np.full(len(test_idx), np.nan, dtype=float)
            if classifier is not None:
                scores = classifier.predict_proba(X[test_idx])[:, 1]
                classification_oof[threshold][test_idx] = scores
            name = "superconducts_above_0k" if threshold == 0.0 else "high_tc_above_77k"
            classification[name] = _classification_metrics(y[test_idx], scores, threshold)
            classification_mean[name] = _classification_metrics(y[test_idx], np.full(len(test_idx), 0.5), threshold)
        folds.append({
            "fold": fold,
            "held_out_families": sorted(set(groups[test_idx])),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "metrics": _metrics(y[test_idx], prediction),
            "mean_predictor_metrics": _metrics(y[test_idx], np.full(len(test_idx), y[train_idx].mean())),
            "classification": classification,
            "classification_mean_predictor": classification_mean,
        })

    final_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X, y)
    tree_predictions = np.asarray([tree.predict(X) for tree in final_model.estimators_])
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline-model.joblib"
    joblib.dump({
        "model": final_model,
        "feature_names": feature_names,
        "group_column": "chemical_family",
        "dataset_sha256": digest,
        "random_state": random_state,
    }, model_path)
    predictions_path = output_dir / "baseline-predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material", "family", "actual_tc_k", "predicted_tc_k", "uncertainty_k", "split"))
        writer.writerows(
            zip(materials, families, y, oof, oof_uncertainty, ("family_grouped_oof",) * len(y))
        )

    classification_report = {
        "superconducts_above_0k": {
            "model": _classification_metrics(y, classification_oof[0.0], 0.0),
            "mean_predictor": _classification_metrics(y, np.full(len(y), 0.5), 0.0),
        },
        "high_tc_above_77k": {
            "model": _classification_metrics(y, classification_oof[77.0], 77.0),
            "mean_predictor": _classification_metrics(y, np.full(len(y), 0.5), 77.0),
        },
    }
    model_metrics = _metrics(y, oof)
    mean_metrics = _metrics(y, mean_oof)
    metrics: dict[str, object] = {
        "model": "RandomForestRegressor",
        "target": "critical_temp",
        "features": feature_names,
        "dataset": str(input_path),
        "dataset_sha256": digest,
        "rows": int(len(y)),
        "feature_count": int(X.shape[1]),
        "family_count": int(len(unique_groups)),
        "family_counts": {family: families.count(family) for family in unique_groups},
        "train_rows_per_fold": [fold["train_rows"] for fold in folds],
        "test_rows_per_fold": [fold["test_rows"] for fold in folds],
        "random_state": random_state,
        "split": "GroupKFold by chemical_family",
        "cv_splits": n_splits,
        "folds": folds,
        "model_metrics": model_metrics,
        "mean_predictor_metrics": mean_metrics,
        "classification": classification_report,
        "comparison": {
            "mae_delta_model_minus_mean": model_metrics["mae_k"] - mean_metrics["mae_k"],
            "rmse_delta_model_minus_mean": model_metrics["rmse_k"] - mean_metrics["rmse_k"],
        },
        "mean_tree_uncertainty_k": float(tree_predictions.std(axis=0).mean()),
        "output_model": str(model_path),
        "output_predictions": str(predictions_path),
        "interpretation": (
            "Family-held-out metrics are the headline. The UCI family labels are deterministic composition heuristics, "
            "not labels supplied by UCI; this baseline is not a pressure-aware predictor."
        ),
    }
    (output_dir / "baseline-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/uci/extracted/unique_m.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/baseline"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(train_baseline(
        input_path=args.input,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_splits=args.n_splits,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
