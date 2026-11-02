#!/usr/bin/env python3
"""Train and evaluate a pressure-aware Tc surrogate on the harmonized table.

The primary evaluation is GroupKFold by chemical family. Records from a family
held out of a fold are never used to fit that fold's model. Matminer's Magpie
elemental statistics are used when the ML extras are installed; a transparent
composition-fraction fallback keeps smoke-test validation usable in a minimal
checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.formula import ELEMENTS
from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized


def _fraction_features(records: list[HarmonizedRecord]) -> tuple[np.ndarray, list[str], str]:
    """Small dependency-free fallback used only when Matminer is unavailable."""

    import re

    token = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
    # Keep the fallback's dimensionality stable between the confirmed training
    # subset and the full candidate table (which may contain new elements).
    elements = sorted(ELEMENTS)
    rows: list[list[float]] = []
    for record in records:
        amounts = {element: 0.0 for element in elements}
        for element, count in token.findall(record.formula):
            if element not in ELEMENTS:
                raise ValueError(f"unsupported element {element!r} in {record.formula!r}")
            amounts[element] += float(count or 1.0)
        total = sum(amounts.values())
        rows.append([amounts[element] / total for element in elements])
    return np.asarray(rows, dtype=float), [f"fraction_{element}" for element in elements], "fraction_fallback"


def _composition_features(records: list[HarmonizedRecord]) -> tuple[np.ndarray, list[str], str]:
    try:
        import pandas as pd
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition

        frame = pd.DataFrame({"composition": [Composition(record.formula) for record in records]})
        featurizer = ElementProperty.from_preset("magpie")
        frame = featurizer.featurize_dataframe(frame, col_id="composition", ignore_errors=False)
        feature_names = [name for name in frame.columns if name != "composition"]
        return frame[feature_names].to_numpy(dtype=float), feature_names, "matminer.magpie"
    except (ImportError, ModuleNotFoundError):
        return _fraction_features(records)


def build_features(
    records: list[HarmonizedRecord], *, hull_fill: float | None = None
) -> tuple[np.ndarray, list[str], str]:
    composition, names, backend = _composition_features(records)
    known_hulls = [record.energy_above_hull for record in records if record.energy_above_hull is not None]
    if hull_fill is None:
        hull_fill = float(np.median(known_hulls)) if known_hulls else 0.0
    pressure = np.asarray([record.applied_pressure_gpa for record in records], dtype=float).reshape(-1, 1)
    hull = np.asarray([
        record.energy_above_hull if record.energy_above_hull is not None else hull_fill
        for record in records
    ], dtype=float).reshape(-1, 1)
    hull_missing = np.asarray([
        record.energy_above_hull is None for record in records
    ], dtype=float).reshape(-1, 1)
    features = np.hstack((composition, pressure, hull, hull_missing))
    return features, names + ["applied_pressure_gpa", "energy_above_hull", "energy_above_hull_missing"], backend


def _model(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    result = {
        "mae_k": float(mean_absolute_error(actual, predicted)),
        "rmse_k": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
    if len(actual) >= 2:
        r2 = float(r2_score(actual, predicted))
        result["r2"] = r2 if math.isfinite(r2) else None
    else:
        result["r2"] = None
    return result


def _classification_metrics(actual: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    all_labels = (actual > threshold).astype(int)
    labels = all_labels
    finite = np.isfinite(scores)
    usable_labels = labels[finite]
    usable_scores = scores[finite]
    auc = None
    if len(usable_labels) and len(np.unique(usable_labels)) == 2:
        auc = float(roc_auc_score(usable_labels, usable_scores))
    return {
        "threshold_k": threshold,
        "positive_count": int(all_labels.sum()),
        "row_count": int(len(all_labels)),
        "roc_auc": auc,
    }


def _fit_classifier(X: np.ndarray, y: np.ndarray, threshold: float, random_state: int) -> RandomForestClassifier | None:
    labels = (y > threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return None
    classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(X, labels)
    return classifier


def _write_predictions(
    path: Path,
    records: Iterable[HarmonizedRecord],
    predictions: Iterable[float],
    *,
    split: str,
    uncertainties: Iterable[float] | None = None,
) -> None:
    uncertainties = iter(uncertainties) if uncertainties is not None else None
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "record_id", "formula", "chemical_family", "tc_kelvin", "predicted_tc_kelvin",
            "uncertainty_k", "applied_pressure_gpa", "energy_above_hull", "review_status",
            "provenance", "is_synthetic", "split",
        ))
        for record, prediction in zip(records, predictions):
            uncertainty = next(uncertainties) if uncertainties is not None else None
            writer.writerow((
                record.record_id, record.formula, record.chemical_family, record.tc_kelvin,
                f"{float(prediction):.8g}", "" if uncertainty is None else f"{float(uncertainty):.8g}",
                record.applied_pressure_gpa,
                "" if record.energy_above_hull is None else record.energy_above_hull,
                record.review_status, record.provenance, record.is_synthetic, split,
            ))


def _write_mixed_predictions(path: Path, rows: Iterable[tuple[HarmonizedRecord, float, str]]) -> None:
    """Write honest OOF predictions plus negative-control predictions."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "record_id", "formula", "chemical_family", "tc_kelvin", "predicted_tc_kelvin",
            "uncertainty_k", "applied_pressure_gpa", "energy_above_hull", "review_status",
            "provenance", "is_synthetic", "split",
        ))
        for record, prediction, split, uncertainty in rows:
            writer.writerow((
                record.record_id, record.formula, record.chemical_family, record.tc_kelvin,
                f"{float(prediction):.8g}", "" if uncertainty is None else f"{float(uncertainty):.8g}",
                record.applied_pressure_gpa,
                "" if record.energy_above_hull is None else record.energy_above_hull,
                record.review_status, record.provenance, record.is_synthetic, split,
            ))


TRAINABLE_REVIEW_STATUSES = {"confirmed", "curated_database"}


def select_training_records(records: list[HarmonizedRecord]) -> list[HarmonizedRecord]:
    """Return only empirical database records eligible for model fitting.

    Automatic SuperCon2 extraction is never a training source, even if a row
    is later marked confirmed. The five human-reviewed literature rows are
    retained as ``provenance=literature`` and may enter through ``confirmed``;
    curated database rows enter through the explicit ``curated_database`` tier.
    """

    return [
        record for record in records
        if record.review_status in TRAINABLE_REVIEW_STATUSES
        and record.is_synthetic == 0
        and not (record.source_db == "supercon2" and record.provenance != "literature")
    ]


def _blocked_report(
    *,
    input_path: Path,
    output_dir: Path,
    records: list[HarmonizedRecord],
    reason: str,
) -> dict[str, object]:
    eligible_statuses = TRAINABLE_REVIEW_STATUSES
    confirmed = [record for record in records if record.review_status in eligible_statuses]
    training_records = select_training_records(records)
    report: dict[str, object] = {
        "training_status": "blocked_insufficient_empirical_data",
        "blocked_reason": reason,
        "dataset": str(input_path),
        "record_count": len(records),
        "confirmed_record_count": len(confirmed),
        "synthetic_record_count": sum(record.is_synthetic for record in records),
        "confirmed_synthetic_excluded_count": sum(record.review_status in eligible_statuses and record.is_synthetic == 1 for record in records),
        "supercon2_automatic_excluded_count": sum(record.source_db == "supercon2" and record.provenance != "literature" for record in records),
        "empirical_training_record_count": len(training_records),
        "empirical_training_record_ids": [record.record_id for record in training_records],
        "chemical_family_count": len({record.chemical_family for record in training_records}),
        "training_filter": "review_status in {'confirmed', 'curated_database'} and is_synthetic == 0 and not (source_db == 'supercon2' and provenance != 'literature')",
        "promotion_status": "not_eligible",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pressure-aware-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def train_pressure_aware(
    *,
    input_path: Path = Path("data/training/harmonized_supercon.csv"),
    output_dir: Path = Path("data/processed/pressure-aware"),
    random_state: int = 42,
    n_splits: int = 5,
    allow_insufficient_data: bool = False,
) -> dict[str, object]:
    records = load_harmonized(input_path)
    confirmed = [record for record in records if record.review_status in TRAINABLE_REVIEW_STATUSES]
    training_records = select_training_records(records)
    if len(training_records) < 2:
        reason = "at least two eligible empirical records are required for training"
        if allow_insufficient_data:
            return _blocked_report(input_path=input_path, output_dir=output_dir, records=records, reason=reason)
        raise ValueError(reason)
    groups = np.asarray([record.chemical_family for record in training_records])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        reason = "GroupKFold requires at least two empirical chemical_family groups"
        if allow_insufficient_data:
            return _blocked_report(input_path=input_path, output_dir=output_dir, records=records, reason=reason)
        raise ValueError(reason)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    n_splits = min(n_splits, len(unique_groups))
    known_hulls = [record.energy_above_hull for record in training_records if record.energy_above_hull is not None]
    hull_fill = float(np.median(known_hulls)) if known_hulls else 0.0
    X, feature_names, feature_backend = build_features(training_records, hull_fill=hull_fill)
    y = np.asarray([record.tc_kelvin for record in training_records], dtype=float)

    oof = np.full(len(training_records), np.nan, dtype=float)
    oof_uncertainty = np.full(len(training_records), np.nan, dtype=float)
    mean_oof = np.full(len(training_records), np.nan, dtype=float)
    classification_oof = {
        0.0: np.full(len(training_records), np.nan, dtype=float),
        77.0: np.full(len(training_records), np.nan, dtype=float),
    }
    fold_results: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = _model(random_state + fold)
        model.fit(X[train_idx], y[train_idx])
        tree_predictions = np.asarray([tree.predict(X[test_idx]) for tree in model.estimators_])
        prediction = tree_predictions.mean(axis=0)
        uncertainty = tree_predictions.std(axis=0)
        oof[test_idx] = prediction
        oof_uncertainty[test_idx] = uncertainty
        mean_oof[test_idx] = float(y[train_idx].mean())
        classification_fold: dict[str, object] = {}
        classification_baseline_fold: dict[str, object] = {}
        for threshold in classification_oof:
            classifier = _fit_classifier(X[train_idx], y[train_idx], threshold, random_state + fold)
            scores = np.full(len(test_idx), np.nan, dtype=float)
            if classifier is not None:
                scores = classifier.predict_proba(X[test_idx])[:, 1]
                classification_oof[threshold][test_idx] = scores
            name = "superconducts_above_0k" if threshold == 0.0 else "high_tc_above_77k"
            classification_fold[name] = _classification_metrics(y[test_idx], scores, threshold)
            classification_baseline_fold[name] = _classification_metrics(
                y[test_idx], np.full(len(test_idx), 0.5), threshold
            )
        fold_results.append({
            "fold": fold,
            "held_out_families": sorted(set(groups[test_idx])),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "metrics": _metrics(y[test_idx], prediction),
            "mean_predictor_metrics": _metrics(y[test_idx], np.full(len(test_idx), y[train_idx].mean())),
            "classification": classification_fold,
            "classification_mean_predictor": classification_baseline_fold,
        })

    model = _model(random_state)
    model.fit(X, y)
    all_X, _, _ = build_features(records, hull_fill=hull_fill)
    all_tree_predictions = np.asarray([tree.predict(all_X) for tree in model.estimators_])
    all_predictions = all_tree_predictions.mean(axis=0)
    all_uncertainties = all_tree_predictions.std(axis=0)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pressure-aware-model.joblib"
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "feature_backend": feature_backend,
        "target": "tc_kelvin",
        "group_column": "chemical_family",
        "random_state": random_state,
        "dataset_sha256": digest,
        "hull_fill": hull_fill,
    }, model_path)
    oof_path = output_dir / "grouped-oof-predictions.csv"
    _write_predictions(oof_path, training_records, oof, split="grouped_oof", uncertainties=oof_uncertainty)
    prediction_path = output_dir / "candidate_predictions.csv"
    _write_predictions(prediction_path, records, all_predictions, split="full_fit", uncertainties=all_uncertainties)
    promotion_path = output_dir / "promotion_predictions.csv"
    confirmed_oof = {record.record_id: float(prediction) for record, prediction in zip(training_records, oof)}
    confirmed_uncertainty = {record.record_id: float(value) for record, value in zip(training_records, oof_uncertainty)}
    mixed_rows = [
        (record, confirmed_oof[record.record_id], "grouped_oof", confirmed_uncertainty[record.record_id])
        if record.record_id in confirmed_oof
        else (record, float(all_predictions[index]), "negative_control_full_fit", float(all_uncertainties[index]))
        for index, record in enumerate(records)
    ]
    _write_mixed_predictions(promotion_path, mixed_rows)
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
    report: dict[str, object] = {
        "model": "RandomForestRegressor",
        "target": "tc_kelvin",
        "features": feature_names,
        "feature_backend": feature_backend,
        "pressure_feature": "applied_pressure_gpa",
        "stability_feature": "energy_above_hull",
        "dataset": str(input_path),
        "dataset_sha256": digest,
        "record_count": len(records),
        "confirmed_record_count": len(confirmed),
        "synthetic_record_count": sum(record.is_synthetic for record in records),
        "confirmed_synthetic_excluded_count": sum(record.review_status in TRAINABLE_REVIEW_STATUSES and record.is_synthetic == 1 for record in records),
        "supercon2_automatic_excluded_count": sum(record.source_db == "supercon2" and record.provenance != "literature" for record in records),
        "empirical_training_record_count": len(training_records),
        "empirical_training_record_ids": [record.record_id for record in training_records],
        "chemical_family_count": len(unique_groups),
        "group_column": "chemical_family",
        "cv": "GroupKFold",
        "cv_splits": n_splits,
        "folds": fold_results,
        "grouped_cv_metrics": _metrics(y, oof),
        "mean_predictor_metrics": _metrics(y, mean_oof),
        "classification": classification_report,
        "comparison": {
            "mae_delta_model_minus_mean": _metrics(y, oof)["mae_k"] - _metrics(y, mean_oof)["mae_k"],
            "rmse_delta_model_minus_mean": _metrics(y, oof)["rmse_k"] - _metrics(y, mean_oof)["rmse_k"],
        },
        "output_model": str(model_path),
        "output_oof_predictions": str(oof_path),
        "output_candidate_predictions": str(prediction_path),
        "output_promotion_predictions": str(promotion_path),
        "random_state": random_state,
        "training_filter": "review_status in {'confirmed', 'curated_database'} and is_synthetic == 0 and not (source_db == 'supercon2' and provenance != 'literature')",
        "training_status": "ready",
        "promotion_status": "not_promoted",
        "interpretation": (
            "OOF metrics are family-held-out estimates. The full-fit artifact is a screening surrogate; "
            "it is not a substitute for experiment or first-principles validation."
        ),
    }
    (output_dir / "pressure-aware-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--input-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pressure-aware"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--allow-insufficient-data",
        action="store_true",
        help="write a blocked report and exit successfully when empirical GroupKFold data is insufficient",
    )
    args = parser.parse_args()
    if args.input_dir is not None:
        parser.error("--input-dir is obsolete; use --input data/training/harmonized_supercon.csv")
    report = train_pressure_aware(
        input_path=args.input_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_splits=args.n_splits,
        allow_insufficient_data=args.allow_insufficient_data,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
