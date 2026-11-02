#!/usr/bin/env python3
"""Compare composition-only and CIF-derived Matminer features by family holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import numbers
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized

from train_pressure_aware import build_features, select_training_records


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    value = float(r2_score(actual, prediction)) if len(actual) > 1 else float("nan")
    return {
        "mae_k": float(mean_absolute_error(actual, prediction)),
        "rmse_k": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": value if np.isfinite(value) else None,
    }


def _structure_features(path: Path) -> tuple[list[float], list[str]]:
    from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
    from pymatgen.core import Structure

    structure = Structure.from_file(path)
    if not structure.is_ordered:
        # 3DSC synthetic-doping CIFs can contain partial occupancies. Matminer
        # structure featurizers used here require ordered sites, so use the
        # highest-occupancy species and keep this preprocessing explicit in the
        # report rather than silently dropping the structure.
        species = [max(site.species.items(), key=lambda item: item[1])[0] for site in structure.sites]
        structure = Structure(structure.lattice, species, structure.frac_coords)
    values: list[float] = []
    names: list[str] = []
    for featurizer in (DensityFeatures(), GlobalSymmetryFeatures()):
        for name, value in zip(featurizer.feature_labels(), featurizer.featurize(structure)):
            if not isinstance(value, numbers.Number):
                continue
            values.append(float(value))
            names.append(f"{featurizer.__class__.__name__}:{name}")
    return values, names


def evaluate_structure_features(
    *,
    input_path: Path,
    structure_index: Path,
    output_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    records = load_harmonized(input_path)
    index: dict[str, Path] = {}
    if structure_index.is_file():
        with structure_index.open(newline="", encoding="utf-8-sig") as handle:
            index = {row["record_id"]: Path(row["cif_path"]) for row in csv.DictReader(handle)}
    training = select_training_records(records)
    usable: list[HarmonizedRecord] = []
    structures: list[list[float]] = []
    feature_names: list[str] = []
    failures: dict[str, str] = {}
    for record in training:
        path = index.get(record.record_id)
        if path is None or not path.is_file():
            continue
        try:
            values, names = _structure_features(path)
            if not feature_names:
                feature_names = names
            if len(values) != len(feature_names):
                raise ValueError("structure feature length changed between CIFs")
            usable.append(record)
            structures.append(values)
        except Exception as exc:
            failures[record.record_id] = f"{type(exc).__name__}: {exc}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_report: dict[str, object] = {
        "input": str(input_path),
        "dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "structure_index": str(structure_index),
        "training_record_count": len(training),
        "structure_record_count": len(usable),
        "structure_failures": failures,
        "feature_backend": "matminer DensityFeatures + GlobalSymmetryFeatures; disordered CIF sites use highest-occupancy ordered proxies",
        "gnn": {"status": "not_attempted", "reason": "ALIGNN/M3GNet is an optional second-stage backend and is not enabled in this reproducible baseline."},
    }
    if len(usable) < 2 or len({record.chemical_family for record in usable}) < 2:
        base_report.update({
            "status": "blocked_insufficient_structures",
            "blocked_reason": "at least two CIF-backed empirical chemical families are required for GroupKFold",
            "comparison": "not_available",
        })
        output_path.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return base_report

    X_composition, composition_names, _ = build_features(usable)
    X_structure = SimpleImputer(strategy="median").fit_transform(np.asarray(structures, dtype=float))
    X_combined = np.hstack((X_composition, X_structure))
    y = np.asarray([record.tc_kelvin for record in usable], dtype=float)
    groups = np.asarray([record.chemical_family for record in usable])
    splits = min(n_splits, len(set(groups)))
    if splits < 2:
        raise ValueError("structure comparison requires at least two family groups")
    scores: dict[str, list[np.ndarray]] = {"composition_only": [], "composition_plus_structure": []}
    actuals: list[np.ndarray] = []
    folds: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=splits).split(X_composition, y, groups), start=1):
        fold_report: dict[str, object] = {"fold": fold, "held_out_families": sorted(set(groups[test_idx]))}
        for name, matrix in (("composition_only", X_composition), ("composition_plus_structure", X_combined)):
            model = RandomForestRegressor(n_estimators=200, random_state=random_state + fold, n_jobs=-1, min_samples_leaf=1)
            model.fit(matrix[train_idx], y[train_idx])
            scores[name].append(model.predict(matrix[test_idx]))
            fold_report[name] = _metrics(y[test_idx], scores[name][-1])
        fold_report["mean_predictor"] = _metrics(y[test_idx], np.full(len(test_idx), y[train_idx].mean()))
        actuals.append(y[test_idx])
        folds.append(fold_report)
    actual = np.concatenate(actuals)
    composition_prediction = np.concatenate(scores["composition_only"])
    structure_prediction = np.concatenate(scores["composition_plus_structure"])
    composition_metrics = _metrics(actual, composition_prediction)
    structure_metrics = _metrics(actual, structure_prediction)
    base_report.update({
        "status": "ready",
        "groups": sorted(set(groups)),
        "cv": "GroupKFold by chemical_family",
        "folds": folds,
        "composition_only": {"features": composition_names, "metrics": composition_metrics},
        "composition_plus_structure": {"features": composition_names + feature_names, "metrics": structure_metrics},
        "mean_predictor": _metrics(actual, np.full(len(actual), float(actual.mean()))),
        "comparison": {
            "structure_mae_delta_minus_composition": structure_metrics["mae_k"] - composition_metrics["mae_k"],
            "structure_beats_composition": structure_metrics["mae_k"] < composition_metrics["mae_k"],
        },
    })
    output_path.write_text(json.dumps(base_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return base_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--structure-index", type=Path, default=Path("data/processed/3dsc-structure-index.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/structure-features-report.json"))
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate_structure_features(
        input_path=args.input,
        structure_index=args.structure_index,
        output_path=args.report,
        n_splits=args.n_splits,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
