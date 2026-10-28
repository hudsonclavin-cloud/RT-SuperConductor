#!/usr/bin/env python3
"""Fit an exploratory pressure-aware model from confirmed reviewed records.

This model is a prototype only. Small, correlated literature sets are fit for
pipeline development, but this script deliberately withholds generalization
metrics and promotion until the dataset has adequate independent diversity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from rt_superconductor.reviewed import load_reviewed_records


ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
MIN_RECORDS_FOR_EVALUATION = 50
MIN_FAMILIES_FOR_EVALUATION = 10


def composition_fractions(formula: str) -> dict[str, float]:
    text = "".join(formula.split()).replace("−", "-")
    # Common variable oxygen stoichiometries are represented by an oxygen
    # count of one here; this is a coarse prototype descriptor, not exact
    # stoichiometry. Retain the original formula in the source record.
    text = re.sub(r"O(?:[0-9.]*)?(?:-?δ|[xy])$", "O", text)
    amounts: dict[str, float] = {}
    position = 0
    for match in TOKEN.finditer(text):
        if match.start() != position or match.group(1) not in ELEMENTS:
            raise ValueError(f"formula is not supported by simple composition featurizer: {formula!r}")
        amount = float(match.group(2) or 1.0)
        amounts[match.group(1)] = amounts.get(match.group(1), 0.0) + amount
        position = match.end()
    if position != len(text) or not amounts:
        raise ValueError(f"formula is not supported by simple composition featurizer: {formula!r}")
    total = sum(amounts.values())
    return {element: amount / total for element, amount in amounts.items()}


def _features(records):
    categories = sorted({record.measurement_method.casefold() for record in records})
    elements = sorted({element for record in records for element in composition_fractions(record.formula)})
    names = [f"fraction_{element}" for element in elements]
    names.extend(("pressure_gpa", "pressure_unknown"))
    names.extend(f"method:{category}" for category in categories)
    rows = []
    for record in records:
        fractions = composition_fractions(record.formula)
        pressure_unknown = record.pressure_gpa is None
        row = [fractions.get(element, 0.0) for element in elements]
        row.extend((record.pressure_gpa if record.pressure_gpa is not None else 0.0, float(pressure_unknown)))
        row.extend(float(record.measurement_method.casefold() == category) for category in categories)
        rows.append(row)
    return np.asarray(rows, dtype=float), names


def train_pressure_aware(*, input_dir: Path, output_dir: Path, random_state: int = 42) -> dict[str, object]:
    records = load_reviewed_records(input_dir)
    if not records:
        raise ValueError(f"no confirmed reviewed records found in {input_dir}")
    X, feature_names = _features(records)
    y = np.asarray([record.tc_kelvin for record in records], dtype=float)
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    predictions = model.predict(X)
    composition_families = {tuple(sorted(composition_fractions(record.formula))) for record in records}
    family_count = len(composition_families)
    doi_counts: dict[str, int] = {}
    for record in records:
        doi_counts[record.source_doi] = doi_counts.get(record.source_doi, 0) + 1
    pressure_values = sorted({record.pressure_gpa for record in records if record.pressure_gpa is not None})
    target_window_count = sum(273.15 <= record.tc_kelvin <= 298.15 for record in records)
    evaluation_ready = len(records) >= MIN_RECORDS_FOR_EVALUATION and family_count >= MIN_FAMILIES_FOR_EVALUATION
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pressure-aware-prototype.joblib"
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    source_text = "\n".join(
        f"{record.record_id}|{record.source_doi}|{record.tc_kelvin}|{record.pressure_gpa}|{record.formula}"
        for record in records
    )
    report = {
        "model": "RandomForestRegressor",
        "model_status": "prototype_only",
        "target": "reported critical temperature in kelvin",
        "features": feature_names,
        "feature_count": len(feature_names),
        "record_count": len(records),
        "independent_composition_family_count": family_count,
        "source_doi_count": len({record.source_doi for record in records}),
        "records_per_source_doi": dict(sorted(doi_counts.items())),
        "duplicate_source_record_count": sum(count - 1 for count in doi_counts.values() if count > 1),
        "pressure_known_count": sum(record.pressure_gpa is not None for record in records),
        "pressure_unknown_count": sum(record.pressure_gpa is None for record in records),
        "distinct_known_pressure_levels_gpa": pressure_values,
        "target_window_record_count_273_15_to_298_15_k": target_window_count,
        "data_gaps": [
            gap
            for gap, missing in (
                ("at least 50 confirmed records", len(records) < MIN_RECORDS_FOR_EVALUATION),
                ("at least 10 composition families", family_count < MIN_FAMILIES_FOR_EVALUATION),
                ("records spanning multiple pressure levels", len(pressure_values) < 2),
                ("records in the 273.15–298.15 K target window", target_window_count == 0),
            )
            if missing
        ],
        "training_data_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "random_state": random_state,
        "evaluation_status": "ready" if evaluation_ready else "blocked_insufficient_independent_data",
        "evaluation_minimums": {
            "records": MIN_RECORDS_FOR_EVALUATION,
            "independent_composition_families": MIN_FAMILIES_FOR_EVALUATION,
        },
        "training_fit_mae_k_diagnostic_only": float(np.mean(np.abs(predictions - y))),
        "promotion_status": "not_eligible",
        "output_model": str(model_path),
        "record_ids": [record.record_id for record in records],
        "interpretation": (
            "Exploratory model fit to confirmed reviewed records. In-sample fit is not a validation score; "
            "the model must not be used to claim discovery or generalization."
        ),
    }
    (output_dir / "pressure-aware-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/reviewed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pressure-aware"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    report = train_pressure_aware(input_dir=args.input_dir, output_dir=args.output_dir, random_state=args.random_state)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
