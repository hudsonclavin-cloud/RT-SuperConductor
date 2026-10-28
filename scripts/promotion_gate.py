#!/usr/bin/env python3
"""Evaluate the strict surrogate-model promotion gate.

The gate is intentionally fail-closed: missing benchmark rows, missing
excluded references, or missing model artifacts do not count as passing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

AMBIENT_PRESSURE_MAX_GPA = 0.0001
AMBIENT_MAE_MAX_K = 12.0
HYDRIDE_TREND_MIN_PEARSON_R = 0.80
FALSE_POSITIVE_MAX_RATE = 0.05
PREDICTED_TC_FILTER_K = 77.0
BENCHMARK_FORMULAS = {
    "H3S": {"H3S"},
    # Accept both the conventional LaH10 spelling and the harmonizer's
    # alphabetized H10La spelling.
    "LaH10": {"LaH10", "H10La"},
}


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "formula", "tc_kelvin", "predicted_tc_kelvin", "applied_pressure_gpa", "review_status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: prediction CSV must include {sorted(required)}")
    return rows


def _pearson(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    mean_a = sum(actual) / len(actual)
    mean_p = sum(predicted) / len(predicted)
    numerator = sum((a - mean_a) * (p - mean_p) for a, p in zip(actual, predicted))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in actual) * sum((p - mean_p) ** 2 for p in predicted)
    )
    return None if denominator == 0 else numerator / denominator


def evaluate_gate(*, predictions_path: Path, model_path: Path) -> dict[str, object]:
    rows = _read_predictions(predictions_path)
    ambient = [
        row for row in rows
        if row["review_status"] == "confirmed" and float(row["applied_pressure_gpa"]) <= AMBIENT_PRESSURE_MAX_GPA
    ]
    ambient_mae = (
        sum(abs(float(row["tc_kelvin"]) - float(row["predicted_tc_kelvin"])) for row in ambient) / len(ambient)
        if ambient else None
    )
    trend: dict[str, float | None] = {}
    trend_rows: dict[str, int] = {}
    for benchmark_name, aliases in BENCHMARK_FORMULAS.items():
        benchmark = [
            row for row in rows
            if row["formula"] in aliases
            and float(row["applied_pressure_gpa"]) > AMBIENT_PRESSURE_MAX_GPA
            and row["review_status"] == "confirmed"
        ]
        trend_rows[benchmark_name] = len(benchmark)
        trend[benchmark_name] = _pearson(
            [float(row["tc_kelvin"]) for row in benchmark],
            [float(row["predicted_tc_kelvin"]) for row in benchmark],
        )
    excluded = [row for row in rows if row["review_status"] == "excluded"]
    false_positives = sum(float(row["predicted_tc_kelvin"]) > PREDICTED_TC_FILTER_K for row in excluded)
    false_positive_rate = false_positives / len(excluded) if excluded else None
    checks = {
        "model_artifact_exists": model_path.is_file(),
        "ambient_mae": ambient_mae is not None and ambient_mae <= AMBIENT_MAE_MAX_K,
        "hydride_trend": all(value is not None and value >= HYDRIDE_TREND_MIN_PEARSON_R for value in trend.values()),
        "excluded_false_positive_rate": false_positive_rate is not None and false_positive_rate < FALSE_POSITIVE_MAX_RATE,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "thresholds": {
            "ambient_pressure_max_gpa": AMBIENT_PRESSURE_MAX_GPA,
            "ambient_mae_max_k": AMBIENT_MAE_MAX_K,
            "hydride_trend_min_pearson_r": HYDRIDE_TREND_MIN_PEARSON_R,
            "false_positive_rate_strictly_less_than": FALSE_POSITIVE_MAX_RATE,
            "predicted_tc_positive_filter_k": PREDICTED_TC_FILTER_K,
        },
        "model_artifact": str(model_path),
        "predictions": str(predictions_path),
        "ambient_confirmed_rows": len(ambient),
        "ambient_mae_k": ambient_mae,
        "hydride_trend_pearson_r": trend,
        "hydride_benchmark_rows": trend_rows,
        "excluded_reference_rows": len(excluded),
        "excluded_false_positives": false_positives,
        "excluded_false_positive_rate": false_positive_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/pressure-aware/promotion_predictions.csv"))
    parser.add_argument("--model", type=Path, default=Path("data/processed/pressure-aware/pressure-aware-model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/pressure-aware/promotion-gate.json"))
    args = parser.parse_args()
    report = evaluate_gate(predictions_path=args.predictions, model_path=args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
