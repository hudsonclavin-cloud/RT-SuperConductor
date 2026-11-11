#!/usr/bin/env python3
"""Does cuprate-specific knowledge improve Tc prediction on unseen cuprate families?

Compares, on the same held-out folds:

* ``formula``                 a forest on the project's formula features (the current approach)
* ``cuprate_only``            a forest on the cuprate features alone
* ``formula_plus_cuprate``    a forest on both
* ``two_part_learned``        Tc = Tc_max(family) x share(doping): one forest learns each
                              family's best Tc from the structure features, a second learns
                              what share of it a given doping reaches
* ``two_part_with_formula``   the same, with the formula features added to the Tc_max forest
* ``two_part_presland``       Tc = Tc_max(family) x the fixed Presland doping dome
* ``median``                  always predict the training median

Folds hold out whole families, so La2-xSrxCuO4 at ten dopings can never sit on
both sides of a split. Two definitions of "family" are scored:

* ``parent``  rows that share a Materials Project parent structure (the main result);
* ``strict``  rows linked by a shared parent structure or a shared element set, taken
              transitively. This is the harshest test, but the links chain together and
              one group ends up holding almost half the rows, so its folds are uneven.

Two questions are scored separately, because the second is the one that matters
for finding better materials:

* row level: how close is each predicted Tc (MAE, Spearman)?
* family level: does the model rank unseen families by their best Tc?
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from cuprate_features import FEATURE_NAMES  # noqa: E402

DOPING_FEATURES = ["cu_valence_formula", "hole_doping_p", "abs_doping", "doping_distance_from_optimum",
                   "presland_dome", "electron_doped", "valence_ambiguous", "fractional_species_count",
                   "largest_minor_fraction", "dopant_fraction_total"]
STRUCTURE_FEATURES = [name for name in FEATURE_NAMES if name not in DOPING_FEATURES]
MODELS = ["median", "formula", "cuprate_only", "formula_plus_cuprate", "two_part_learned",
          "two_part_with_formula", "two_part_presland"]
GROUPINGS = ("parent", "strict")
REFERENCE = "formula"


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    def number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
    return np.asarray([[number(row.get(name, "")) for name in names] for row in rows], dtype=float)


def family_groups(rows: list[dict[str, str]], grouping: str = "parent") -> np.ndarray:
    """Family label per row: shared parent structure, or (strict) also shared element set."""
    from pymatgen.core import Composition

    if grouping == "parent":
        labels = {key: index for index, key in enumerate(sorted({row["mp_parent_id"] for row in rows}))}
        return np.asarray([labels[row["mp_parent_id"]] for row in rows])
    if grouping != "strict":
        raise ValueError(f"grouping must be one of {GROUPINGS}")

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = []
    for row in rows:
        structure_key = "mp:" + row["mp_parent_id"]
        element_key = "el:" + "-".join(sorted(Composition(row["formula"]).get_el_amt_dict()))
        parent[find(structure_key)] = find(element_key)
        keys.append(structure_key)
    labels = {root: index for index, root in enumerate(sorted({find(key) for key in keys}))}
    return np.asarray([labels[find(key)] for key in keys])


def formula_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    from score_pool_physics import formula_feature_rows

    features = formula_feature_rows([row["formula"] for row in rows])
    names = [key for key in features[0] if key.startswith(("magpie__", "shell__", "lewis_"))]
    return to_matrix(features, names), names


def _forest(seed: int):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1))


def predict_two_part(x_structure: np.ndarray, x_structure_formula: np.ndarray, x_share: np.ndarray,
                     dome: np.ndarray, tc: np.ndarray, groups: np.ndarray, train: np.ndarray,
                     test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Fit Tc_max per family and the doping share on the training families only."""
    family_max = {g: tc[train][groups[train] == g].max() for g in np.unique(groups[train])}
    target_max = np.asarray([family_max[g] for g in groups[train]])
    # Each family counts once in the Tc_max model, however many dopings it has.
    sizes = {g: int((groups[train] == g).sum()) for g in family_max}
    weights = np.asarray([1.0 / sizes[g] for g in groups[train]])
    max_model = _forest(seed)
    max_model.fit(x_structure[train], target_max, randomforestregressor__sample_weight=weights)
    predicted_max = np.clip(max_model.predict(x_structure[test]), 0.0, None)
    max_model_formula = _forest(seed + 2)
    max_model_formula.fit(x_structure_formula[train], target_max, randomforestregressor__sample_weight=weights)
    predicted_max_formula = np.clip(max_model_formula.predict(x_structure_formula[test]), 0.0, None)
    # The share model learns only from families that superconduct at some doping.
    usable = target_max > 0
    share_model = _forest(seed + 1)
    share_model.fit(x_share[train][usable], tc[train][usable] / target_max[usable])
    share = np.clip(share_model.predict(x_share[test]), 0.0, 1.0)
    fixed = np.where(np.isnan(dome[test]), np.nanmean(dome[train]), dome[test])
    return {"two_part_learned": predicted_max * share, "two_part_with_formula": predicted_max_formula * share,
            "two_part_presland": predicted_max * fixed}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation, or None when it is undefined (fewer than 3 points or a constant side)."""
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def evaluate(rows: list[dict[str, str]], *, grouping: str = "parent", repeats: int = 3, splits: int = 5,
             seed: int = 2026, formula_features: tuple[np.ndarray, list[str]] | None = None) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    tc = np.asarray([float(row["tc_k"]) for row in rows])
    groups = family_groups(rows, grouping)
    x_formula, formula_names = formula_features or formula_matrix(rows)
    x_cuprate = to_matrix(rows, FEATURE_NAMES)
    x_structure = to_matrix(rows, STRUCTURE_FEATURES)
    x_structure_formula = np.hstack([x_structure, x_formula])
    x_share = to_matrix(rows, DOPING_FEATURES)
    dome = to_matrix(rows, ["presland_dome"])[:, 0]
    matrices = {"formula": x_formula, "cuprate_only": x_cuprate,
                "formula_plus_cuprate": np.hstack([x_formula, x_cuprate])}
    fold_scores: list[dict[str, Any]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(x_formula, tc > 0, groups)):
            predictions = {"median": np.full(len(test), float(np.median(tc[train])))}
            for name, matrix in matrices.items():
                model = _forest(seed + 100 * repeat + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(matrix[train], tc[train])
                    predictions[name] = model.predict(matrix[test])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions.update(predict_two_part(x_structure, x_structure_formula, x_share, dome, tc, groups,
                                                    train, test, seed + 100 * repeat + fold))
            # Family level: best true Tc of each held-out family against the model's best prediction for it.
            families = np.unique(groups[test])
            true_max = np.asarray([tc[test][groups[test] == g].max() for g in families])
            record: dict[str, Any] = {"repeat": repeat, "fold": fold, "test_rows": int(len(test)),
                                      "test_families": int(len(families))}
            for name in MODELS:
                values = predictions[name]
                positive = tc[test] > 0
                family_pred = np.asarray([values[groups[test] == g].max() for g in families])
                record[name] = {
                    "mae_k": float(np.mean(np.abs(values - tc[test]))),
                    "mae_superconducting_rows_k": float(np.mean(np.abs(values[positive] - tc[test][positive]))),
                    "spearman": _spearman(values, tc[test]),
                    "family_max_mae_k": float(np.mean(np.abs(family_pred - true_max))),
                    "family_max_spearman": _spearman(family_pred, true_max),
                }
            fold_scores.append(record)

    def summary(name: str, metric: str) -> dict[str, float] | None:
        values = [record[name][metric] for record in fold_scores if record[name][metric] is not None]
        if not values:
            return None
        return {"mean": float(np.mean(values)), "sd_across_folds": float(np.std(values)), "folds_scored": len(values)}

    metrics = ["mae_k", "mae_superconducting_rows_k", "spearman", "family_max_mae_k", "family_max_spearman"]
    results = {name: {metric: summary(name, metric) for metric in metrics} for name in MODELS}
    paired = {}
    for name in MODELS:
        if name in (REFERENCE, "median"):
            continue
        paired[name] = {}
        for metric in metrics:
            usable = [record for record in fold_scores
                      if record[name][metric] is not None and record[REFERENCE][metric] is not None]
            differences = np.asarray([record[name][metric] - record[REFERENCE][metric] for record in usable])
            by_repeat = [float(np.mean([record[name][metric] - record[REFERENCE][metric]
                                        for record in usable if record["repeat"] == r]))
                         for r in range(repeats) if any(record["repeat"] == r for record in usable)]
            lower_is_better = metric.endswith("_k")
            paired[name][metric] = {
                "mean_difference_vs_formula": float(differences.mean()),
                "better_in_folds": int(((differences < 0) if lower_is_better else (differences > 0)).sum()),
                "folds": int(len(differences)),
                "better_in_repeats": int(sum((d < 0) if lower_is_better else (d > 0) for d in by_repeat)),
                "repeats": len(by_repeat),
            }
    return {
        "rows": len(rows), "superconducting_rows": int((tc > 0).sum()), "families": int(len(np.unique(groups))),
        "largest_family_rows": int(np.bincount(groups).max()),
        "grouping": grouping,
        "split": f"{splits}-fold StratifiedGroupKFold by family ({grouping}), {repeats} shuffles",
        "formula_feature_count": len(formula_names),
        "results": results, "paired_against_formula": paired, "fold_scores": fold_scores,
        "seed": seed,
    }


def importances(rows: list[dict[str, str]], seed: int = 2026) -> list[dict[str, Any]]:
    """Which cuprate features the cuprate-only forest leans on (in-sample, so read as a description)."""
    from sklearn.inspection import permutation_importance

    tc = np.asarray([float(row["tc_k"]) for row in rows])
    x = to_matrix(rows, FEATURE_NAMES)
    model = _forest(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, tc)
        result = permutation_importance(model, x, tc, n_repeats=10, random_state=seed,
                                        scoring="neg_mean_absolute_error")
    ranked = sorted(zip(FEATURE_NAMES, result.importances_mean), key=lambda item: -item[1])
    return [{"feature": name, "mae_increase_when_shuffled_k": round(float(value), 3)} for name, value in ranked]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=ROOT / "data/training/cuprate-features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/cuprate-model.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = read_table(args.features)
    formula_features = formula_matrix(rows)
    report = {
        "question": "Do cuprate-specific features, or a Tc_max x doping-share model, predict Tc better than "
                    "formula features on cuprate families the model has not seen?",
        "main_grouping": "parent",
        "cuprate_features": FEATURE_NAMES, "structure_features": STRUCTURE_FEATURES,
        "doping_features": DOPING_FEATURES, "models": MODELS, "reference_model": REFERENCE,
        "by_grouping": {grouping: evaluate(rows, grouping=grouping, repeats=args.repeats,
                                           formula_features=formula_features) for grouping in GROUPINGS},
        "cuprate_feature_importance": importances(rows),
        "cuprate_feature_importance_note": "Permutation importance of the cuprate-only forest fitted on every row. "
                                           "It is in-sample: it describes what the model uses, not what is true.",
    }
    report["features_path"] = str(args.features.relative_to(ROOT)) if args.features.is_relative_to(ROOT) else str(args.features)
    report["features_sha256"] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for grouping, part in report["by_grouping"].items():
        print(f"{grouping}: {part['rows']} rows, {part['families']} families, largest {part['largest_family_rows']}")
        for name, scores in part["results"].items():
            print("  %-24s" % name + "  ".join(
                f"{metric}={scores[metric]['mean']:.3f}" for metric in scores if scores[metric]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
