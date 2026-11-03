#!/usr/bin/env python3
"""Build DFT-spin/CIF geometry features and evaluate 3DSC feature ablations.

All magnetic columns originate in the 3DSC Materials Project export and are
DFT-computed descriptors, not experimental measurements. Structure-derived
features are calculated from the linked CIF with pymatgen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ingest_3dsc import _read_rows, _record_id
from train_pressure_aware import build_features, elemental_system_signature, select_training_records
from rt_superconductor.harmonized import load_harmonized

ORDERINGS = ("NM", "FM", "AFM", "FiM")
SPIN_FEATURES = ["is_magnetic_2", "total_magnetization_2_per_atom", "num_magnetic_sites_2_per_atom"] + [
    f"ordering_2_{value}" for value in ORDERINGS
]
BOND_FEATURES = [
    "nearest_neighbor_min_angstrom", "nearest_neighbor_mean_angstrom",
    "nearest_neighbor_std_angstrom", "metal_nonmetal_mean_distance_angstrom",
    "crystalnn_mean_coordination_number",
]
PATTERN_FEATURES = ["dimensionality_0d", "dimensionality_1d", "dimensionality_2d",
                    "dimensionality_3d", "layers_present"]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _spin_features(row: dict[str, str]) -> dict[str, float | str | None]:
    """Normalize table-provided DFT magnetization/site counts per atom."""
    nsites = _number(row.get("nsites_2"))
    total_mag = _number(row.get("total_magnetization_2"))
    magnetic_sites = _number(row.get("num_magnetic_sites_2"))
    magnetic_raw = str(row.get("is_magnetic_2", "")).strip().lower()
    is_magnetic = 1.0 if magnetic_raw in {"true", "1", "yes"} else (
        0.0 if magnetic_raw in {"false", "0", "no"} else None
    )
    ordering = str(row.get("ordering_2", "")).strip()
    output: dict[str, float | None] = {
        "is_magnetic_2": is_magnetic,
        "total_magnetization_2_per_atom": total_mag / nsites if total_mag is not None and nsites and nsites > 0 else None,
        "num_magnetic_sites_2_per_atom": magnetic_sites / nsites if magnetic_sites is not None and nsites and nsites > 0 else None,
    }
    output.update({f"ordering_2_{value}": float(ordering == value) if ordering in ORDERINGS else None
                   for value in ORDERINGS})
    output["ordering_2"] = ordering or None
    return output


def _ordered_proxy(structure):
    """Return a highest-occupancy proxy when a synthetic CIF is disordered."""
    from pymatgen.core import Structure

    if structure.is_ordered:
        return structure, False
    species = [max(site.species.items(), key=lambda item: item[1])[0] for site in structure.sites]
    return Structure(structure.lattice, species, structure.frac_coords), True


def _cif_features(cif_path: Path) -> dict[str, float | int | bool | None]:
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Element, Structure

    structure, used_disorder_proxy = _ordered_proxy(Structure.from_file(cif_path))
    # One periodic nearest neighbor per site; symmetry-equivalent and periodic
    # neighbors are handled by Structure.get_all_neighbors.
    neighbor_lists = structure.get_all_neighbors(r=12.0, include_index=True)
    nearest = [min(float(n.nn_distance) for n in neighbors if n.nn_distance > 1e-8)
               for neighbors in neighbor_lists if any(n.nn_distance > 1e-8 for n in neighbors)]
    crystal_nn = CrystalNN()
    coordination: list[float] = []
    hetero_bonds: list[float] = []
    for index, site in enumerate(structure):
        coordination.append(float(crystal_nn.get_cn(structure, index)))
        center_element = next(iter(site.species))
        center_is_metal = Element(str(center_element)).is_metal
        for info in crystal_nn.get_nn_info(structure, index):
            neighbor_element = next(iter(info["site"].species))
            if center_is_metal != Element(str(neighbor_element)).is_metal:
                hetero_bonds.append(float(info["site"].distance(site)))
    graph = StructureGraph.from_local_env_strategy(structure, crystal_nn)
    dimension = int(get_dimensionality_larsen(graph))
    dimension = min(3, max(0, dimension))
    output: dict[str, float | int | bool | None] = {
        "nearest_neighbor_min_angstrom": min(nearest) if nearest else None,
        "nearest_neighbor_mean_angstrom": float(np.mean(nearest)) if nearest else None,
        "nearest_neighbor_std_angstrom": float(np.std(nearest)) if nearest else None,
        "metal_nonmetal_mean_distance_angstrom": float(np.mean(hetero_bonds)) if hetero_bonds else None,
        "crystalnn_mean_coordination_number": float(np.mean(coordination)) if coordination else None,
        "dimensionality": dimension,
        "layers_present": dimension == 2,
        "disorder_highest_occupancy_proxy": used_disorder_proxy,
    }
    output.update({f"dimensionality_{level}d": float(dimension == level) for level in range(4)})
    output["layers_present"] = float(dimension == 2)
    return output


def build_feature_rows(
    *, harmonized_path: Path, source_csv: Path, structure_index: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = select_training_records(load_harmonized(harmonized_path))
    source_rows = _read_rows(source_csv)
    # The importer canonicalizes the formula before forming an ID. Reproduce
    # that stable key from the ingested record rather than trusting raw spelling.
    from rt_superconductor.formula import canonicalize_formula
    source_by_id = {}
    for row_number, row in enumerate(source_rows, start=2):
        parsed = canonicalize_formula(row.get("formula_sc", ""))
        try:
            tc = float(row.get("tc", ""))
        except (TypeError, ValueError):
            continue
        if parsed.status == "parsed" and parsed.canonical:
            source_by_id[_record_id(row, row_number, parsed.canonical, tc)] = row
    index: dict[str, Path] = {}
    with structure_index.open(newline="", encoding="utf-8-sig") as handle:
        index = {row["record_id"]: Path(row["cif_path"]) for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    missing: dict[str, int] = {"source_record": 0, "cif": 0}
    for position, record in enumerate(records, start=1):
        source = source_by_id.get(record.record_id)
        cif = index.get(record.record_id)
        if source is None:
            missing["source_record"] += 1
            continue
        if cif is None or not cif.is_file():
            missing["cif"] += 1
            continue
        try:
            row = {"record_id": record.record_id, "formula": record.formula,
                   "chemical_family": record.chemical_family,
                   "spin_feature_provenance": "3DSC_MP Materials Project DFT-computed; not measured"}
            row.update(_spin_features(source))
            row.update(_cif_features(cif))
            row["cif_path"] = str(cif)
            rows.append(row)
        except Exception as exc:
            failures[record.record_id] = f"{type(exc).__name__}: {exc}"
        if position % 250 == 0 or position == len(records):
            print(f"3DSC structure/spin features: {position}/{len(records)} processed; {len(rows)} ready", file=sys.stderr, flush=True)
    return rows, {
        "training_records": len(records), "feature_rows": len(rows),
        "missing": missing, "structure_failures": failures,
        "feature_coverage": {
            name: sum(row.get(name) is not None for row in rows)
            for name in ["is_magnetic_2", "ordering_2", "total_magnetization_2_per_atom",
                         "num_magnetic_sites_2_per_atom", "nearest_neighbor_min_angstrom",
                         "nearest_neighbor_mean_angstrom", "nearest_neighbor_std_angstrom",
                         "metal_nonmetal_mean_distance_angstrom", "crystalnn_mean_coordination_number",
                         "dimensionality", "layers_present"]
        },
        "magnetic_data_provenance": "DFT-computed 3DSC/Materials Project fields; not experimental measurements",
        "cif_feature_method": "pymatgen CrystalNN coordination/bonds and Larsen dimensionality on a CrystalNN structure graph; disordered CIFs use highest-occupancy ordered proxy",
    }


def _reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = None
    if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1:
        value = float(spearmanr(y, pred).statistic)
        rho = value if math.isfinite(value) else None
    return {"mae_k": float(mean_absolute_error(y, pred)),
            "rmse_k": float(np.sqrt(mean_squared_error(y, pred))),
            "spearman_rho": rho}


def _auc(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = (y > threshold).astype(int)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    return {"threshold_k": threshold, "positive_count": int(np.sum(y > threshold)),
            "row_count": len(y), "roc_auc": float(roc_auc_score(labels, scores))
            if len(labels) and len(np.unique(labels)) == 2 else None}


def evaluate_ablation(records, feature_rows: list[dict[str, Any]], *, n_splits: int = 5) -> dict[str, Any]:
    """Run cross-family and within-family OOF ablations on a common CIF cohort."""
    by_id = {row["record_id"]: row for row in feature_rows}
    common = [record for record in select_training_records(records) if record.record_id in by_id]
    if len(common) < 2:
        return {"status": "blocked_insufficient_cif_coverage", "records": len(common)}
    base, base_names, backend = build_features(common)
    def numeric_columns(names: list[str]) -> np.ndarray:
        values = []
        for record in common:
            values.append([_number(by_id[record.record_id].get(name)) for name in names])
        return np.asarray([[np.nan if value is None else value for value in row] for row in values], dtype=float)

    spin = numeric_columns(SPIN_FEATURES)
    bonds = numeric_columns(BOND_FEATURES)
    patterns = numeric_columns(PATTERN_FEATURES)
    groups = {
        "baseline": (base, list(base_names)),
        "plus_spin": (np.column_stack((base, spin)), list(base_names) + SPIN_FEATURES),
        "plus_bonds": (np.column_stack((base, bonds)), list(base_names) + BOND_FEATURES),
        "plus_patterns": (np.column_stack((base, patterns)), list(base_names) + PATTERN_FEATURES),
        "plus_all_structure_spin": (np.column_stack((base, spin, bonds, patterns)), list(base_names) + SPIN_FEATURES + BOND_FEATURES + PATTERN_FEATURES),
    }
    y = np.asarray([r.tc_kelvin for r in common], dtype=float)
    family = np.asarray([r.chemical_family for r in common])
    family_splits = min(n_splits, len(set(family)))
    if family_splits < 2:
        return {"status": "blocked_one_family", "records": len(common)}

    def run_splits(indices: np.ndarray, split_iter, *, progress_label: str) -> dict[str, Any]:
        fold_list = list(split_iter)
        results: dict[str, Any] = {}
        for name, (matrix, names) in groups.items():
            pred = np.full(len(indices), np.nan)
            baseline = np.full(len(indices), np.nan)
            classifier_scores = {threshold: np.full(len(indices), np.nan) for threshold in (0.0, 77.0)}
            fold_reports = []
            for fold, (train_local, test_local) in enumerate(fold_list, start=1):
                train, test = indices[train_local], indices[test_local]
                imputer = SimpleImputer(strategy="median", add_indicator=True)
                x_train = imputer.fit_transform(matrix[train])
                x_test = imputer.transform(matrix[test])
                reg = RandomForestRegressor(n_estimators=150, random_state=4200 + fold, n_jobs=-1, min_samples_leaf=1)
                reg.fit(x_train, y[train])
                pred[test_local] = reg.predict(x_test)
                baseline[test_local] = float(y[train].mean())
                per_fold = {"fold": fold, "train_rows": len(train), "test_rows": len(test)}
                for class_idx, threshold in enumerate((0.0, 77.0)):
                    labels = (y[train] > threshold).astype(int)
                    if len(np.unique(labels)) > 1:
                        clf = RandomForestClassifier(n_estimators=100, random_state=5200 + fold + class_idx, n_jobs=-1)
                        clf.fit(x_train, labels)
                        score = clf.predict_proba(x_test)[:, 1]
                        classifier_scores[threshold][test_local] = score
                    per_fold[f"roc_auc_tc_gt_{int(threshold)}k"] = _auc(y[test], classifier_scores[threshold][test_local], threshold)["roc_auc"]
                fold_reports.append(per_fold)
            metrics = _reg_metrics(y[indices], pred)
            metrics["mean_predictor_mae_k"] = float(mean_absolute_error(y[indices], baseline))
            metrics["feature_count"] = len(names)
            metrics["folds"] = fold_reports
            metrics["roc_auc_tc_gt_0k"] = _auc(y[indices], classifier_scores[0.0], 0.0)
            metrics["roc_auc_tc_gt_77k"] = _auc(y[indices], classifier_scores[77.0], 77.0)
            results[name] = metrics
            print(f"3DSC ablation {progress_label}: {name} complete", file=sys.stderr, flush=True)
        return results

    cross = run_splits(np.arange(len(common)), GroupKFold(n_splits=family_splits).split(base, y, family), progress_label="cross-family")
    within_results: dict[str, Any] = {}
    for family_name in sorted(set(family)):
        fam_indices = np.flatnonzero(family == family_name)
        if len(fam_indices) < 100:
            continue
        signature_groups = np.asarray([elemental_system_signature(common[i].formula) for i in fam_indices])
        split_count = min(n_splits, len(set(signature_groups)))
        if split_count < 2:
            within_results[family_name] = {"status": "blocked_insufficient_elemental_systems", "rows": len(fam_indices)}
            continue
        within_results[family_name] = {"status": "ready", "rows": len(fam_indices),
            "grouping": "GroupKFold by elemental-system signature within family",
            "feature_sets": run_splits(fam_indices, GroupKFold(n_splits=split_count).split(base[fam_indices], y[fam_indices], signature_groups), progress_label=f"within-{family_name}")}
    return {"status": "ready", "feature_backend": backend, "records": len(common),
            "families": sorted(set(family)), "cross_family": {"cv": "GroupKFold by chemical_family", "splits": family_splits, "feature_sets": cross},
            "within_family": within_results,
            "feature_groups": {"spin": SPIN_FEATURES, "bonds": BOND_FEATURES, "patterns": PATTERN_FEATURES},
            "notes": ["All feature ablations use the same CIF-backed training cohort and identical fold partitions.",
                      "Magnetic fields are DFT-computed 3DSC/Materials Project table values, not measurements.",
                      "Classifier AUC is null when the training/test folds do not contain scoreable examples from both classes."]}


def run(*, harmonized: Path, source_csv: Path, structure_index: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "3dsc-structure-spin-features.csv"
    records = load_harmonized(harmonized)
    training_ids = {record.record_id for record in select_training_records(records)}
    feature_rows: list[dict[str, Any]] = []
    if feature_path.is_file():
        with feature_path.open(newline="", encoding="utf-8-sig") as handle:
            cached = list(csv.DictReader(handle))
        cached_ids = [row.get("record_id", "") for row in cached]
        if len(cached_ids) == len(training_ids) and set(cached_ids) == training_ids:
            feature_rows = cached
    if feature_rows:
        coverage = {
            "training_records": len(training_ids), "feature_rows": len(feature_rows),
            "missing": {"source_record": 0, "cif": 0}, "structure_failures": {},
            "feature_coverage": {
                name: sum((row.get(name) not in (None, "") if name == "ordering_2" else _number(row.get(name)) is not None)
                          for row in feature_rows)
                for name in ["is_magnetic_2", "ordering_2", "total_magnetization_2_per_atom",
                             "num_magnetic_sites_2_per_atom", "nearest_neighbor_min_angstrom",
                             "nearest_neighbor_mean_angstrom", "nearest_neighbor_std_angstrom",
                             "metal_nonmetal_mean_distance_angstrom", "crystalnn_mean_coordination_number",
                             "dimensionality", "layers_present"]
            },
            "magnetic_data_provenance": "DFT-computed 3DSC/Materials Project fields; not experimental measurements",
            "cif_feature_method": "pymatgen CrystalNN coordination/bonds and Larsen dimensionality on a CrystalNN structure graph; disordered CIFs use highest-occupancy ordered proxy",
            "feature_cache_reused": True,
        }
    else:
        feature_rows, coverage = build_feature_rows(harmonized_path=harmonized, source_csv=source_csv, structure_index=structure_index)
        if feature_rows:
            fields = list(dict.fromkeys(key for row in feature_rows for key in row))
            with feature_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(feature_rows)
        coverage["feature_cache_reused"] = False
    evaluation = evaluate_ablation(records, feature_rows)
    report = {"coverage": coverage, "evaluation": evaluation, "features_csv": str(feature_path)}
    report_path = output_dir / "3dsc-structure-spin-ablation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--source", type=Path, default=Path("data/raw/3dsc/MP/3DSC_MP.csv"))
    parser.add_argument("--structure-index", type=Path, default=Path("data/processed/active-learning/3dsc-structure-index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/active-learning"))
    args = parser.parse_args()
    report = run(harmonized=args.harmonized, source_csv=args.source, structure_index=args.structure_index, output_dir=args.output_dir)
    print(json.dumps({"coverage": report["coverage"], "evaluation_status": report["evaluation"].get("status"),
                      "feature_rows": report["coverage"]["feature_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
