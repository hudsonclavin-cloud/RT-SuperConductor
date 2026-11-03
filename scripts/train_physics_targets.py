#!/usr/bin/env python3
"""Train auditable electron-phonon target models on JARVIS-SuperconDB.

The script downloads a version-pinned, public DFT dataset, verifies both its
publisher-provided MD5 and this repository's SHA-256 pin, builds one
characteristics row per 3D compound, evaluates grouped feature ablations, and
fits lambda/omega_log random-forest models. It does not run DFT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.descriptors import descriptor_names, formula_descriptor_record

DATASET_URL = "https://ndownloader.figshare.com/files/38307921"
DATASET_DOI = "10.6084/m9.figshare.21370572.v4"
PAPER_DOI = "10.1038/s41524-022-00933-1"
DATASET_FILE = "jarvis_epc_data_figshare_1058.json.zip"
PUBLISHER_MD5 = "bb4a91e899458f34cbb3b9bf03e9bd56"
# SHA-256 measured from the Figshare v4 file downloaded for this run.
DATASET_SHA256 = "4213651cee9b4c13a376dd74ac1f8d3bd9efcd451f1b8da615255c8df0e1487f"
MU_STAR = 0.1
LIGHT_ELEMENT_Z_MAX = 10
BOND_FEATURE_NAMES = [
    "bond_nearest_neighbor_min_angstrom", "bond_nearest_neighbor_mean_angstrom",
    "bond_nearest_neighbor_std_angstrom", "bond_metal_nonmetal_mean_distance_angstrom",
    "bond_crystalnn_mean_coordination_number",
]
ATOMIC_MASS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403, "Ne": 20.1797,
}


def _download_dataset(path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(DATASET_URL, path)
    payload = path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()
    if sha256 != DATASET_SHA256 or md5 != PUBLISHER_MD5:
        raise ValueError(f"JARVIS-SuperconDB checksum mismatch (sha256={sha256}, md5={md5})")
    return sha256, md5


def allen_dynes_tc_k(lam: float, omega_log_k: float, mu_star: float = MU_STAR) -> float:
    """Modified McMillan/Allen-Dynes Tc without strong-coupling f1/f2 factors."""

    if not (math.isfinite(lam) and math.isfinite(omega_log_k)) or lam <= 0 or omega_log_k <= 0:
        return float("nan")
    denominator = lam - mu_star * (1.0 + 0.62 * lam)
    if denominator <= 0:
        return 0.0
    return omega_log_k / 1.2 * math.exp(-1.04 * (1.0 + lam) / denominator)


def _formula_and_structure(row: dict[str, Any]) -> tuple[str, float, float, float, int]:
    from pymatgen.core import Composition

    atoms = row["atoms"]
    symbols = [str(symbol) for symbol in atoms["elements"]]
    counts: dict[str, float] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0.0) + 1.0
    composition = Composition(counts)
    formula = composition.reduced_formula
    if any(mark in formula for mark in "()[]"):
        # The project's formula and shell descriptors accept flattened
        # element/count tokens, so flatten grouped formulas without changing
        # stoichiometry when pymatgen's preferred formula retains parentheses.
        reduced = composition.reduced_composition
        formula = "".join(
            f"{element}{'' if abs(float(amount) - 1.0) < 1e-8 else format(float(amount), '.8g')}"
            for element, amount in sorted(reduced.get_el_amt_dict().items())
        )
    lattice = np.asarray(atoms["lattice_mat"], dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    density = float(composition.weight) / volume * 1.66053906660
    total = sum(counts.values())
    h_fraction = counts.get("H", 0.0) / total
    light_fraction = sum(value for symbol, value in counts.items()
                         if symbol in {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"}) / total
    return formula, density, h_fraction, light_fraction, len(symbols)


def _jarvis_bond_features(atoms: dict[str, Any]) -> dict[str, float | None]:
    """Compute periodic bond geometry and CrystalNN coordination on JARVIS atoms."""
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Element, Structure

    structure = Structure(
        np.asarray(atoms["lattice_mat"], dtype=float), atoms["elements"],
        atoms["coords"], coords_are_cartesian=bool(atoms.get("cartesian", False)),
    )
    neighbors = structure.get_all_neighbors(r=12.0, include_index=True)
    nearest = [min(float(n.nn_distance) for n in row if n.nn_distance > 1e-8)
               for row in neighbors if any(n.nn_distance > 1e-8 for n in row)]
    crystal_nn = CrystalNN()
    coordination: list[float] = []
    hetero_bonds: list[float] = []
    for index, site in enumerate(structure):
        coordination.append(float(crystal_nn.get_cn(structure, index)))
        center_is_metal = Element(str(next(iter(site.species)))).is_metal
        for info in crystal_nn.get_nn_info(structure, index):
            neighbor = info["site"]
            neighbor_is_metal = Element(str(next(iter(neighbor.species)))).is_metal
            if center_is_metal != neighbor_is_metal:
                hetero_bonds.append(float(site.distance(neighbor)))
    return {
        "bond_nearest_neighbor_min_angstrom": min(nearest) if nearest else None,
        "bond_nearest_neighbor_mean_angstrom": float(np.mean(nearest)) if nearest else None,
        "bond_nearest_neighbor_std_angstrom": float(np.std(nearest)) if nearest else None,
        "bond_metal_nonmetal_mean_distance_angstrom": float(np.mean(hetero_bonds)) if hetero_bonds else None,
        "bond_crystalnn_mean_coordination_number": float(np.mean(coordination)) if coordination else None,
    }


def build_characteristics(
    rows: list[dict[str, Any]], mp_cache_path: Path | None = None
) -> tuple[list[dict[str, Any]], list[str], str]:
    """Build one auditable feature row per material, leaving absent MP data null."""

    import pandas as pd
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    mp_by_jid: dict[str, dict[str, Any]] = {}
    mp_status = "not_requested"
    if mp_cache_path and mp_cache_path.is_file():
        with mp_cache_path.open(newline="", encoding="utf-8-sig") as handle:
            mp_by_jid = {r["jid"]: r for r in csv.DictReader(handle) if r.get("jid")}
        mp_status = "loaded_cache"

    formula_rows = [_formula_and_structure(row) for row in rows]
    compositions = [Composition(formula) for formula, *_ in formula_rows]
    frame = pd.DataFrame({"composition": compositions})
    magpie = ElementProperty.from_preset("magpie").featurize_dataframe(
        frame, col_id="composition", ignore_errors=False, pbar=False
    )
    magpie_names = [str(name) for name in magpie.columns if name != "composition"]
    output: list[dict[str, Any]] = []
    for source, (formula, structure_density, h_fraction, light_fraction, site_count), (_, magpie_row) in zip(
        rows, formula_rows, magpie.iterrows()
    ):
        descriptor = formula_descriptor_record(formula)
        amounts = Composition(formula).get_el_amt_dict()
        lewis_values = {
            token.split(":", 1)[0]: float(token.split(":", 1)[1])
            for token in str(descriptor["lewis_dot_proxy_outer_electrons"]).split(";") if ":" in token
        }
        total_atoms = sum(float(value) for value in amounts.values())
        lewis_avg = sum(float(amounts[element]) * lewis_values[element] for element in amounts) / total_atoms
        mp = mp_by_jid.get(str(source["jid"]), {})
        row: dict[str, Any] = {
            "jid": source["jid"], "formula": formula, "source_dataset_doi": DATASET_DOI,
            "source_dataset_license": "CC BY 4.0", "source_stability": source.get("stability"),
            "lambda": float(source["lamb"]), "omega_log_k": float(source["wlog"]),
            "source_tc_k": _float_or_none(source.get("Tc")), "allen_dynes_tc_mu0p1_k": allen_dynes_tc_k(
                float(source["lamb"]), float(source["wlog"])
            ),
            "source_pressure": source.get("press"), "structure_density_g_cm3": structure_density,
            "hydrogen_fraction": h_fraction, "light_element_fraction_z_le_10": light_fraction,
            **_jarvis_bond_features(source["atoms"]),
            "lewis_dot_proxy_avg_outer_electrons": lewis_avg,
            "lewis_dot_proxy_total_electrons_per_reduced_formula": lewis_avg * total_atoms,
            "structure_site_count": site_count,
            "mp_material_id": mp.get("mp_material_id", ""),
            "mp_dos_ef_states_per_ev_per_cell": _float_or_none(mp.get("mp_dos_ef_states_per_ev_per_cell")),
            "mp_band_gap_ev": _float_or_none(mp.get("mp_band_gap_ev")),
            "mp_is_metal": _bool_or_none(mp.get("mp_is_metal")),
            "mp_density_g_cm3": _float_or_none(mp.get("mp_density_g_cm3")),
            "mp_match_status": mp.get("mp_match_status", mp_status),
        }
        row.update({f"magpie__{name}": float(magpie_row[name]) for name in magpie_names})
        row.update({f"shell__{key}": value for key, value in descriptor.items()
                    if key in descriptor_names() and value is not None})
        row.update({key: value for key, value in descriptor.items() if key not in descriptor_names()})
        output.append(row)
    feature_names = (
        [f"magpie__{name}" for name in magpie_names]
        + [f"shell__{name}" for name in descriptor_names()]
        + ["lewis_dot_proxy_avg_outer_electrons", "lewis_dot_proxy_total_electrons_per_reduced_formula"]
    )
    return output, feature_names, mp_status


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in {"true", "1", "yes"} else False if text in {"false", "0", "no"} else None


def _element_system(formula: str) -> str:
    from pymatgen.core import Composition

    return "-".join(sorted(str(el) for el in Composition(formula).elements))


def _make_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[np.nan if row.get(name) in (None, "") else float(row[name])
                        for name in feature_names] for row in rows], dtype=float)


def evaluate_ablation(rows: list[dict[str, Any]], base_feature_names: list[str], n_splits: int = 5) -> dict[str, Any]:
    """Group-held-out ablation for direct Tc baseline and lambda/wlog physics model."""

    from sklearn.ensemble import RandomForestRegressor

    labels_lam = np.asarray([float(row["lambda"]) for row in rows])
    labels_wlog = np.asarray([float(row["omega_log_k"]) for row in rows])
    labels_tc = np.asarray([float(row["allen_dynes_tc_mu0p1_k"]) for row in rows])
    groups = np.asarray([_element_system(row["formula"]) for row in rows])
    splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    mp_names = [
        "mp_dos_ef_states_per_ev_per_cell", "mp_band_gap_ev", "mp_is_metal", "mp_density_g_cm3"
    ]
    mp_coverage = sum(all(_float_or_none(row.get(name)) is not None for name in mp_names) for row in rows)
    feature_sets = {
        "magpie_plus_lewis_shell": list(base_feature_names),
        "plus_structure": list(base_feature_names) + [
            "structure_density_g_cm3", "hydrogen_fraction", "light_element_fraction_z_le_10", "structure_site_count"
        ],
        "plus_bond_features": list(base_feature_names) + BOND_FEATURE_NAMES,
        "plus_structure_and_bonds": list(base_feature_names) + [
            "structure_density_g_cm3", "hydrogen_fraction", "light_element_fraction_z_le_10", "structure_site_count"
        ] + BOND_FEATURE_NAMES,
    }
    if mp_coverage >= max(50, int(0.2 * len(rows))):
        feature_sets["plus_materials_project_electronic"] = feature_sets["plus_structure_and_bonds"] + mp_names

    results: dict[str, Any] = {}
    for label, feature_names in feature_sets.items():
        X = _make_matrix(rows, feature_names)
        fold_predictions: dict[str, list[float]] = {"direct_tc": [], "physics_tc": [], "lambda": [], "omega_log": []}
        fold_truth: dict[str, list[float]] = {"direct_tc": [], "physics_tc": [], "lambda": [], "omega_log": []}
        for fold, (train, test) in enumerate(splitter.split(X, labels_tc, groups), 1):
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            x_train = imputer.fit_transform(X[train])
            x_test = imputer.transform(X[test])
            direct = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=2026 + fold, n_jobs=-1)
            model_lam = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=3026 + fold, n_jobs=-1)
            model_wlog = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=4026 + fold, n_jobs=-1)
            direct.fit(x_train, labels_tc[train])
            model_lam.fit(x_train, labels_lam[train])
            model_wlog.fit(x_train, np.log(labels_wlog[train]))
            p_direct = direct.predict(x_test)
            p_lam = model_lam.predict(x_test)
            p_wlog = np.exp(model_wlog.predict(x_test))
            p_physics = np.asarray([allen_dynes_tc_k(lam, wlog) for lam, wlog in zip(p_lam, p_wlog)])
            for key, pred, actual in [
                ("direct_tc", p_direct, labels_tc[test]),
                ("physics_tc", p_physics, labels_tc[test]),
                ("lambda", p_lam, labels_lam[test]),
                ("omega_log", p_wlog, labels_wlog[test]),
            ]:
                fold_predictions[key].extend(float(x) for x in pred)
                fold_truth[key].extend(float(x) for x in actual)
        metrics: dict[str, Any] = {}
        for key in fold_truth:
            truth = np.asarray(fold_truth[key])
            pred = np.asarray(fold_predictions[key])
            metrics[key] = {
                "mae": float(mean_absolute_error(truth, pred)),
                "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
                "unit": "K" if key in {"direct_tc", "physics_tc", "omega_log"} else "dimensionless",
            }
        results[label] = {"feature_count": len(feature_names), "metrics": metrics}
    baseline_mae = results["magpie_plus_lewis_shell"]["metrics"]["physics_tc"]["mae"]
    baseline_omega_log_mae = results["magpie_plus_lewis_shell"]["metrics"]["omega_log"]["mae"]
    for result in results.values():
        result["delta_physics_tc_mae_vs_baseline_k"] = (
            result["metrics"]["physics_tc"]["mae"] - baseline_mae
        )
        result["delta_omega_log_mae_vs_baseline_k"] = (
            result["metrics"]["omega_log"]["mae"] - baseline_omega_log_mae
        )
    if "plus_materials_project_electronic" not in results:
        results["plus_materials_project_electronic"] = {
            "status": "blocked_insufficient_mp_coverage",
            "complete_mp_rows": mp_coverage,
            "required_complete_mp_rows": max(50, int(0.2 * len(rows))),
            "metrics": None,
            "delta_physics_tc_mae_vs_baseline_k": None,
        }
    return {
        "split": "5-fold GroupKFold by full elemental-system signature",
        "rows": len(rows), "groups": len(set(groups)), "mp_complete_rows": mp_coverage,
        "mu_star": MU_STAR, "feature_sets": results,
    }


def fit_physics_models(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    X = _make_matrix(rows, feature_names)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_fit = imputer.fit_transform(X)
    lam = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    wlog = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=43, n_jobs=-1)
    lam.fit(X_fit, [float(row["lambda"]) for row in rows])
    wlog.fit(X_fit, np.log([float(row["omega_log_k"]) for row in rows]))
    models = {"lambda": lam, "log_omega_log_k": wlog, "imputer": imputer,
              "feature_names": feature_names, "mu_star": MU_STAR}
    model_meta = {
        "training_rows": len(rows), "feature_count": len(feature_names),
        "lambda_training_range": [min(float(r["lambda"]) for r in rows), max(float(r["lambda"]) for r in rows)],
        "omega_log_training_range_k": [min(float(r["omega_log_k"]) for r in rows), max(float(r["omega_log_k"]) for r in rows)],
        "training_tc_max_k": max(float(r["source_tc_k"]) for r in rows if r["source_tc_k"] is not None),
        "target_tc_max_mu0p1_k": max(float(r["allen_dynes_tc_mu0p1_k"]) for r in rows),
        "uncertainty": "standard deviation of per-tree predictions; empirical ensemble spread, not calibrated confidence interval",
    }
    return models, model_meta


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(*, data_dir: Path, output_dir: Path, mp_cache: Path | None = None) -> dict[str, Any]:
    from json import load

    archive = data_dir / DATASET_FILE
    file_sha256, file_md5 = _download_dataset(archive)
    with zipfile.ZipFile(archive) as zf:
        source_rows = load(zf.open("jarvis_epc_data_figshare_1058.json"))
    usable_rows = [
        row for row in source_rows
        if _float_or_none(row.get("lamb")) is not None
        and _float_or_none(row.get("wlog")) is not None
        and float(row["lamb"]) > 0
        and float(row["wlog"]) > 0
    ]
    table_path = output_dir / "jarvis-supercond-characteristics.csv"
    required_bonds = set(BOND_FEATURE_NAMES)
    cached_rows: list[dict[str, Any]] = []
    if table_path.is_file():
        with table_path.open(newline="", encoding="utf-8-sig") as handle:
            cached_rows = list(csv.DictReader(handle))
    cached_jids = {str(row.get("jid")) for row in cached_rows}
    expected_jids = {str(row["jid"]) for row in usable_rows}
    if cached_rows and required_bonds.issubset(cached_rows[0]) and cached_jids == expected_jids:
        rows = cached_rows
        # Recover the stable base schema from its prefixed feature keys.
        feature_names = [key for key in rows[0]
                         if key.startswith("magpie__") or key.startswith("shell__")]
        feature_names += ["lewis_dot_proxy_avg_outer_electrons",
                          "lewis_dot_proxy_total_electrons_per_reduced_formula"]
        mp_status = "loaded_existing_characteristics_cache"
    else:
        rows, feature_names, mp_status = build_characteristics(usable_rows, mp_cache)
        _write_csv(table_path, rows)
    ablation = evaluate_ablation(rows, feature_names)
    model_feature_names = list(feature_names) + BOND_FEATURE_NAMES
    models, model_meta = fit_physics_models(rows, model_feature_names)
    model_path = output_dir / "electron-phonon-target-models.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_path)
    mp_status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("mp_match_status", "unknown"))
        mp_status_counts[status] = mp_status_counts.get(status, 0) + 1
    report = {
        "dataset": {
            "name": "JARVIS-SuperconDB (3D, 1058-row file)", "doi": DATASET_DOI,
            "paper_doi": PAPER_DOI, "url": DATASET_URL, "license": "CC BY 4.0",
            "publisher_md5": file_md5, "sha256": file_sha256, "raw_rows": len(source_rows),
            "rows_used": len(rows), "rows_excluded_invalid_electron_phonon_targets": len(source_rows) - len(rows),
            "mp_characteristics_status": mp_status, "mp_match_status_counts": mp_status_counts,
        },
        "method": {
            "targets": ["lambda", "omega_log_k"], "mu_star": MU_STAR,
            "bond_features": BOND_FEATURE_NAMES,
            "bond_feature_method": "periodic per-site nearest-neighbor statistics, CrystalNN heterometal/nonmetal neighbor distance, and mean CrystalNN coordination",
            "tc_equation": "omega_log/1.2 * exp[-1.04*(1+lambda)/(lambda-mu_star*(1+0.62*lambda))]; omega_log in kelvin; f1/f2 strong-coupling corrections omitted",
            "grouping": "5-fold GroupKFold by elemental-system signature",
            "random_state": 2026,
            "cross_validation_estimators": 180,
            "cross_validation_min_samples_leaf": 2,
            "full_fit_estimators": 300,
            "full_fit_min_samples_leaf": 2,
        },
        "model": model_meta,
        "ablation": ablation,
        "artifacts": {"characteristics_csv": str(table_path), "model_joblib": str(model_path)},
        "limitations": [
            "This is a phonon-mediated model trained on the JARVIS-SuperconDB domain, not unconventional superconductivity.",
            "Random forests are bounded in lambda and omega_log targets by training target extrema; derived Tc can differ because the nonlinear Allen-Dynes map combines both targets.",
            "Ensemble tree spread is not calibrated predictive uncertainty.",
            "Materials Project features remain null unless a formula-verified Materials Project cache is provided.",
            "The characteristics table's Lewis and shell descriptors are neutral-atom formula proxies, not solid-state charge or bond analyses.",
            "Nearest-neighbor and CrystalNN descriptors are geometric/coordination proxies; they are not charge-density bond analyses.",
        ],
    }
    report_path = output_dir / "physics-target-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/jarvis-supercond"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/physics-target"))
    parser.add_argument("--mp-cache", type=Path, default=Path("data/raw/jarvis-supercond/mp-characteristics.csv"))
    args = parser.parse_args()
    report = run(data_dir=args.data_dir, output_dir=args.output_dir, mp_cache=args.mp_cache)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
