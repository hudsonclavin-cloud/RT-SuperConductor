#!/usr/bin/env python3
"""Cuprate-specific features: the quantities experiments tie to cuprate Tc.

Nobody can compute a cuprate's Tc from first principles, so there is no
cuprate counterpart of EPW. What exists are empirical regularities. This
script turns the ones that can be read from a formula and a crystal structure
into numbers:

* hole doping per copper, from charge balance with nominal valences. Tc
  follows a dome in doping (Presland et al., Physica C 176, 95 (1991)):
  Tc / Tc_max = 1 - 82.6 (p - 0.16)^2.
* how many CuO2 planes sit next to each other. Tc_max rises from one plane to
  three, then falls.
* the apical-oxygen distance and how many apical oxygens each planar copper
  has. A more distant apical oxygen goes with a higher Tc_max (Pavarini et
  al., Phys. Rev. Lett. 87, 047003 (2001)).
* in-plane Cu-O bond length and plane buckling.
* how much chemical substitution the formula carries. Disorder near the planes
  lowers Tc_max (Eisaki et al., Phys. Rev. B 69, 064512 (2004)).

Limits, stated once here and repeated in the output file's report:

* The 3DSC structures are Materials Project parent structures with the doping
  added artificially, so the geometry describes the parent compound.
* Doping is a formula-level average over every copper atom. In compounds with
  Cu-O chains (YBa2Cu3O7) part of the charge sits on the chains, and oxygen
  contents in the source formulas are often nominal.
* These are correlations, not a mechanism. Each one has to earn its place in
  scripts/eval_cuprate_models.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PRESLAND_OPTIMUM = 0.16
PRESLAND_CURVATURE = 82.6
MAX_LAYER_WALK = 6          # more adjacent planes than this means an infinite-layer compound

# Nominal valences for charge balance. AMBIGUOUS lists elements whose valence in
# cuprates is genuinely uncertain or mixed; rows containing them are flagged.
VALENCE = {
    **{el: 1 for el in ("Li", "Na", "K", "Rb", "Cs", "Ag")},
    **{el: 2 for el in ("Mg", "Ca", "Sr", "Ba", "Zn", "Cd", "Hg", "Pb", "Ni", "Co", "Mn", "Be")},
    **{el: 3 for el in ("Y", "Sc", "La", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                        "Yb", "Lu", "Bi", "Tl", "Al", "Ga", "In", "Fe", "B", "Cr", "Au")},
    **{el: 4 for el in ("Ce", "Ti", "Zr", "Hf", "Sn", "C", "Th", "Ge", "Si", "Ru", "Ir", "Pt", "Te")},
    **{el: 5 for el in ("Nb", "Ta", "V", "P", "Sb", "As", "N")},
    **{el: 6 for el in ("Mo", "W", "S", "Se", "U")},
    "Re": 7,
    "O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1, "H": 1,
}
AMBIGUOUS = {"Pb", "Tl", "Bi", "Ru", "Pr", "Tb", "Fe", "Co", "Mn", "Mo", "W", "Re", "V", "Cr", "Ce", "S", "Se",
             "Ir", "Sb", "U", "N", "Au", "Ag", "Te"}

FEATURE_NAMES = [
    "cu_valence_formula", "hole_doping_p", "abs_doping", "doping_distance_from_optimum", "presland_dome",
    "electron_doped", "valence_ambiguous",
    "cuo2_layer_count", "infinite_layer", "planar_cu_fraction", "in_plane_cu_o_mean_a", "plane_buckling_rms_a",
    "apical_o_count_mean", "apical_o_distance_min_a", "apical_o_distance_mean_a", "has_apical_oxygen",
    "in_plane_cu_cu_mean_a", "fractional_species_count", "largest_minor_fraction", "dopant_fraction_total",
]


def presland_dome(p: float) -> float:
    """Tc / Tc_max from the empirical doping dome, clipped at zero. Uses |p| so
    electron-doped compounds, whose dome is similar in width, are treated alike."""
    return max(0.0, 1.0 - PRESLAND_CURVATURE * (abs(p) - PRESLAND_OPTIMUM) ** 2)


def doping_features(formula: str) -> dict[str, float]:
    """Formula-level copper valence and hole doping from nominal charge balance."""
    from pymatgen.core import Composition

    amounts = Composition(formula).get_el_amt_dict()
    cu = amounts.get("Cu", 0.0)
    unknown = [el for el in amounts if el != "Cu" and el not in VALENCE]
    nan = float("nan")
    out = {"cu_valence_formula": nan, "hole_doping_p": nan, "abs_doping": nan,
           "doping_distance_from_optimum": nan, "presland_dome": nan, "electron_doped": nan,
           "valence_ambiguous": float(bool(unknown) or any(el in AMBIGUOUS for el in amounts))}
    if cu <= 0 or unknown:
        return out
    other_charge = sum(VALENCE[el] * amt for el, amt in amounts.items() if el != "Cu")
    valence = -other_charge / cu
    p = valence - 2.0
    out.update({
        "cu_valence_formula": valence, "hole_doping_p": p, "abs_doping": abs(p),
        "doping_distance_from_optimum": abs(abs(p) - PRESLAND_OPTIMUM),
        "presland_dome": presland_dome(p), "electron_doped": float(p < 0),
    })
    return out


def disorder_features(formula: str) -> dict[str, float]:
    """How much substitution the formula carries (a proxy for cation disorder)."""
    from pymatgen.core import Composition

    amounts = {el: amt for el, amt in Composition(formula).get_el_amt_dict().items() if el not in ("O", "Cu")}
    fractional = {el: amt for el, amt in amounts.items() if abs(amt - round(amt)) > 1e-6}
    minor = [min(amt - math.floor(amt), math.ceil(amt) - amt) for amt in fractional.values()]
    return {
        "fractional_species_count": float(len(fractional)),
        "largest_minor_fraction": float(max(minor)) if minor else 0.0,
        "dopant_fraction_total": float(sum(minor)),
    }


def _has(site: Any, element: str, minimum: float) -> bool:
    return any(str(sp.symbol if hasattr(sp, "symbol") else sp) == element and occ >= minimum
               for sp, occ in site.species.items())


def plane_geometry(structure: Any) -> dict[str, float]:
    """CuO2-plane geometry: layer count, bond lengths, buckling and apical oxygens.

    A copper atom counts as planar when its four nearest oxygens form a flat
    square around it and each of those oxygens bridges in a nearly straight
    line to another copper, which is what a corner-sharing CuO2 sheet looks
    like. The second condition separates planes from the Cu-O chains of
    YBa2Cu3O7 and YBa2Cu4O8, whose coppers also have four flat oxygen
    neighbours. Buckling is the copper's height above its four oxygens.
    """
    nan = float("nan")
    empty = {"cuo2_layer_count": nan, "infinite_layer": nan, "planar_cu_fraction": nan,
             "in_plane_cu_o_mean_a": nan, "plane_buckling_rms_a": nan, "apical_o_count_mean": nan,
             "apical_o_distance_min_a": nan, "apical_o_distance_mean_a": nan, "has_apical_oxygen": nan,
             "in_plane_cu_cu_mean_a": nan}
    cu_sites = [i for i, site in enumerate(structure) if _has(site, "Cu", 0.5)]
    if not cu_sites:
        return empty
    neighbours = structure.get_all_neighbors(4.4)
    planar: dict[int, dict[str, Any]] = {}
    for i in cu_sites:
        origin = structure[i].coords
        oxygens = sorted((n for n in neighbours[i] if _has(n, "O", 0.3) and n.nn_distance < 3.3),
                         key=lambda n: n.nn_distance)
        if len(oxygens) < 4:
            continue
        square = oxygens[:4]
        vectors = np.asarray([n.coords - origin for n in square])
        distances = [n.nn_distance for n in square]
        centre = vectors.mean(axis=0)
        _, singular, vt = np.linalg.svd(vectors - centre)
        normal = vt[-1]
        oxygen_rms = float(singular[-1] / 2.0)               # how far the four oxygens are from a common plane
        cu_offset = float(abs(np.dot(centre, normal)))       # copper's height above that plane ("dimpling")
        sideways = float(np.linalg.norm(centre - np.dot(centre, normal) * normal))
        if max(distances) > 2.2 or oxygen_rms > 0.35 or cu_offset > 0.5 or sideways > 0.35:
            continue
        coppers = [n for n in neighbours[i] if _has(n, "Cu", 0.5)]
        # In a CuO2 sheet every one of the four oxygens bridges, almost in a
        # straight line and at a similar distance, to another copper. Chain
        # coppers fail this: their apical oxygens reach the plane copper only at
        # about 2.3 A, and double-chain oxygens bridge at right angles.
        bridged = 0
        for oxygen in square:
            to_here = origin - oxygen.coords
            for copper in coppers:
                to_there = copper.coords - oxygen.coords
                far = float(np.linalg.norm(to_there))
                if 1e-6 < far < 2.2 and np.dot(to_here, to_there) / (oxygen.nn_distance * far) < -0.866:
                    bridged += 1
                    break
        if bridged < 4:
            continue
        in_plane_cu = [n for n in coppers if n.nn_distance < 4.2 and abs(np.dot(n.coords - origin, normal)) < 0.8]
        apical = [n.nn_distance for n in oxygens[4:]
                  if abs(np.dot(n.coords - origin, normal)) / n.nn_distance > 0.85]
        planar[i] = {"normal": normal, "cu_o": float(np.mean(distances)), "buckling": cu_offset,
                     "apical": apical,
                     "cu_cu": float(np.mean([n.nn_distance for n in in_plane_cu])) if in_plane_cu else float("nan"),
                     "coppers": coppers, "origin": origin}
    if not planar:
        return {**empty, "planar_cu_fraction": 0.0, "cuo2_layer_count": 0.0, "infinite_layer": 0.0}

    def stacked(i: int, direction: np.ndarray) -> int | None:
        """The planar copper directly above ``i`` along ``direction`` with no oxygen in between."""
        for n in planar[i]["coppers"]:
            offset = n.coords - planar[i]["origin"]
            if n.index in planar and n.nn_distance < 4.3 and np.dot(offset, direction) / n.nn_distance > 0.95:
                return n.index
        return None

    def walk(i: int, direction: np.ndarray) -> int:
        steps, current = 0, i
        while steps < MAX_LAYER_WALK:
            following = stacked(current, direction)
            if following is None:
                break
            steps, current = steps + 1, following
        return steps

    layers = []
    for i, info in planar.items():
        layers.append(1 + walk(i, info["normal"]) + walk(i, -info["normal"]))
    layer_count = max(layers)
    infinite = layer_count > MAX_LAYER_WALK
    apical_all = [d for info in planar.values() for d in info["apical"]]
    return {
        "cuo2_layer_count": float(min(layer_count, MAX_LAYER_WALK + 1)),
        "infinite_layer": float(infinite),
        "planar_cu_fraction": len(planar) / len(cu_sites),
        "in_plane_cu_o_mean_a": float(np.mean([info["cu_o"] for info in planar.values()])),
        "plane_buckling_rms_a": float(np.mean([info["buckling"] for info in planar.values()])),
        "apical_o_count_mean": float(np.mean([len(info["apical"]) for info in planar.values()])),
        "apical_o_distance_min_a": float(min(apical_all)) if apical_all else nan,
        "apical_o_distance_mean_a": float(np.mean(apical_all)) if apical_all else nan,
        "has_apical_oxygen": float(bool(apical_all)),
        "in_plane_cu_cu_mean_a": float(np.nanmean([info["cu_cu"] for info in planar.values()])),
    }


def read_3dsc_cuprates(table: Path) -> list[dict[str, str]]:
    lines = table.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if "formula_sc" in line)
    return [row for row in csv.DictReader(lines[start:]) if row.get("sc_class") == "Cuprate"]


def find_cif(row: dict[str, str], cif_roots: list[Path]) -> Path | None:
    """The row's CIF under either layout: the 3DSC clone or the downloader's cifs folder."""
    relative = row.get("cif", "")
    for root in cif_roots:
        for candidate in (root / relative, root / Path(relative).name):
            if candidate.is_file():
                return candidate
    return None


def load_overrides(directory: Path | None) -> list[dict[str, Any]]:
    """Rules that replace a wrongly matched parent structure (see rules.json)."""
    if directory is None or not (directory / "rules.json").is_file():
        return []
    rules = json.loads((directory / "rules.json").read_text(encoding="utf-8"))["rules"]
    for rule in rules:
        rule["cif_path"] = directory / rule["replacement_cif"]
    return rules


def override_for(formula: str, parent_id: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The rule that applies to this row, decided from the formula alone."""
    from pymatgen.core import Composition

    elements = set(Composition(formula).get_el_amt_dict())
    for rule in rules:
        if rule["mp_parent_id"] != parent_id:
            continue
        if "applies_if_formula_contains" in rule and not elements & set(rule["applies_if_formula_contains"]):
            continue
        if elements & set(rule.get("applies_unless_formula_contains", [])):
            continue
        return rule
    return None


def build(table: Path, cif_roots: list[Path], output: Path, report: Path,
          overrides: Path | None = None) -> dict[str, Any]:
    import warnings

    from pymatgen.core import Structure

    rows = read_3dsc_cuprates(table)
    rules = load_overrides(overrides)
    overridden: dict[str, int] = {}
    geometry_cache: dict[str, dict[str, float]] = {}
    out_rows, missing_cif, failed = [], 0, 0
    for row in rows:
        formula = row["formula_sc"]
        record: dict[str, Any] = {
            "formula": formula, "tc_k": float(row["tc"]), "mp_parent_id": row.get("material_id_2", ""),
            "parent_formula": row.get("pretty_formula_2", ""), "spacegroup": row.get("spacegroup_2", ""),
        }
        record.update(doping_features(formula))
        record.update(disorder_features(formula))
        # The geometry comes from the parent structure, so compute it once per parent.
        key = row.get("material_id_2", "") or row.get("cif", "")
        rule = override_for(formula, row.get("material_id_2", ""), rules)
        record["geometry_source_id"] = rule["replacement_id"] if rule else row.get("material_id_2", "")
        record["geometry_overridden"] = int(bool(rule))
        if rule:
            key = "override:" + rule["replacement_id"]
            overridden[rule["mp_parent_id"]] = overridden.get(rule["mp_parent_id"], 0) + 1
        if key not in geometry_cache:
            cif = rule["cif_path"] if rule else find_cif(row, cif_roots)
            if cif is None:
                missing_cif += 1
                geometry_cache[key] = plane_geometry_unavailable()
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        geometry_cache[key] = plane_geometry(Structure.from_file(str(cif)))
                except Exception:  # noqa: BLE001 - a CIF pymatgen cannot read is recorded, not fatal
                    failed += 1
                    geometry_cache[key] = plane_geometry_unavailable()
        record.update(geometry_cache[key])
        out_rows.append(record)
    out_rows.sort(key=lambda r: (r["mp_parent_id"], r["formula"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["formula", "tc_k", "mp_parent_id", "parent_formula", "spacegroup", "geometry_source_id",
              "geometry_overridden"] + FEATURE_NAMES
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in out_rows:
            writer.writerow({k: ("" if isinstance(record[k], float) and math.isnan(record[k]) else
                                 (round(record[k], 6) if isinstance(record[k], float) else record[k]))
                             for k in fields})

    def share(name: str) -> float:
        values = [r[name] for r in out_rows]
        return sum(not (isinstance(v, float) and math.isnan(v)) for v in values) / max(1, len(values))

    layer_counts: dict[str, int] = {}
    for record in out_rows:
        value = record["cuo2_layer_count"]
        label = "unknown" if math.isnan(value) else ("infinite" if record["infinite_layer"] else str(int(value)))
        layer_counts[label] = layer_counts.get(label, 0) + 1
    summary = {
        "rows": len(out_rows), "superconducting_rows": sum(r["tc_k"] > 0 for r in out_rows),
        "parent_structures": len(geometry_cache), "rows_without_cif": missing_cif, "unreadable_cifs": failed,
        "rows_with_corrected_parent_structure": dict(sorted(overridden.items())),
        "parent_structure_rules": [{k: v for k, v in rule.items() if k != "cif_path"} for rule in rules],
        "coverage": {name: round(share(name), 4) for name in ("hole_doping_p", "cuo2_layer_count",
                                                               "apical_o_distance_min_a")},
        "rows_with_ambiguous_valence": int(sum(r["valence_ambiguous"] == 1.0 for r in out_rows)),
        "rows_by_cuo2_layer_count": dict(sorted(layer_counts.items())),
        "source_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "features": FEATURE_NAMES,
        "limits": [
            "Geometry is the Materials Project parent structure; 3DSC adds the doping artificially.",
            "3DSC can match a family to the wrong polymorph. Two large cases are corrected by "
            "data/training/cuprate-parent-overrides/rules.json; smaller ones may remain.",
            "Doping is a formula-level average over all copper atoms, with nominal valences and nominal oxygen.",
            "These are empirical correlations from the literature, not a mechanism.",
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def plane_geometry_unavailable() -> dict[str, float]:
    return {name: float("nan") for name in ("cuo2_layer_count", "infinite_layer", "planar_cu_fraction",
                                              "in_plane_cu_o_mean_a", "plane_buckling_rms_a", "apical_o_count_mean",
                                              "apical_o_distance_min_a", "apical_o_distance_mean_a",
                                              "has_apical_oxygen", "in_plane_cu_cu_mean_a")}


def default_table() -> Path:
    for path in (ROOT / "data/raw/3dsc/MP/3DSC_MP.csv",
                 ROOT / "data/raw/3dsc/repo/superconductors_3D/data/final/MP/3DSC_MP.csv"):
        if path.is_file():
            return path
    return ROOT / "data/raw/3dsc/MP/3DSC_MP.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=default_table(), help="3DSC_MP.csv")
    parser.add_argument("--cif-root", type=Path, action="append", default=None,
                        help="folder the table's cif paths are relative to (repeatable)")
    parser.add_argument("--overrides", type=Path, default=ROOT / "data/training/cuprate-parent-overrides",
                        help="folder with rules.json and replacement CIFs; pass an empty folder to disable")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/cuprate-features.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/cuprate-features.json")
    args = parser.parse_args()
    roots = args.cif_root or [ROOT / "data/raw/3dsc/repo/superconductors_3D", ROOT / "data/raw/3dsc/MP/cifs",
                              ROOT / "data/raw/3dsc/MP"]
    if not args.table.is_file():
        raise SystemExit(f"3DSC table missing: {args.table}\nrun: python scripts/download_training_data.py --all-cifs")
    print(json.dumps(build(args.table, roots, args.output, args.report, args.overrides), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
