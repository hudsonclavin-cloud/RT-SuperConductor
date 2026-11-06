#!/usr/bin/env python3
"""Structure, symmetry and stiffness features for Materials Project materials.

Reads data/raw/mp-structures/ (scripts/fetch_mp_structures.py) and writes one
row per material to data/candidates/mp-structure-features.csv.gz:

* symmetry (categorical): point group one-hot, lattice centering one-hot,
  centrosymmetric flag, point-group order
* lattice: volume per atom, packing fraction, reduced-cell length ratios,
  mean departure of cell angles from 90 degrees
* local geometry: nearest-neighbour distances (absolute and relative to the
  sum of atomic radii), coordination counts, share of unlike-element and
  metal-metal nearest-neighbour contacts
* connectivity: Larsen bonded dimensionality (0D molecules ... 3D network),
  one-hot
* stiffness: Voigt-Reuss-Hill bulk and shear moduli, Pugh ratio and the
  Debye temperature from them (missing for materials without elastic data)

All are computed from Materials Project's relaxed, ordered structures, not
measured sample structures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mp-structures"
POINT_GROUPS = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m",
                "4/mmm", "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
                "23", "m-3", "432", "-43m", "m-3m"]
CENTROSYMMETRIC = {"-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"}
CENTERINGS = ["P", "I", "F", "C", "A", "R"]
H_OVER_KB = 4.799243073e-11  # K s


def load_feature_table(path: Path, mp_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return a matrix aligned to ``mp_ids``; missing materials or values become NaN."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [n for n in (reader.fieldnames or []) if n != "mp_id"]
        table = {row["mp_id"]: row for row in reader}

    def value(row: dict[str, str] | None, name: str) -> float:
        try:
            return float(row[name]) if row is not None and row.get(name) not in (None, "") else float("nan")
        except ValueError:
            return float("nan")

    matrix = np.asarray([[value(table.get(mid), n) for n in names] for mid in mp_ids], dtype=float)
    return matrix, names


def debye_temperature(bulk_gpa: float, shear_gpa: float, density_g_cm3: float, volume_per_atom_a3: float
                      ) -> float | None:
    """Debye temperature (K) from isotropic elastic moduli (Anderson's method)."""
    if not all(v is not None and math.isfinite(v) and v > 0 for v in (bulk_gpa, shear_gpa, density_g_cm3,
                                                                     volume_per_atom_a3)):
        return None
    rho = density_g_cm3 * 1000.0
    v_t = math.sqrt(shear_gpa * 1e9 / rho)
    v_l = math.sqrt((bulk_gpa + 4.0 * shear_gpa / 3.0) * 1e9 / rho)
    v_m = ((2.0 / v_t ** 3 + 1.0 / v_l ** 3) / 3.0) ** (-1.0 / 3.0)
    return H_OVER_KB * (3.0 / (4.0 * math.pi * volume_per_atom_a3 * 1e-30)) ** (1.0 / 3.0) * v_m


def _radius(element) -> float:
    for attr in ("atomic_radius", "atomic_radius_calculated", "average_ionic_radius"):
        value = getattr(element, attr, None)
        if value:
            return float(value)
    return 1.5


def structure_row(doc: dict[str, Any]) -> dict[str, Any]:
    """Features for one {mp_id, structure} record."""
    warnings.filterwarnings("ignore")
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import MinimumDistanceNN
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    s = Structure.from_dict(doc["structure"])
    row: dict[str, Any] = {"mp_id": doc["mp_id"]}
    try:
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        pg = sga.get_point_group_symbol()
        centering = sga.get_space_group_symbol()[0]
        order = len(sga.get_point_group_operations())
    except Exception:  # noqa: BLE001 - spglib can fail on odd cells
        pg, centering, order = "", "", float("nan")
    for name in POINT_GROUPS:
        row[f"sym_pg_{name}"] = 1.0 if pg == name else 0.0
    for name in CENTERINGS:
        row[f"sym_centering_{name}"] = 1.0 if centering == name else 0.0
    row["sym_centrosymmetric"] = 1.0 if pg in CENTROSYMMETRIC else 0.0 if pg else float("nan")
    row["sym_point_group_order"] = order

    n = len(s)
    volume_per_atom = s.volume / n
    radii = [_radius(site.specie) for site in s]
    row["lat_volume_per_atom_a3"] = volume_per_atom
    row["lat_packing_fraction"] = sum(4.0 / 3.0 * math.pi * r ** 3 for r in radii) / s.volume
    lengths = sorted(s.lattice.get_niggli_reduced_lattice().abc)
    row["lat_ratio_mid_min"] = lengths[1] / lengths[0]
    row["lat_ratio_max_min"] = lengths[2] / lengths[0]
    row["lat_angle_departure_deg"] = float(np.mean([abs(a - 90.0) for a in s.lattice.angles]))

    neighbors = s.get_all_neighbors(6.0)
    nn, rel, cn, unlike, metal_metal, contacts = [], [], [], 0, 0, 0
    for i, site_neighbors in enumerate(neighbors):
        if not site_neighbors:
            continue
        d_min = min(nb.nn_distance for nb in site_neighbors)
        nn.append(d_min)
        shell = [nb for nb in site_neighbors if nb.nn_distance <= 1.2 * d_min]
        cn.append(len(shell))
        for nb in shell:
            contacts += 1
            rel.append(nb.nn_distance / (radii[i] + _radius(nb.specie)))
            unlike += nb.specie != s[i].specie
            metal_metal += bool(s[i].specie.is_metal and nb.specie.is_metal)
    row["geo_nn_distance_min_a"] = min(nn) if nn else float("nan")
    row["geo_nn_distance_mean_a"] = float(np.mean(nn)) if nn else float("nan")
    row["geo_nn_distance_std_a"] = float(np.std(nn)) if nn else float("nan")
    row["geo_bond_to_radii_mean"] = float(np.mean(rel)) if rel else float("nan")
    row["geo_bond_to_radii_min"] = float(np.min(rel)) if rel else float("nan")
    row["geo_coordination_mean"] = float(np.mean(cn)) if cn else float("nan")
    row["geo_coordination_std"] = float(np.std(cn)) if cn else float("nan")
    row["geo_coordination_max"] = float(np.max(cn)) if cn else float("nan")
    row["geo_unlike_contact_share"] = unlike / contacts if contacts else float("nan")
    row["geo_metal_metal_contact_share"] = metal_metal / contacts if contacts else float("nan")

    try:
        graph = StructureGraph.from_local_env_strategy(s, MinimumDistanceNN(tol=0.15, cutoff=6.0))
        dim = int(get_dimensionality_larsen(graph))
    except Exception:  # noqa: BLE001
        dim = -1
    for d in range(4):
        row[f"conn_dimensionality_{d}"] = 1.0 if dim == d else 0.0 if dim >= 0 else float("nan")
    row["conn_dimensionality"] = dim if dim >= 0 else float("nan")
    row["_density_g_cm3"] = s.density
    return row


def add_stiffness(rows: list[dict[str, Any]], elasticity: Path) -> None:
    moduli: dict[str, tuple[float, float]] = {}
    if elasticity.is_file():
        with elasticity.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                try:
                    moduli[r["mp_id"]] = (float(r["bulk_modulus_vrh_gpa"]), float(r["shear_modulus_vrh_gpa"]))
                except ValueError:
                    continue
    for row in rows:
        density = row.pop("_density_g_cm3")
        bulk, shear = moduli.get(row["mp_id"], (float("nan"), float("nan")))
        valid = math.isfinite(bulk) and math.isfinite(shear) and bulk > 0 and shear > 0
        theta = debye_temperature(bulk, shear, density, row["lat_volume_per_atom_a3"]) if valid else None
        row["stiff_bulk_modulus_gpa"] = bulk if valid else float("nan")
        row["stiff_shear_modulus_gpa"] = shear if valid else float("nan")
        row["stiff_pugh_ratio"] = shear / bulk if valid else float("nan")
        row["stiff_debye_temperature_k"] = theta if theta is not None else float("nan")
        row["stiff_has_elastic_data"] = 1.0 if valid else 0.0


def compute(raw_dir: Path, out: Path, workers: int = 4) -> dict[str, Any]:
    with gzip.open(raw_dir / "structures.jsonl.gz", "rt", encoding="utf-8") as handle:
        docs = [json.loads(line) for line in handle]
    with Pool(workers) as pool:
        rows = pool.map(structure_row, docs, chunksize=200)
    add_stiffness(rows, raw_dir / "elasticity.csv")
    rows.sort(key=lambda r: r["mp_id"])
    fields = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v)
                             else (round(v, 6) if isinstance(v, float) else v)) for k, v in row.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buffer.getvalue().encode("utf-8"))
    return {"materials": len(rows), "features": len(fields) - 1,
            "with_elastic_data": int(sum(r["stiff_has_elastic_data"] for r in rows)),
            "dimensionality_counts": {str(d): int(sum(r["conn_dimensionality"] == d for r in rows))
                                      for d in range(4)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=ROOT / "data/candidates/mp-structure-features.csv.gz")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(compute(args.raw_dir, args.output, args.workers), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
