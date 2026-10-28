#!/usr/bin/env python3
"""Generate Quantum ESPRESSO and EPW starter inputs for ML-screened candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
ATOMIC_MASSES = {
    "H": 1.008, "B": 10.81, "C": 12.011, "O": 15.999, "S": 32.06,
    "Fe": 55.845, "Se": 78.971, "Cu": 63.546, "Ba": 137.327, "Ca": 40.078,
    "Y": 88.906, "Hg": 200.592, "Bi": 208.980, "Sr": 87.62, "As": 74.922,
    "K": 39.098, "Sm": 150.36, "La": 138.905, "Mg": 24.305, "Nb": 92.906,
    "Sn": 118.710, "Si": 28.085,
}


def atoms_from_formula(formula: str) -> list[tuple[str, float]]:
    """Expand a simple normalized formula into element/count pairs."""

    atoms: list[tuple[str, float]] = []
    position = 0
    for match in ELEMENT_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"unsupported formula syntax in {formula!r}")
        count = float(match.group(2) or 1.0)
        if count <= 0 or count != int(count):
            raise ValueError(f"QE generator requires integer stoichiometry: {formula!r}")
        atoms.append((match.group(1), count))
        position = match.end()
    if not atoms or position != len(formula):
        raise ValueError(f"unsupported formula syntax in {formula!r}")
    return atoms


def _expanded_atoms(formula: str) -> list[str]:
    return [element for element, count in atoms_from_formula(formula) for _ in range(int(count))]


def render_scf(candidate_id: str, formula: str) -> str:
    elements = atoms_from_formula(formula)
    expanded = _expanded_atoms(formula)
    unique_elements = [element for element, _ in elements]
    positions = []
    for index, element in enumerate(expanded):
        # Deterministic non-overlapping fractional seed positions. Replace with
        # a relaxed MP/CIF structure before production electron-phonon work.
        positions.append(f" {element:<3} {((index * 0.6180339887) % 1):.8f} {((index * 0.4142135623) % 1):.8f} {((index * 0.2718281828) % 1):.8f}")
    species = [
        f" {element:<3} {ATOMIC_MASSES.get(element, 1.0):.5f} {element}.UPF"
        for element in unique_elements
    ]
    return f"""! Candidate {candidate_id}; ML screen passed (hull < 0.05 eV/atom, predicted Tc > 77 K)
! PLACEHOLDER STRUCTURE: replace cell and fractional positions with a relaxed structure.
&CONTROL
  calculation = 'scf',
  prefix = '{candidate_id}',
  pseudo_dir = './pseudo',
  outdir = './tmp',
  tstress = .true.,
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 0,
  nat = {len(expanded)},
  ntyp = {len(unique_elements)},
  ecutwfc = 80.0,
  ecutrho = 640.0,
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.02,
/
&ELECTRONS
  conv_thr = 1.0d-10,
  electron_maxstep = 200,
  mixing_beta = 0.7,
/
ATOMIC_SPECIES
{chr(10).join(species)}
CELL_PARAMETERS angstrom
  3.000000  0.000000  0.000000
  0.000000  3.000000  0.000000
  0.000000  0.000000  3.000000
ATOMIC_POSITIONS crystal
{chr(10).join(positions)}
K_POINTS automatic
  8 8 8 0 0 0
"""


def render_ph(candidate_id: str) -> str:
    return f"""&INPUTPH
  prefix = '{candidate_id}',
  outdir = './tmp',
  tr2_ph = 1.0d-14,
  ldisp = .true.,
  nq1 = 4, nq2 = 4, nq3 = 4,
  fildyn = '{candidate_id}.dyn',
/
"""


def render_epw(candidate_id: str) -> str:
    return f"""&inputepw
  prefix = '{candidate_id}',
  outdir = './tmp',
  amass(1) = 1.0,
  elph = .true.,
  wannierize = .true.,
  epbwrite = .true.,
  epbread = .false.,
  dvscf_dir = './tmp/_ph0/{candidate_id}.phsave',
  filkf = '{candidate_id}.ephmat',
  fildvscf = '{candidate_id}.dvscf',
  nk1 = 8, nk2 = 8, nk3 = 8,
  nq1 = 4, nq2 = 4, nq3 = 4,
/
"""


def generate_inputs(candidates_path: Path, output_dir: Path) -> dict[str, object]:
    with candidates_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "formula", "predicted_tc_kelvin", "energy_above_hull"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{candidates_path}: candidate CSV must include {sorted(required)}")
    selected = [
        row for row in rows
        if row.get("review_status", "") != "excluded"
        and row.get("energy_above_hull", "").strip()
        and float(row["energy_above_hull"]) < 0.05
        and float(row["predicted_tc_kelvin"]) > 77.0
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    for row in selected:
        candidate_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row["record_id"])
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        scf_text = render_scf(candidate_id, row["formula"])
        (candidate_dir / "scf.in").write_text(scf_text, encoding="utf-8")
        (candidate_dir / "scf_template.in").write_text(scf_text, encoding="utf-8")
        (candidate_dir / "ph.in").write_text(render_ph(candidate_id), encoding="utf-8")
        (candidate_dir / "epw.in").write_text(render_epw(candidate_id), encoding="utf-8")
        metadata = {
            "record_id": row["record_id"],
            "formula": row["formula"],
            "predicted_tc_kelvin": float(row["predicted_tc_kelvin"]),
            "energy_above_hull": float(row["energy_above_hull"]),
            "filter": {"energy_above_hull_lt_eV_per_atom": 0.05, "predicted_tc_gt_kelvin": 77.0},
            "structure_status": "placeholder_fractional_positions_require_relaxation",
            "files": ["scf.in", "scf_template.in", "ph.in", "epw.in"],
        }
        (candidate_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        generated.append(metadata)
    summary = {"candidate_input": str(candidates_path), "output_dir": str(output_dir), "selected_count": len(selected), "candidates": generated}
    (output_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/pressure-aware/candidate_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/qe-inputs"))
    args = parser.parse_args()
    print(json.dumps(generate_inputs(args.candidates, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
