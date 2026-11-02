#!/usr/bin/env python3
"""Generate blocked Quantum ESPRESSO/EPW inputs for screened candidates.

The generator can preserve the legacy formula-only placeholder mode for smoke
tests. Ferrite production hand-off uses ``--require-real-cif`` so that input
coordinates come from a real CIF. No DFT is executed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
ATOMIC_MASSES = {
    "H": 1.008, "B": 10.81, "C": 12.011, "O": 15.999, "S": 32.06,
    "Fe": 55.845, "Se": 78.971, "Cu": 63.546, "Ba": 137.327, "Ca": 40.078,
    "Y": 88.906, "Hg": 200.592, "Bi": 208.980, "Sr": 87.62, "As": 74.922,
    "K": 39.098, "Sm": 150.36, "La": 138.905, "Mg": 24.305, "Nb": 92.906,
    "Sn": 118.710, "Si": 28.085,
}
DEFAULT_PSEUDOPOTENTIAL_CATALOG = Path("data/pseudopotentials/pslibrary-catalog.csv")


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


def _catalog_names(catalog_path: Path | None) -> dict[str, str]:
    if catalog_path is None or not catalog_path.is_file():
        return {}
    with catalog_path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row.get("element", "").strip(): (row.get("source_pattern", "").strip() or "PLACEHOLDER.UPF")
            for row in csv.DictReader(handle)
            if row.get("element", "").strip()
        }


def _pseudo_filename(element: str, catalog: dict[str, str]) -> str:
    return catalog.get(element, f"{element}.UPF").replace("$fct", "PBE").replace("*", "PLACEHOLDER")


def _structure_blocks(structure: Any, catalog: dict[str, str]) -> tuple[list[str], list[str], list[str], list[str]]:
    unique_elements = [str(element) for element in structure.composition.elements]
    species = [
        f" {element:<3} {ATOMIC_MASSES.get(element, 1.0):.5f} {_pseudo_filename(element, catalog)}"
        for element in unique_elements
    ]
    positions = [
        f" {site.specie.symbol:<3} {site.coords[0]:.8f} {site.coords[1]:.8f} {site.coords[2]:.8f}"
        for site in structure.sites
    ]
    cell = [
        f"  {vector[0]:.8f}  {vector[1]:.8f}  {vector[2]:.8f}"
        for vector in structure.lattice.matrix
    ]
    return unique_elements, species, positions, cell


def render_scf(
    candidate_id: str,
    formula: str,
    *,
    structure: Any | None = None,
    pseudopotential_catalog: dict[str, str] | None = None,
    calculation: str = "scf",
) -> str:
    elements = atoms_from_formula(formula)
    pseudopotential_catalog = pseudopotential_catalog or {}
    if structure is None:
        expanded = _expanded_atoms(formula)
        unique_elements = [element for element, _ in elements]
        positions = [
            f" {element:<3} {((index * 0.6180339887) % 1):.8f} {((index * 0.4142135623) % 1):.8f} {((index * 0.2718281828) % 1):.8f}"
            for index, element in enumerate(expanded)
        ]
        species = [
            f" {element:<3} {ATOMIC_MASSES.get(element, 1.0):.5f} {_pseudo_filename(element, pseudopotential_catalog)}"
            for element in unique_elements
        ]
        cell = [
            "  3.000000  0.000000  0.000000",
            "  0.000000  3.000000  0.000000",
            "  0.000000  0.000000  3.000000",
        ]
        positions_label = "crystal"
        structure_note = "! PLACEHOLDER STRUCTURE: replace cell and positions with a relaxed structure."
    else:
        unique_elements, species, positions, cell = _structure_blocks(structure, pseudopotential_catalog)
        positions_label = "angstrom"
        structure_note = "! STRUCTURE SOURCE: real CIF; relax this reference structure before electron-phonon work."
    return f"""! Candidate {candidate_id}; selected for a reviewable DFT hand-off
{structure_note}
&CONTROL
  calculation = '{calculation}',
  prefix = '{candidate_id}',
  pseudo_dir = './pseudo',
  outdir = './tmp',
  tstress = .true.,
  tprnfor = .true.,
  forc_conv_thr = 1.0d-5,
  etot_conv_thr = 1.0d-6,
  ion_dynamics = 'bfgs',
/
&SYSTEM
  ibrav = 0,
  nat = {len(positions)},
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
{chr(10).join(cell)}
ATOMIC_POSITIONS {positions_label}
{chr(10).join(positions)}
K_POINTS automatic
  8 8 8 0 0 0
"""


def render_relax(candidate_id: str, formula: str, *, structure: Any | None = None, pseudopotential_catalog: dict[str, str] | None = None) -> str:
    return render_scf(candidate_id, formula, structure=structure, pseudopotential_catalog=pseudopotential_catalog, calculation="relax")


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


def _structure_path(row: dict[str, str], structure_dir: Path | None, manifest: dict[str, str]) -> Path | None:
    candidates: list[Path] = []
    if row.get("cif_path"):
        candidates.append(Path(row["cif_path"]))
    if row.get("mp_id"):
        candidates.extend([Path(row["mp_id"] + ".cif"), Path(row["mp_id"] + ".CIF")])
    if row.get("candidate_id"):
        candidates.append(Path(row["candidate_id"] + ".cif"))
    if row.get("record_id"):
        candidates.append(Path(row["record_id"] + ".cif"))
    if row.get("formula"):
        candidates.append(Path(re.sub(r"[^A-Za-z0-9_.-]", "_", row["formula"]) + ".cif"))
    if row.get("mp_id") and row["mp_id"] in manifest:
        candidates.insert(0, Path(manifest[row["mp_id"]]))
    for path in candidates:
        resolved = path if path.is_absolute() else (structure_dir / path if structure_dir else path)
        if resolved.is_file():
            return resolved
    return None


def generate_inputs(
    candidates_path: Path,
    output_dir: Path,
    *,
    structure_dir: Path | None = None,
    structure_manifest: Path | None = None,
    pseudopotential_catalog: Path = DEFAULT_PSEUDOPOTENTIAL_CATALOG,
    allow_placeholder: bool = True,
    min_predicted_tc: float = 77.0,
) -> dict[str, object]:
    with candidates_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    required = {"formula", "predicted_tc_kelvin", "energy_above_hull"}
    if not required.issubset(fieldnames) or not ("record_id" in fieldnames or "candidate_id" in fieldnames):
        raise ValueError(f"{candidates_path}: candidate CSV must include {sorted(required)} and record_id or candidate_id")
    manifest: dict[str, str] = {}
    if structure_manifest and structure_manifest.is_file():
        payload = json.loads(structure_manifest.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []):
            if entry.get("mp_id") and entry.get("cif_path"):
                manifest[str(entry["mp_id"])] = str(entry["cif_path"])
    catalog = _catalog_names(pseudopotential_catalog)
    selected = [
        row for row in rows
        if row.get("review_status", "") != "excluded"
        and row.get("energy_above_hull", "").strip()
        and float(row["energy_above_hull"]) < 0.05
        and float(row["predicted_tc_kelvin"]) > min_predicted_tc
        and row.get("is_metal", "true").lower() not in {"false", "0"}
        and not (row.get("band_gap_ev", "").strip() and float(row["band_gap_ev"]) > 0.1)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    for row in selected:
        candidate_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row.get("record_id") or row.get("candidate_id") or "candidate")
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        cif_path = _structure_path(row, structure_dir, manifest)
        structure = None
        if cif_path is not None:
            try:
                from pymatgen.core import Structure
                structure = Structure.from_file(cif_path)
            except Exception as exc:
                blocked.append({"candidate_id": candidate_id, "formula": row["formula"], "status": "blocked_invalid_cif", "reason": f"{type(exc).__name__}: {exc}"})
                continue
        if structure is None and not allow_placeholder:
            blocked.append({"candidate_id": candidate_id, "formula": row["formula"], "status": "blocked_missing_real_cif", "reason": "a real Materials Project CIF is required for ferrite DFT input generation"})
            continue
        scf_text = render_scf(candidate_id, row["formula"], structure=structure, pseudopotential_catalog=catalog)
        (candidate_dir / "relax.in").write_text(render_relax(candidate_id, row["formula"], structure=structure, pseudopotential_catalog=catalog), encoding="utf-8")
        (candidate_dir / "scf.in").write_text(scf_text, encoding="utf-8")
        (candidate_dir / "scf_template.in").write_text(scf_text, encoding="utf-8")
        (candidate_dir / "ph.in").write_text(render_ph(candidate_id), encoding="utf-8")
        (candidate_dir / "epw.in").write_text(render_epw(candidate_id), encoding="utf-8")
        metadata = {
            "record_id": row.get("record_id") or row.get("candidate_id"),
            "formula": row["formula"],
            "predicted_tc_kelvin": float(row["predicted_tc_kelvin"]),
            "energy_above_hull": float(row["energy_above_hull"]),
            "filter": {"energy_above_hull_lt_eV_per_atom": 0.05, "predicted_tc_gt_kelvin": min_predicted_tc, "metallic_only": True},
            "structure_status": "real_cif" if structure is not None else "placeholder_fractional_positions_require_relaxation",
            "cif_path": str(cif_path) if cif_path is not None else None,
            "pseudopotential_status": "catalog_only_no_upf_files_verified",
            "job_status": "blocked_missing_pseudopotentials_and_hazel_access",
            "files": ["relax.in", "scf.in", "scf_template.in", "ph.in", "epw.in"],
        }
        (candidate_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        generated.append(metadata)
    summary = {
        "candidate_input": str(candidates_path),
        "output_dir": str(output_dir),
        "selected_count": len(selected),
        "generated_count": len(generated),
        "blocked_count": len(blocked),
        "status": "ready_with_blocked_jobs" if generated else "blocked_no_real_cif_or_candidates",
        "blocked_reason": "No DFT was run; pseudopotential UPF files and Hazel access are not available in this repository.",
        "pseudopotential_catalog": str(pseudopotential_catalog),
        "candidates": generated,
        "blocked": blocked,
    }
    (output_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/pressure-aware/candidate_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/qe-inputs"))
    parser.add_argument("--structure-dir", type=Path)
    parser.add_argument("--structure-manifest", type=Path)
    parser.add_argument("--pseudopotential-catalog", type=Path, default=DEFAULT_PSEUDOPOTENTIAL_CATALOG)
    parser.add_argument("--require-real-cif", action="store_true")
    parser.add_argument("--min-predicted-tc", type=float, default=77.0)
    args = parser.parse_args()
    print(json.dumps(generate_inputs(
        args.candidates,
        args.output_dir,
        structure_dir=args.structure_dir,
        structure_manifest=args.structure_manifest,
        pseudopotential_catalog=args.pseudopotential_catalog,
        allow_placeholder=not args.require_real_cif,
        min_predicted_tc=args.min_predicted_tc,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
