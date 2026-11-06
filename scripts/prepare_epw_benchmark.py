#!/usr/bin/env python3
"""Prepare a blocked Quantum ESPRESSO/EPW benchmark hand-off for Nb3Sn.

This script retrieves a composition-validated Materials Project reference
structure and writes input templates. It never runs Quantum ESPRESSO or EPW.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from generate_qe_inputs import (  # noqa: E402
    render_epw,
    render_matdyn,
    render_nscf,
    render_ph,
    render_q2r,
    render_relax,
    render_scf,
    render_vc_relax,
)
from rt_superconductor.materials_project import normalize_fixed_formula  # noqa: E402


FORMULA = "Nb3Sn"
MP_QUERY = "Nb3Sn"
LITERATURE_TC_K = 18.0
LITERATURE_DOI = "10.1103/PhysRev.95.1435"


def _entry_formula(document: Any) -> str:
    value = getattr(document, "formula_pretty", None)
    if value:
        return str(value)
    composition = getattr(document, "composition_reduced", None)
    return str(composition) if composition is not None else ""


def _choose_entry(documents: list[Any]) -> Any:
    target = normalize_fixed_formula(FORMULA)
    exact = [
        document for document in documents
        if normalize_fixed_formula(_entry_formula(document)).signature == target.signature
    ]
    if not exact:
        raise ValueError("Materials Project returned no exact Nb3Sn composition")
    finite_hull = [
        document for document in exact
        if getattr(document, "energy_above_hull", None) is not None
    ]
    return min(
        finite_hull or exact,
        key=lambda document: (
            float(getattr(document, "energy_above_hull", float("inf")) or 0.0)
            if finite_hull else str(getattr(document, "material_id", ""))
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_pseudopotential_path(text: str, pseudo_dir: Path, output_dir: Path) -> str:
    relative = os.path.relpath(pseudo_dir.resolve(), output_dir.resolve())
    return text.replace("pseudo_dir = './pseudo'", f"pseudo_dir = '{relative}'")


def _write_blocked_report(output_dir: Path, status: str, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "formula": FORMULA,
        "structure_source": "Materials Project",
        "blocked_reason": reason,
        "input_files_written": [],
        "dft_executed": False,
    }
    (output_dir / "generation-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def prepare_epw_benchmark(
    *,
    output_dir: Path = Path("data/processed/epw-benchmark/nb3sn"),
    api_key: str | None = None,
    pseudopotential_dir: Path = Path("data/pseudopotentials/upf"),
    hazel_access_confirmed: bool = False,
    rester_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Write reference-structure-based QE/EPW inputs and prerequisite status."""

    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        return _write_blocked_report(
            output_dir, "blocked_missing_materials_project_key", "MP_API_KEY is unavailable"
        )
    if rester_factory is None:
        try:
            from mp_api.client import MPRester
        except ImportError:
            return _write_blocked_report(
                output_dir, "blocked_missing_mp_api_client", "Install the project mp optional dependencies"
            )
        rester_factory = MPRester

    try:
        with rester_factory(api_key) as mpr:
            documents = mpr.materials.summary.search(
                formula=MP_QUERY,
                fields=["material_id", "formula_pretty", "composition_reduced", "energy_above_hull"],
            )
            selected = _choose_entry(list(documents or []))
            material_id = str(selected.material_id)
            structure = mpr.get_structure_by_material_id(material_id)
    except Exception as exc:
        return _write_blocked_report(
            output_dir,
            "blocked_structure_retrieval",
            f"Materials Project retrieval failed ({type(exc).__name__}); verify network and API access",
        )

    structure_formula = normalize_fixed_formula(structure.composition.reduced_formula)
    target_formula = normalize_fixed_formula(FORMULA)
    if structure_formula.signature != target_formula.signature:
        return _write_blocked_report(
            output_dir, "blocked_structure_composition_mismatch", "Retrieved structure is not composition-matched Nb3Sn"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Explicit placeholder filenames prevent a generator-family pattern from
    # being mistaken for an actual UPF pseudopotential.
    expected_pseudos = {"Nb": "Nb.UPF", "Sn": "Sn.UPF"}
    pseudo_status = {
        element: {
            "filename": filename,
            "path": str(pseudopotential_dir / filename),
            "present": (pseudopotential_dir / filename).is_file(),
            "sha256": _sha256(pseudopotential_dir / filename)
            if (pseudopotential_dir / filename).is_file() else None,
        }
        for element, filename in expected_pseudos.items()
    }

    cif_path = output_dir / "nb3sn-mp-reference.cif"
    cif_path.write_text(structure.to(fmt="cif"), encoding="utf-8")
    (output_dir / "relax.in").write_text(
        f"! BLOCKED Nb3Sn benchmark template; reference structure from Materials Project {material_id}\n"
        + _set_pseudopotential_path(
            render_relax("nb3sn", FORMULA, structure=structure, pseudopotential_catalog=expected_pseudos),
            pseudopotential_dir, output_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "vc-relax.in").write_text(
        f"! BLOCKED variable-cell relaxation template; review target pressure and constraints first.\n"
        + _set_pseudopotential_path(
            render_vc_relax("nb3sn", FORMULA, structure=structure, pseudopotential_catalog=expected_pseudos),
            pseudopotential_dir, output_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "scf.in").write_text(
        "! BLOCKED benchmark template; use a converged relaxed structure before production SCF.\n"
        + _set_pseudopotential_path(
            render_scf("nb3sn", FORMULA, structure=structure, pseudopotential_catalog=expected_pseudos),
            pseudopotential_dir, output_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "ph.in").write_text(
        "! BLOCKED benchmark template; converge the q mesh and verify the relaxed structure first.\n"
        + render_ph("nb3sn"),
        encoding="utf-8",
    )
    (output_dir / "nscf.in").write_text(
        "! BLOCKED NSCF template; run after a reviewed converged SCF calculation.\n"
        + _set_pseudopotential_path(
            render_nscf("nb3sn", FORMULA, structure=structure, pseudopotential_catalog=expected_pseudos),
            pseudopotential_dir, output_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "q2r.in").write_text("! BLOCKED q2r template; requires converged phonon dynamical matrices.\n" + render_q2r("nb3sn"), encoding="utf-8")
    (output_dir / "matdyn.in").write_text("! BLOCKED matdyn template; review q mesh and acoustic sum rule.\n" + render_matdyn("nb3sn"), encoding="utf-8")
    (output_dir / "epw.in").write_text(
        "! BLOCKED benchmark template; confirm Wannier windows, projections, and interpolation meshes with an EPW user.\n"
        + render_epw("nb3sn"),
        encoding="utf-8",
    )

    blocked_reasons = []
    if any(not item["present"] for item in pseudo_status.values()):
        blocked_reasons.append("verified Nb and Sn UPF pseudopotential files are missing")
    if not hazel_access_confirmed:
        blocked_reasons.append("Hazel compute access/allocation has not been confirmed")
    manifest = {
        "benchmark": "Nb3Sn electron-phonon superconductivity reproduction",
        "formula": FORMULA,
        "materials_project_id": material_id,
        "materials_project_formula": _entry_formula(selected),
        "materials_project_energy_above_hull_ev_per_atom": getattr(selected, "energy_above_hull", None),
        "structure_source": "Materials Project computed reference structure; not experimental proof",
        "structure_file": str(cif_path),
        "structure_sha256": _sha256(cif_path),
        "pseudopotentials": pseudo_status,
        "hazel_access_confirmed_by_operator": bool(hazel_access_confirmed),
        "literature_target_tc_k": LITERATURE_TC_K,
        "literature_target_doi": LITERATURE_DOI,
        "status": "blocked_prerequisites" if blocked_reasons else "prepared_for_manual_review",
        "blocked_reasons": blocked_reasons,
        "dft_executed": False,
        "inputs_are_templates": True,
    }
    (output_dir / "benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    input_names = ["relax.in", "vc-relax.in", "scf.in", "nscf.in", "ph.in", "q2r.in", "matdyn.in", "epw.in"]
    report = {
        **manifest,
        "output_dir": str(output_dir),
        "input_files_written": input_names,
        "input_sha256": {name: _sha256(output_dir / name) for name in input_names},
    }
    (output_dir / "generation-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/epw-benchmark/nb3sn"))
    parser.add_argument("--pseudopotential-dir", type=Path, default=Path("data/pseudopotentials/upf"))
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    parser.add_argument(
        "--hazel-access-confirmed", action="store_true",
        help="operator confirms a valid Hazel allocation/access path; this does not run calculations",
    )
    args = parser.parse_args()
    report = prepare_epw_benchmark(
        output_dir=args.output_dir,
        api_key=args.api_key,
        pseudopotential_dir=args.pseudopotential_dir,
        hazel_access_confirmed=args.hazel_access_confirmed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
