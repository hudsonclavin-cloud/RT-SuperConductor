#!/usr/bin/env python3
"""Create an honest execution manifest for density and orbital analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ANALYSES = ("qe_scf", "qe_nscf", "bader", "elf", "cohp", "wannier")


def prepare(input_path: Path, output_path: Path) -> dict[str, object]:
    manifest_path = input_path / "structure-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_rows: list[dict[str, str]] = []
    for item in payload.get("entries", []):
        retrieved = item.get("structure_status") == "retrieved"
        status = "pending_external_dft" if retrieved else "blocked_no_real_structure"
        row = {
            "record_id": str(item.get("record_id", "")),
            "formula": str(item.get("formula", "")),
            "mp_id": str(item.get("mp_id", "")),
            "cif_path": str(item.get("cif_path", "")),
            "structure_status": str(item.get("structure_status", "")),
            "analysis_status": status,
            "required_inputs": (
                "relaxed CIF; pseudopotentials; converged QE SCF/NSCF; charge density; wavefunctions; k-point convergence"
                if retrieved else "real relaxed crystal structure and a valid Materials Project or experimental structure ID"
            ),
        }
        for analysis in ANALYSES:
            row[f"{analysis}_status"] = status
        output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id", "formula", "mp_id", "cif_path", "structure_status",
        *[f"{analysis}_status" for analysis in ANALYSES],
        "analysis_status", "required_inputs",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "input_manifest": str(manifest_path),
        "output": str(output_path),
        "record_count": len(output_rows),
        "pending_external_dft_count": sum(row["analysis_status"] == "pending_external_dft" for row in output_rows),
        "blocked_count": sum(row["analysis_status"] == "blocked_no_real_structure" for row in output_rows),
        "analysis_definitions": {
            "bader": "topological partition of converged charge density into atomic basins",
            "elf": "electron localization function from a converged electronic calculation",
            "cohp": "crystal orbital Hamilton population from compatible wavefunctions and LOBSTER",
            "wannier": "localized orbitals and optionally electron-phonon interpolation using Wannier90/EPW",
        },
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/structure-analysis/structures"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/structure-analysis/electronic-analysis-manifest.csv"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
