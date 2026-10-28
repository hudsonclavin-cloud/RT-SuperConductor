#!/usr/bin/env python3
"""Fetch Materials Project CIF structures for enriched RT-SuperConductor rows.

This command retrieves reference crystal structures only.  It does not calculate
charge density, ELF, COHP, or Wannier functions.  Those analyses require a
relaxed structure, converged electronic-structure calculations, and external
codes described in ``docs/electronic-density.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "record"


def _get_structure(client: Any, mp_id: str) -> Any:
    """Support both current and older mp-api client layouts."""

    getter = getattr(client, "get_structure_by_material_id", None)
    if getter is not None:
        return getter(mp_id)
    materials = getattr(client, "materials", None)
    getter = getattr(materials, "get_structure_by_material_id", None)
    if getter is not None:
        return getter(mp_id)
    raise RuntimeError("installed mp-api client has no structure retrieval method")


def _make_client(api_key: str | None):
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("mp-api is required; install `pip install -e '.[mp]'`") from exc
    if not api_key:
        raise RuntimeError("Materials Project API key is required via --api-key or MP_API_KEY")
    return MPRester(api_key)


def fetch_structures(
    input_path: Path,
    output_dir: Path,
    *,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, object]:
    """Write one CIF and one machine-readable manifest entry per real MP id."""

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{input_path}: no rows found")
    required = {"record_id", "formula", "mp_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{input_path}: missing columns {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    needs_client = any(
        (row.get("mp_id") or "").strip() and not (row.get("mp_id") or "").strip().startswith("mp-demo-")
        for row in rows
    )
    own_client = client is None and needs_client
    context = None
    if own_client:
        context = _make_client(api_key or os.environ.get("MP_API_KEY"))
        client = context.__enter__()
    try:
        for row in rows:
            record_id = row.get("record_id", "")
            formula = row.get("formula", "")
            mp_id = (row.get("mp_id") or "").strip()
            entry: dict[str, object] = {
                "record_id": record_id,
                "formula": formula,
                "mp_id": mp_id,
                "source": "Materials Project",
                "source_url": f"https://materialsproject.org/materials/{mp_id}" if mp_id else "",
                "cif_path": "",
                "structure_status": "pending",
                "analysis_status": "structure_only_no_density_calculation",
            }
            if not mp_id:
                entry["structure_status"] = "missing_mp_id"
            elif mp_id.startswith("mp-demo-"):
                entry["structure_status"] = "synthetic_demo_id_skipped"
            else:
                try:
                    structure = _get_structure(client, mp_id)
                    filename = f"{_safe_name(record_id)}__{_safe_name(mp_id)}.cif"
                    cif_path = output_dir / filename
                    try:
                        from pymatgen.io.cif import CifWriter
                    except ImportError as exc:
                        raise RuntimeError("pymatgen is required to write CIF files") from exc
                    CifWriter(structure).write_file(cif_path)
                    entry["cif_path"] = str(cif_path)
                    entry["structure_status"] = "retrieved"
                except Exception as exc:  # preserve other candidates and report failures
                    entry["structure_status"] = "retrieval_error"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            manifest.append(entry)
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    manifest_path = output_dir / "structure-manifest.json"
    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "record_count": len(manifest),
        "retrieved_count": sum(item["structure_status"] == "retrieved" for item in manifest),
        "entries": manifest,
        "calculation_boundary": "CIF retrieval only; no Bader, ELF, COHP, or Wannier calculation was run",
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/structure-analysis/harmonized_enriched.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/structure-analysis/structures"))
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    args = parser.parse_args()
    report = fetch_structures(args.input, args.output_dir, api_key=args.api_key)
    print(json.dumps(report, indent=2, sort_keys=True))
    # An empty retrieval is still a useful, reviewable artifact: it tells the
    # researcher that the API returned no usable structures for this snapshot.
    # Do not turn that data-quality result into a missing-manifest failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
