#!/usr/bin/env python3
"""Save a reproducible snapshot of every stable, small Materials Project material.

The untested-pool screen queries ~30,000 Materials Project summaries on every
run and keeps only its shortlist. This script saves the whole set once, as a
compressed CSV plus a manifest. Later steps can then read every material
offline, and anyone can see exactly which Materials Project release they used.

Query (same scope as scripts/build_untested_pool.py):
    energy_above_hull 0-0.025 eV/atom, 1-20 sites per cell, not deprecated.

The API key is read from MP_API_KEY and sent in the X-API-Key header. In a
Claude cloud session the network proxy injects the saved key instead, so any
placeholder value works there. Materials Project data is CC BY 4.0: cite
Jain et al., APL Materials 1, 011002 (2013).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.materialsproject.org/materials/summary/"
HULL_MAX_EV_PER_ATOM = 0.025
MAX_SITES = 20
PAGE = 1000
FIELDS = [
    "material_id", "formula_pretty", "elements", "nelements", "nsites",
    "energy_above_hull", "formation_energy_per_atom", "is_stable", "theoretical",
    "symmetry", "volume", "density",
    "is_metal", "band_gap",
    "is_magnetic", "ordering", "total_magnetization",
    "total_magnetization_normalized_formula_units", "num_magnetic_sites",
    "database_IDs",
]
COLUMNS = [
    "mp_id", "formula", "elements", "nelements", "num_sites",
    "energy_above_hull_ev_per_atom", "formation_energy_ev_per_atom", "is_stable", "theoretical",
    "icsd_ids", "crystal_system", "spacegroup_number", "spacegroup_symbol",
    "volume_a3", "density_g_cm3",
    "is_metal", "band_gap_ev",
    "is_magnetic", "ordering", "total_magnetization_mub",
    "total_magnetization_mub_per_formula_unit", "num_magnetic_sites",
]

Fetcher = Callable[[str], dict[str, Any]]


def http_fetch(url: str, *, retries: int = 4) -> dict[str, Any]:
    """GET one page of JSON, retrying transient failures."""
    key = os.environ.get("MP_API_KEY", "proxy-injected")
    # The API rejects Python's default "Python-urllib" user agent with 403.
    headers = {"X-API-Key": key, "Accept": "application/json",
               "User-Agent": "rt-superconductor-snapshot/1.0 (+https://github.com/hudsonclavin-cloud/RT-SuperConductor)"}
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - network errors vary by platform
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def page_url(skip: int, limit: int = PAGE) -> str:
    params = {
        "energy_above_hull_min": 0, "energy_above_hull_max": HULL_MAX_EV_PER_ATOM,
        "nsites_min": 1, "nsites_max": MAX_SITES, "deprecated": "false",
        "_fields": ",".join(FIELDS), "_limit": limit, "_skip": skip,
    }
    return API + "?" + urllib.parse.urlencode(params)


def flatten(doc: dict[str, Any]) -> dict[str, Any]:
    symmetry = doc.get("symmetry") or {}
    ids = (doc.get("database_IDs") or {}).get("icsd") or []
    return {
        "mp_id": doc.get("material_id"), "formula": doc.get("formula_pretty"),
        "elements": " ".join(doc.get("elements") or []), "nelements": doc.get("nelements"),
        "num_sites": doc.get("nsites"),
        "energy_above_hull_ev_per_atom": doc.get("energy_above_hull"),
        "formation_energy_ev_per_atom": doc.get("formation_energy_per_atom"),
        "is_stable": doc.get("is_stable"), "theoretical": doc.get("theoretical"),
        "icsd_ids": " ".join(ids),
        "crystal_system": symmetry.get("crystal_system"), "spacegroup_number": symmetry.get("number"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "volume_a3": doc.get("volume"), "density_g_cm3": doc.get("density"),
        "is_metal": doc.get("is_metal"), "band_gap_ev": doc.get("band_gap"),
        "is_magnetic": doc.get("is_magnetic"), "ordering": doc.get("ordering"),
        "total_magnetization_mub": doc.get("total_magnetization"),
        "total_magnetization_mub_per_formula_unit": doc.get("total_magnetization_normalized_formula_units"),
        "num_magnetic_sites": doc.get("num_magnetic_sites"),
    }


def fetch_all(fetch: Fetcher = http_fetch, *, progress: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = fetch(page_url(0))
    meta = first.get("meta", {})
    total = int(meta["total_doc"])
    rows = [flatten(d) for d in first.get("data", [])]
    while len(rows) < total:
        page = fetch(page_url(len(rows)))
        data = page.get("data", [])
        if not data:
            break
        rows.extend(flatten(d) for d in data)
        if progress:
            print(f"fetched {len(rows)}/{total}", flush=True)
    ids = [r["mp_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate material IDs across pages; the result set changed during paging")
    if len(rows) != total:
        raise RuntimeError(f"expected {total} materials, received {len(rows)}")
    rows.sort(key=lambda r: str(r["mp_id"]))
    return rows, meta


def write_snapshot(rows: list[dict[str, Any]], meta: dict[str, Any], out_csv: Path, manifest: Path) -> dict[str, Any]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode("utf-8")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes identical for identical content.
    with out_csv.open("wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    metal = sum(r["is_metal"] is True for r in rows)
    magnetic = sum(r["is_magnetic"] is True for r in rows)
    record = {
        "source": "Materials Project summary API", "api_url": API,
        "api_version": meta.get("api_version"), "retrieved_utc": meta.get("time_stamp")
        or datetime.now(timezone.utc).isoformat(),
        "query": {"energy_above_hull_ev_per_atom": [0, HULL_MAX_EV_PER_ATOM], "nsites": [1, MAX_SITES],
                  "deprecated": False, "fields": FIELDS},
        "materials": len(rows),
        "counts": {
            "is_metal_true": metal, "is_magnetic_true": magnetic,
            "metallic_and_non_magnetic": sum(r["is_metal"] is True and r["is_magnetic"] is False for r in rows),
            "has_icsd_experimental_entry": sum(bool(r["icsd_ids"]) for r in rows),
            "theoretical_only": sum(r["theoretical"] is True for r in rows),
        },
        "csv_gz": str(out_csv.relative_to(ROOT)) if out_csv.resolve().is_relative_to(ROOT) else str(out_csv),
        "csv_sha256_uncompressed": hashlib.sha256(raw).hexdigest(),
        "license": "CC BY 4.0 (Materials Project); cite Jain et al., APL Materials 1, 011002 (2013)",
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "data/candidates/mp-stable-summary.csv.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/candidates/mp-stable-summary.manifest.json")
    args = parser.parse_args()
    rows, meta = fetch_all()
    record = write_snapshot(rows, meta, args.output, args.manifest)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
