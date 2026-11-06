#!/usr/bin/env python3
"""Fetch crystal structures and elastic moduli for the Materials Project snapshot.

Same query as scripts/fetch_mp_snapshot.py. Writes:

* ``data/raw/mp-structures/structures.jsonl.gz`` - one {mp_id, structure} per
  line (pymatgen dict). About 60 MB uncompressed, so it stays out of git
  (data/raw is ignored); features computed from it are committed instead.
* ``data/raw/mp-structures/elasticity.csv`` - Voigt-Reuss-Hill bulk and shear
  moduli (GPa) for the ~7,400 materials that have elastic calculations.

Uses MP_API_KEY like fetch_mp_snapshot.py (the cloud proxy injects it).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_mp_snapshot import API, HULL_MAX_EV_PER_ATOM, MAX_SITES, PAGE, http_fetch  # noqa: E402

OUT = ROOT / "data/raw/mp-structures"


def url(fields: list[str], skip: int, **extra: str) -> str:
    params = {"energy_above_hull_min": 0, "energy_above_hull_max": HULL_MAX_EV_PER_ATOM,
              "nsites_min": 1, "nsites_max": MAX_SITES, "deprecated": "false",
              "_fields": ",".join(fields), "_limit": PAGE, "_skip": skip, **extra}
    return API + "?" + urllib.parse.urlencode(params)


def paged(fields: list[str], **extra: str):
    first = http_fetch(url(fields, 0, **extra))
    total = int(first["meta"]["total_doc"])
    docs = list(first["data"])
    while len(docs) < total:
        page = http_fetch(url(fields, len(docs), **extra))["data"]
        if not page:
            break
        docs.extend(page)
        print(f"{fields[-1]}: {len(docs)}/{total}", flush=True)
    if len(docs) != total or len({d["material_id"] for d in docs}) != total:
        raise RuntimeError(f"incomplete or duplicated page set: {len(docs)} of {total}")
    return docs


def _vrh(value):
    if isinstance(value, dict):
        return value.get("vrh")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    structures = paged(["material_id", "structure"])
    with gzip.open(args.out_dir / "structures.jsonl.gz", "wt", encoding="utf-8") as handle:
        for doc in sorted(structures, key=lambda d: d["material_id"]):
            handle.write(json.dumps({"mp_id": doc["material_id"], "structure": doc["structure"]}) + "\n")
    elastic = paged(["material_id", "bulk_modulus", "shear_modulus"], has_props="elasticity")
    with (args.out_dir / "elasticity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["mp_id", "bulk_modulus_vrh_gpa", "shear_modulus_vrh_gpa"])
        for doc in sorted(elastic, key=lambda d: d["material_id"]):
            writer.writerow([doc["material_id"], _vrh(doc.get("bulk_modulus")), _vrh(doc.get("shear_modulus"))])
    print(json.dumps({"structures": len(structures), "elastic": len(elastic)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
