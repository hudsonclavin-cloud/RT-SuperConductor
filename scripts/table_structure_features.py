#!/usr/bin/env python3
"""Structure features for the training table's compositions (plan: docs/structure-cycle.md).

    python scripts/table_structure_features.py     # writes data/processed/structures/features.csv.gz and a count summary

Three features where a trustworthy structure exists, else missing: space-group number, crystal system (from the
space group) and volume per atom. Sources, in order of preference:
1. 3DSC (SuperCon matched to Materials Project structures): ``spacegroup_2``, ``cell_volume_2 / nsites_2``;
2. this project's COD / Materials Project matching (docs/structure-match.md), confidence high or medium only.
Compositions only are read; no label, Tc or split.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import structure_match as S  # noqa: E402

OUT = ROOT / "data/processed/structures/features.csv.gz"
SUMMARY = ROOT / "docs/table-structure-features.json"
SYSTEMS = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"), (167, "trigonal"),
           (194, "hexagonal"), (230, "cubic")]


def crystal_system(sg: int) -> str:
    return next(name for top, name in SYSTEMS if sg <= top)


_SYMBOLS: dict[str, int | None] = {}


def sg_number(value) -> int | None:
    """A space-group number from a number or a Hermann-Mauguin symbol ('I 4/m m m', as 3DSC writes it)."""
    try:
        n = int(float(value))
        return n if 1 <= n <= 230 else None
    except (TypeError, ValueError):
        pass
    text = str(value or "").replace(" ", "").split(":")[0]
    if not text:
        return None
    if text not in _SYMBOLS:
        from pymatgen.symmetry.groups import SpaceGroup
        try:
            _SYMBOLS[text] = SpaceGroup(text).int_number
        except Exception:
            _SYMBOLS[text] = None
    return _SYMBOLS[text]


def from_3dsc() -> dict[str, dict]:
    import re

    out: dict[str, dict] = {}
    with open(M.THREEDSC, encoding="utf-8") as handle:
        handle.readline()
        for row in csv.DictReader(handle):
            fractions = M.fractions(row["formula_sc"]) if row["formula_sc"] else None
            sg = sg_number(row.get("spacegroup_2"))
            try:
                vpa = float(row["cell_volume_2"]) / float(row["nsites_2"])
            except (TypeError, ValueError, ZeroDivisionError):
                vpa = None
            if fractions and sg and vpa:
                out.setdefault(B.composition_key(fractions), {"sg": sg, "vpa": vpa, "source": "3DSC"})
    return out


def from_matching(mp_volume: dict[str, float]) -> dict[str, dict]:
    out = {}
    cod_cells = {}
    for path in S.CACHE.glob("*.json"):
        for e in json.loads(path.read_text(encoding="utf-8")):
            cod_cells[e["file"]] = e
    with open(S.SUMMARY_CSV, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["confidence"] not in ("high", "medium"):
                continue
            entry = r["cod_id"] or r["parent_entry"] or r["mp_id"]
            sg, vpa = None, None
            if entry in cod_cells:
                e = cod_cells[entry]
                sg = sg_number(e.get("sgNumber"))
                fractions = S.cod_fractions(e)
                try:
                    atoms_per_fu = sum(float(x) for x in __import__("re").findall(r"[A-Z][a-z]?([\d.]*)", (e.get("calcformula") or e.get("formula") or "").replace("-", "")) for x in [x or "1"])
                    vpa = float(e["vol"]) / (float(e["Z"]) * atoms_per_fu)
                except (TypeError, ValueError, ZeroDivisionError, KeyError):
                    vpa = None
            elif entry in mp_volume:
                sg, vpa = mp_volume[entry]
            if sg and vpa and 3 < vpa < 80:
                out[r["composition_key"]] = {"sg": sg, "vpa": vpa, "source": f"match ({r['confidence']})"}
    return out


def main() -> int:
    import scratchnet_common as C

    mp_volume = {}
    with gzip.open(S.MP, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                mp_volume[row["mp_id"]] = (sg_number(row["spacegroup_number"]),
                                           float(row["volume_a3"]) / float(row["num_sites"]))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    ds = C.load_dataset()
    threedsc, matched = from_3dsc(), from_matching(mp_volume)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": len(ds.rows), "3DSC": 0, "match": 0, "none": 0}
    with gzip.open(OUT, "wt", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["composition", "sg_number", "crystal_system", "volume_per_atom", "source"])
        for row in ds.rows:
            key = row["composition"]
            f = threedsc.get(key) or matched.get(key)
            if f:
                counts["3DSC" if f["source"] == "3DSC" else "match"] += 1
                w.writerow([key, f["sg"], crystal_system(f["sg"]), round(f["vpa"], 4), f["source"]])
            else:
                counts["none"] += 1
                w.writerow([key, "", "", "", ""])
    SUMMARY.write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
