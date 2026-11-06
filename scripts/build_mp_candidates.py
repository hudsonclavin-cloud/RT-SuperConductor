#!/usr/bin/env python3
"""Turn the Materials Project snapshot into a labelled candidate table.

Input: data/candidates/mp-stable-summary.csv.gz (scripts/fetch_mp_snapshot.py).
For every material this adds:

* ``chemical_family``: Cuprates (Cu and O), Ferrites (Fe with As, P, S, Se or
  Te), Oxides (other O-containing), Other. This is a simple element rule, not
  3DSC's curated classification, recorded as ``family_source``.
* ``known_status``: whether the compound appears in the SuperCon (Stanev
  2018) or 3DSC tables, matched by atomic fractions: ``exact`` (within 0.0015)
  or ``near`` (same elements, every fraction within 0.02, as with MgB2 and the
  reported B2Mg0.98). ``superconductor`` means a reported Tc > 0;
  ``tested_non_superconductor`` means only Tc = 0 reports (3DSC controls);
  ``untested_by_formula`` means no exact formula match. An exact-formula miss
  does not prove nobody has ever tested the compound.

Doped formulas in SuperCon (e.g. La1.85Sr0.15CuO4) rarely equal an integer MP
formula, so they match only when the stoichiometries really agree.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
THREEDSC = ROOT / "data/raw/3dsc/repo/superconductors_3D/data/final/MP/3DSC_MP.csv"
STANEV = ROOT / "data/raw/3dsc/repo/superconductors_3D/data/source/SuperCon/raw/Supercon_data_by_2018_Stanev.csv"
PNICTOGEN_CHALCOGEN = {"As", "P", "S", "Se", "Te"}
# Same elements and every atomic fraction within 0.02 counts as the same
# compound (e.g. MgB2 vs the reported B2Mg0.98 at 38.4 K).
NEAR_FRACTION_TOLERANCE = 0.02


def fractions(formula: str) -> dict[str, float] | None:
    """Atomic fractions, so 'MgB2', 'Mg0.5B1' and 'B0.67Mg0.33' compare by ratio."""
    from pymatgen.core import Composition

    try:
        composition = Composition(formula)
    except Exception:  # noqa: BLE001 - unparseable literature formulas are skipped
        return None
    total = composition.num_atoms
    if not composition.elements or total <= 0:
        return None
    return {str(el): float(amount) / total for el, amount in composition.get_el_amt_dict().items()}


def reduced(formula: str) -> str | None:
    """Ratio key: elements with atomic fractions rounded to 0.001."""
    frac = fractions(formula)
    if frac is None:
        return None
    return " ".join(f"{el}:{frac[el]:.3f}" for el in sorted(frac))


def family(elements: set[str]) -> str:
    if {"Cu", "O"} <= elements:
        return "Cuprates"
    if "Fe" in elements and elements & PNICTOGEN_CHALCOGEN:
        return "Ferrites"
    if "O" in elements:
        return "Oxides"
    return "Other"


def known_index(threedsc: Path, stanev: Path) -> dict[frozenset[str], list[tuple[dict[str, float], float, str]]]:
    """Reported (fractions, Tc, source) grouped by element set."""
    index: dict[frozenset[str], list[tuple[dict[str, float], float, str]]] = defaultdict(list)

    def add(formula: str, tc: str | None, source: str) -> None:
        frac = fractions(formula)
        if frac and tc not in (None, ""):
            index[frozenset(frac)].append((frac, float(tc), source))

    if stanev.is_file():
        with stanev.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                add(row.get("name", ""), row.get("Tc"), "supercon_stanev2018")
    if threedsc.is_file():
        lines = threedsc.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        for row in csv.DictReader(lines):
            add(row.get("formula_sc", ""), row.get("tc"), "3dsc_mp")
    return dict(index)


def match_known(formula: str, index: dict[frozenset[str], list[tuple[dict[str, float], float, str]]]
                ) -> tuple[str, list[tuple[float, str]]]:
    """('exact' | 'near' | '', [(Tc, source), ...]) for reports of the same compound."""
    frac = fractions(formula)
    if frac is None:
        return "", []
    exact, near = [], []
    for other, tc, source in index.get(frozenset(frac), []):
        gap = max(abs(frac[el] - other[el]) for el in frac)
        if gap <= 0.0015:
            exact.append((tc, source))
        elif gap <= NEAR_FRACTION_TOLERANCE:
            near.append((tc, source))
    if exact:
        return "exact", exact
    if near:
        return "near", near
    return "", []


def read_snapshot(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(snapshot: Path, out: Path, *, threedsc: Path = THREEDSC, stanev: Path = STANEV) -> dict[str, Any]:
    rows = read_snapshot(snapshot)
    index = known_index(threedsc, stanev)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        elements = set(row["elements"].split())
        row["chemical_family"] = family(elements)
        row["family_source"] = "element_rule"
        kind, reports = match_known(row["formula"], index)
        if reports:
            tc_max = max(tc for tc, _ in reports)
            row["known_match"] = kind
            row["known_tc_max_k"] = f"{tc_max:g}"
            row["known_reports"] = str(len(reports))
            row["known_sources"] = " ".join(sorted({source for _, source in reports}))
            row["known_status"] = "superconductor" if tc_max > 0 else "tested_non_superconductor"
        else:
            row["known_match"] = row["known_tc_max_k"] = row["known_reports"] = row["known_sources"] = ""
            row["known_status"] = "untested_by_formula"
        counts[f"family_{row['chemical_family']}"] += 1
        counts[f"known_{row['known_status']}"] += 1
        if row["known_match"]:
            counts[f"known_match_{row['known_match']}"] += 1
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as gz:
        gz.write(buffer.getvalue().encode("utf-8"))
    return {"materials": len(rows), "counts": dict(sorted(counts.items())),
            "reference_reports_indexed": sum(len(v) for v in index.values()),
            "near_fraction_tolerance": NEAR_FRACTION_TOLERANCE,
            "reference_tables": {"supercon_stanev2018": stanev.is_file(), "3dsc_mp": threedsc.is_file()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/candidates/mp-stable-summary.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    args = parser.parse_args()
    if not (THREEDSC.is_file() and STANEV.is_file()):
        raise SystemExit("reference tables missing: run scripts/download_training_data.py first")
    print(json.dumps(build(args.snapshot, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
