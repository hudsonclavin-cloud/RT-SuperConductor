#!/usr/bin/env python3
"""Turn the Materials Project snapshot into a labelled candidate table.

Input: data/candidates/mp-stable-summary.csv.gz (scripts/fetch_mp_snapshot.py).
For every material this adds:

* ``chemical_family``: Cuprates (Cu and O), Ferrites (Fe with As, P, S, Se or
  Te), Oxides (other O-containing), Other. This is a simple element rule, not
  3DSC's curated classification, recorded as ``family_source``.
* ``known_status``: whether the compound appears in the reference tables,
  matched by atomic fractions: ``exact`` (within 0.0015) or ``near`` (same
  elements, every fraction within 0.02, as with MgB2 and the reported
  B2Mg0.98). ``superconductor`` means a reported Tc > 0;
  ``tested_non_superconductor`` means only Tc = 0 reports;
  ``untested_by_formula`` means no formula match. An exact-formula miss does
  not prove nobody has ever tested the compound.
* ``negative_evidence_tier``: how much a Tc = 0 label can be trusted. A
  database zero is not a verified null result, so every negative is graded
  (``NEGATIVE_TIERS`` below) and ``negative_training_use`` says whether the
  classifier may use it.

Reference tables and their lineages (independent chains of evidence):

  supercon   SuperCon via the 2018 Stanev export, 3DSC (whose Tc values all
             come from that export) and SuperCon-MTG ``S-`` rows (the 2024
             release, re-cleaned)
  hosono     SuperCon-MTG ``H-`` rows: Hosono's unsuccessful-materials search,
             which mixes measured non-superconductors with synthesis failures
  supermat   SuperCon-MTG ``SM-`` rows
  audit      data/training/negative_evidence.csv, filled in by a person from
             the original paper (see docs/negative-labels.md)
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
MTG = ROOT / "data/training/external/supercon-mtg/SuperCon_MTG.csv"
AUDIT = ROOT / "data/training/negative_evidence.csv"
PNICTOGEN_CHALCOGEN = {"As", "P", "S", "Se", "Te"}
# Same elements and every atomic fraction within 0.02 counts as the same
# compound (e.g. MgB2 vs the reported B2Mg0.98 at 38.4 K).
NEAR_FRACTION_TOLERANCE = 0.02
INSULATOR_GAP_EV = 1.0
LINEAGE = {"supercon_stanev2018": "supercon", "3dsc_mp": "supercon", "supercon_mtg": "supercon",
           "hosono_mtg": "hosono", "supermat_mtg": "supermat", "audit": "audit"}
NEGATIVE_TIERS = {
    "A_referenced_null": "a person checked the paper: no transition down to a stated temperature",
    "B_corroborated_zero": "Tc = 0 in at least two independent lineages, exact formula match",
    "C_database_zero": "Tc = 0 in one lineage, exact formula match, metallic in MP",
    "H_unsuccessful_search": "Tc = 0 only in Hosono's search list, which includes synthesis failures",
    "D_suspect": "near match only, or a wide-gap insulator in MP: likely never measured or a doping-boundary blur",
}
TRAINING_TIERS = {"A_referenced_null", "B_corroborated_zero", "C_database_zero"}

Report = tuple[dict[str, float], float, str]


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


def _lineage(source: str) -> str:
    return LINEAGE.get(source.split(":", 1)[0], source.split(":", 1)[0])


def known_index(threedsc: Path, stanev: Path, mtg: Path | None = None, audit: Path | None = None
                ) -> dict[frozenset[str], list[Report]]:
    """Reported (fractions, Tc, source) grouped by element set."""
    index: dict[frozenset[str], list[Report]] = defaultdict(list)

    def add(formula: str, tc: str | float | None, source: str) -> None:
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
    if mtg is not None and mtg.is_file():
        prefix_source = {"S": "supercon_mtg", "H": "hosono_mtg", "SM": "supermat_mtg"}
        with mtg.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                prefix = str(row.get("Material-ID", "")).split("-", 1)[0]
                source = prefix_source.get(prefix)
                if source:
                    add(row.get("Chemical_Formula", ""), row.get("Temp_critical"), source)
    if audit is not None and audit.is_file():
        with audit.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                observed = str(row.get("superconductivity_observed", "")).strip().lower()
                doi = str(row.get("publication_doi", "")).strip()
                if observed == "no" and doi:
                    add(row.get("formula", ""), 0.0, f"audit:{doi}")
                elif observed == "yes" and doi and row.get("tc_k"):
                    add(row.get("formula", ""), row["tc_k"], f"audit:{doi}")
    return dict(index)


def match_known(formula: str, index: dict[frozenset[str], list[Report]]) -> tuple[str, list[tuple[float, str]]]:
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


def negative_tier(kind: str, reports: list[tuple[float, str]], band_gap_ev: float | None) -> str:
    """Grade a Tc = 0 label; '' when the reports are not all zero."""
    if not reports or max(tc for tc, _ in reports) > 0:
        return ""
    lineages = {_lineage(source) for _, source in reports}
    if "audit" in lineages:
        return "A_referenced_null"
    if kind != "exact" or (band_gap_ev is not None and band_gap_ev > INSULATOR_GAP_EV):
        return "D_suspect"
    if len(lineages) >= 2:
        return "B_corroborated_zero"
    if lineages == {"hosono"}:
        return "H_unsuccessful_search"
    return "C_database_zero"


def training_label(row: dict[str, Any]) -> int | None:
    """1 for a superconductor, 0 for a usable negative, None otherwise."""
    status = row.get("known_status")
    if status == "superconductor":
        return 1
    if status == "tested_non_superconductor":
        use = row.get("negative_training_use")
        if use is None or use == "":  # tables built before tiers existed
            return 0
        return 0 if use == "use" else None
    return None


def read_snapshot(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gap(row: dict[str, str]) -> float | None:
    try:
        return float(row.get("band_gap_ev", ""))
    except (TypeError, ValueError):
        return None


def build(snapshot: Path, out: Path, *, threedsc: Path = THREEDSC, stanev: Path = STANEV,
          mtg: Path | None = MTG, audit: Path | None = AUDIT) -> dict[str, Any]:
    rows = read_snapshot(snapshot)
    index = known_index(threedsc, stanev, mtg, audit)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        elements = set(row["elements"].split())
        row["chemical_family"] = family(elements)
        row["family_source"] = "element_rule"
        kind, reports = match_known(row["formula"], index)
        if reports:
            tcs = [tc for tc, _ in reports]
            row["known_match"] = kind
            row["known_tc_max_k"] = f"{max(tcs):g}"
            row["known_tc_min_k"] = f"{min(tcs):g}"
            row["known_reports"] = str(len(reports))
            row["known_zero_reports"] = str(sum(tc == 0 for tc in tcs))
            row["known_positive_reports"] = str(sum(tc > 0 for tc in tcs))
            row["known_mixed_reports"] = str(min(tcs) == 0 < max(tcs))
            row["known_sources"] = " ".join(sorted({source.split(":", 1)[0] for _, source in reports}))
            row["known_lineages"] = " ".join(sorted({_lineage(source) for _, source in reports}))
            row["known_status"] = "superconductor" if max(tcs) > 0 else "tested_non_superconductor"
            row["negative_evidence_tier"] = negative_tier(kind, reports, _gap(row))
        else:
            for key in ("known_match", "known_tc_max_k", "known_tc_min_k", "known_reports", "known_zero_reports",
                        "known_positive_reports", "known_mixed_reports", "known_sources", "known_lineages",
                        "negative_evidence_tier"):
                row[key] = ""
            row["known_status"] = "untested_by_formula"
        tier = row["negative_evidence_tier"]
        row["negative_training_use"] = ("use" if tier in TRAINING_TIERS else "exclude") if tier else ""
        counts[f"family_{row['chemical_family']}"] += 1
        counts[f"known_{row['known_status']}"] += 1
        if row["known_match"]:
            counts[f"known_match_{row['known_match']}"] += 1
        if tier:
            counts[f"negative_tier_{tier}"] += 1
        if row["known_mixed_reports"] == "True":
            counts["known_mixed_reports"] += 1
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
            "negative_tiers": NEGATIVE_TIERS, "negative_training_tiers": sorted(TRAINING_TIERS),
            "reference_tables": {"supercon_stanev2018": stanev.is_file(), "3dsc_mp": threedsc.is_file(),
                                 "supercon_mtg": bool(mtg and mtg.is_file()),
                                 "audit": bool(audit and audit.is_file())}}


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
