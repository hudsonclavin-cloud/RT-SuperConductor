#!/usr/bin/env python3
"""The second blind table (``docs/blind-test-2.md``): nulls read from raw susceptibility curves, then one scoring.

    python scripts/blind_test2.py nulls     # raw files -> one row per file and one per composition, by the fixed rule

The raw files are the public dataset of Pogue et al. (2023), doi:10.34863/w3xm-w506, under
``data/raw/pogue-2023/files/`` (git ignores it). What is committed has no raw data: the per-file numbers the rule
used, and the per-composition verdicts.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scratchnet_common import ROOT

RAW = ROOT / "data/raw/pogue-2023/files"
OUT = ROOT / "data/benchmarks/blind2"
DOCS = ROOT / "docs/blind2"
FIELD_AT_MOST_OE = 100.0
REACH_K = 2.1
DENSITY_G_CM3 = 7.0
SIGNAL_FRACTION = 0.02
FLAGGED_SYSTEMS = ({"Zr", "In", "Ni"}, {"Zr", "In", "Cu"}, {"Zr", "Fe", "Sn"})
SAMPLE_CODE = re.compile(r"^[A-Z]{2,3}\d{3,4}[A-Za-z]?\d?$")
NOT_FORMULAS = {"HC", "SC", "AC", "DC", "In", "SPS", "OS", "CN", "HF", "PC"}
ELEMENT = ("H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|"
           "Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|"
           "Ir|Pt|Au|Hg|Tl|Pb|Bi|Th|U")
FORMULA = re.compile(rf"^(?:(?:{ELEMENT})(?:\d*\.?\d+)?|\((?:(?:{ELEMENT})(?:\d*\.?\d+)?)+\)(?:\d*\.?\d+)?)+$")
ONE_ELEMENT = re.compile(rf"(?:{ELEMENT})(?![a-z])")


def formula_in(name: str) -> tuple[str | None, str]:
    """(formula, reason) from a file name. Decimals are written with 'p' (Sn3p75) and codes like IH1004A are skipped."""
    lowered = name.lower()
    for word, reason in (("mix", "a mixture of targets"), ("reduced", "named 'reduced': oxygen content unknown"),
                         ("onsrtio3", "a film"), ("+", "a mixture of targets")):
        if word in lowered:
            return None, reason
    best = None
    for token in re.split(r"[_\-\s]+", name):
        token = re.sub(r"(?<=[A-Za-z\d\)])p(?=\d)", ".", token)
        if SAMPLE_CODE.match(token) or token in NOT_FORMULAS or not FORMULA.match(token):
            continue
        if token.isupper() and token.isalpha() and len(token) <= 3:      # initials such as BW, not boron-tungsten
            continue
        if len(set(ONE_ELEMENT.findall(token))) < 2:
            continue
        if any(float(x) > 100 for x in re.findall(r"\d*\.?\d+", token)):
            continue
        if best is None or len(token) > len(best):
            best = token
    return (best, "") if best else (None, "no composition in the file name")


def mass_mg(header: list[str], name: str) -> float | None:
    for line in header:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0] == "INFO":
            for value, label in ((parts[1], parts[2]), (parts[2], parts[1])):
                if label.upper() in ("SAMPLE_MASS", "MASS") or label.startswith("MASS:Sample Mass"):
                    try:
                        if float(value) > 0:
                            return float(value)
                    except ValueError:
                        pass
    found = re.search(r"(\d+)(?:(?:p|pt|\.)(\d+))?mg", name)
    return float(f"{found.group(1)}.{found.group(2) or 0}") if found else None


def running_median(values: list[float]) -> list[float]:
    if len(values) < 3:
        return list(values)
    return [statistics.median(values[max(0, i - 1):i + 2]) for i in range(len(values))]


def read_scan(text: str, name: str) -> dict:
    """One file -> the numbers the rule uses, or the reason the file is not a usable temperature scan."""
    lines = text.splitlines()
    if "[Data]" not in lines:
        return {"usable": False, "reason": "no data block"}
    start = lines.index("[Data]")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start + 1:]))))
    mass = mass_mg(lines[:start], name)

    def number(row: dict, column: str) -> float:
        try:
            return float(row.get(column) or "nan")
        except ValueError:
            return float("nan")

    points, kind = [], None
    for ac, drive in (("AC X' (emu/Oe)", None), ("M' (emu)", "Amplitude (Oe)")):
        found = [(number(r, "Temperature (K)"), number(r, "Magnetic Field (Oe)"),
                  number(r, ac) / (number(r, drive) if drive else 1.0)) for r in rows]
        found = [p for p in found if all(math.isfinite(x) for x in p)]
        if len(found) >= 5:
            points, kind = found, "AC"
            break
    if not points:
        for dc in ("DC Moment Free Ctr (emu)", "M-DC (emu)", "Moment (emu)"):
            found = [(number(r, "Temperature (K)"), number(r, "Magnetic Field (Oe)"), number(r, dc)) for r in rows]
            found = [(t, h, m / h) for t, h, m in found if all(math.isfinite(x) for x in (t, h, m)) and abs(h) >= 1.0]
            if len(found) >= 5:
                points, kind = found, "DC"
                break
    if not points:
        return {"usable": False, "reason": "no susceptibility column"}
    temperatures, fields = [p[0] for p in points], [p[1] for p in points]
    if max(temperatures) - min(temperatures) < 1.0:
        return {"usable": False, "reason": "not a temperature scan"}
    if max(fields) - min(fields) > 5.0:
        return {"usable": False, "reason": "the field changes during the scan"}
    if max(abs(h) for h in fields) > FIELD_AT_MOST_OE:
        return {"usable": False, "reason": "field above 100 Oe"}
    if mass is None:
        return {"usable": False, "reason": "no sample mass"}
    points.sort()
    smooth = running_median([p[2] / (mass / 1000.0) for p in points])
    lowest = min(smooth)
    return {"usable": True, "reason": "", "kind": kind, "mass_mg": mass, "points": len(points),
            "lowest_k": round(min(temperatures), 3), "highest_k": round(max(temperatures), 2),
            "field_oe": round(statistics.median(fields), 1), "lowest_chi_emu_per_oe_g": lowest,
            "shielding_fraction": -lowest * 4.0 * math.pi * DENSITY_G_CM3}


def judge(scans: list[dict]) -> tuple[str, str]:
    """A composition's verdict from all of its usable scans."""
    usable = [s for s in scans if s["usable"]]
    if not usable:
        return "left out", "no usable scan"
    if any(s["shielding_fraction"] >= SIGNAL_FRACTION for s in usable):
        return "signal seen", "a diamagnetic signal of 2% of full shielding or more"
    if any(s["lowest_k"] > REACH_K for s in usable):
        return "left out", "a scan does not reach 2.1 K"
    return "null", ""


FILE_FIELDS = ["file", "instrument", "formula", "composition", "usable", "reason", "kind", "mass_mg", "points",
               "lowest_k", "highest_k", "field_oe", "lowest_chi_emu_per_oe_g", "shielding_fraction"]
NULL_FIELDS = ["composition", "formula", "verdict", "reason", "family", "element_system", "scans", "lowest_k",
               "largest_shielding_fraction", "files"]


def cmd_nulls() -> dict:
    import build_mp_candidates as M
    import build_scratchnet_data as B
    import scratchnet_common as C

    files, skipped = [], Counter()
    for path in sorted(RAW.glob("*/*")):
        name = path.name.rsplit(".", 1)[0]
        formula, why = formula_in(name)
        row = {"file": path.name, "instrument": path.parent.name, "formula": formula or "", "composition": "",
               "usable": False, "reason": why}
        if formula:
            fractions = M.fractions(formula)
            if fractions is None:
                row["reason"] = "the composition does not parse"
            else:
                row["composition"] = B.composition_key(fractions)
                row.update(read_scan(path.read_text(encoding="utf-8", errors="replace"), name))
        if not row["usable"]:
            skipped[row["reason"]] += 1
        files.append(row)
    by_key = defaultdict(list)
    for row in files:
        if row["composition"]:
            by_key[row["composition"]].append(row)
    verdicts = []
    for key, scans in sorted(by_key.items()):
        fractions = B.parse_key(key)
        elements = set(fractions)
        verdict, reason = judge(scans)
        if any(system <= elements for system in FLAGGED_SYSTEMS):
            verdict, reason = "left out", "a system the authors flag as unresolved"
        usable = [s for s in scans if s["usable"]]
        verdicts.append({"composition": key, "formula": scans[0]["formula"], "verdict": verdict, "reason": reason,
                         "family": M.family(elements), "element_system": "-".join(sorted(elements)),
                         "scans": len(usable), "lowest_k": min((s["lowest_k"] for s in usable), default=""),
                         "largest_shielding_fraction": max((round(s["shielding_fraction"], 5) for s in usable),
                                                           default=""),
                         "files": " | ".join(s["file"] for s in scans)})
    for row in files:
        for column in ("lowest_chi_emu_per_oe_g", "shielding_fraction"):
            if column in row:
                row[column] = f"{row[column]:.4g}"
    C.write_table(OUT / "pogue-files.csv", files, FILE_FIELDS)
    C.write_table(OUT / "pogue-compositions.csv", verdicts, NULL_FIELDS)
    nulls = [v for v in verdicts if v["verdict"] == "null"]
    report = {"files": len(files), "files_with_a_usable_scan": sum(bool(f["usable"]) for f in files),
              "files_not_used": dict(skipped.most_common()), "compositions": len(verdicts),
              "verdicts": dict(Counter(v["verdict"] for v in verdicts)),
              "left_out_or_signal_reasons": dict(Counter(v["reason"] for v in verdicts if v["verdict"] != "null")),
              "nulls_by_family": dict(Counter(v["family"] for v in nulls)),
              "note": "before the rule that drops compositions the training tables hold"}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "nulls.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------------------------ papers

PAPERS = ROOT / "data/raw/blind2"                     # abstracts are publishers' text: they stay out of git
TITLE_PHRASES = ('"new superconductor"', '"discovery of superconductivity"', '"superconductivity in"')
# Whole words (or word beginnings), so that "investigate" is not "gate". "theor" and "predict" are checked in the
# title only: experimental abstracts often compare with theory.
DROP_ANYWHERE = re.compile(
    r"\b(?:films?|monolayers?|bilayers?|heterostructures?|interfaces?|pressures?|gpa|hydrides?|twisted|graphene|"
    r"gate|gated|gating|nanowires?|first[- ]principles|ab initio|density functional|cuprates?|nickelates?|"
    r"iron-based|pnictides?|ionic liquid|intercalat\w*|proximity|josephson|topological insulators?)\b")
DROP_IN_TITLE = re.compile(r"\b(?:theor\w*|predict\w*|calculat\w*)\b")
KELVIN = re.compile(r"\d\s*K\b")
PAPER_CAP, PAPERS_PER_BATCH, SAMPLE_SEED = 400, 25, 0


def abstract_text(inverted: dict | None) -> str:
    if not inverted:
        return ""
    words = sorted((position, word) for word, positions in inverted.items() for position in positions)
    return " ".join(word for _, word in words)


def keep_paper(title: str, abstract: str) -> bool:
    """The fixed filter: an abstract that gives a temperature in kelvin and names none of the excluded settings."""
    text = re.sub(r"ambient[- ]pressures?", " ", f"{title} {abstract}".lower())
    return (bool(KELVIN.search(abstract)) and "superconduct" in text and not DROP_ANYWHERE.search(text)
            and not DROP_IN_TITLE.search(title.lower()))


def cmd_papers() -> dict:
    import random
    import time
    import urllib.parse
    import urllib.request

    import blind_test as B1

    used = {row["identifier"] for row in csv.DictReader((B1.OUT / "records.csv").open(encoding="utf-8"))}
    found: dict[str, dict] = {}
    counts = Counter()
    for phrase in TITLE_PHRASES:
        cursor = "*"
        while cursor:
            query = urllib.parse.urlencode({
                "filter": f"title.search:{phrase},abstract.search:superconduct,from_publication_date:2021-01-01,"
                          "type:article",
                "per-page": 200, "cursor": cursor,
                "select": "id,doi,title,publication_year,abstract_inverted_index"})
            with urllib.request.urlopen(f"https://api.openalex.org/works?{query}", timeout=60) as reply:
                page = json.load(reply)
            for work in page["results"]:
                counts["returned"] += 1
                ident = B1.identifier(work.get("doi") or "")
                if ident is None:
                    counts["no usable doi"] += 1
                elif ident in used:
                    counts["used in the first test"] += 1
                elif ident not in found:
                    found[ident] = {"identifier": ident, "year": work["publication_year"], "title": work["title"] or "",
                                    "abstract": abstract_text(work.get("abstract_inverted_index"))}
            cursor = page["meta"].get("next_cursor") if page["results"] else None
            time.sleep(0.2)
    kept = sorted((p for p in found.values() if keep_paper(p["title"], p["abstract"])), key=lambda p: p["identifier"])
    counts["distinct papers"], counts["pass the filter"] = len(found), len(kept)
    if len(kept) > PAPER_CAP:
        kept = sorted(random.Random(SAMPLE_SEED).sample(kept, PAPER_CAP), key=lambda p: p["identifier"])
    counts["read"] = len(kept)
    (PAPERS / "batches").mkdir(parents=True, exist_ok=True)
    (PAPERS / "papers.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    batches = [kept[i:i + PAPERS_PER_BATCH] for i in range(0, len(kept), PAPERS_PER_BATCH)]
    for number, batch in enumerate(batches, 1):
        lines = []
        for paper in batch:
            lines += [f"IDENTIFIER: {paper['identifier']}", f"YEAR: {paper['year']}", f"TITLE: {paper['title']}",
                      f"ABSTRACT: {paper['abstract']}", ""]
        (PAPERS / "batches" / f"batch-{number:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
    report = {**counts, "batches": len(batches), "by_year": dict(sorted(Counter(p["year"] for p in kept).items()))}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "papers.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("nulls", "papers"))
    {"nulls": cmd_nulls, "papers": cmd_papers}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
