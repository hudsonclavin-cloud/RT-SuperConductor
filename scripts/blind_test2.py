#!/usr/bin/env python3
"""The second blind table (``docs/blind-test-2.md``): nulls read from raw susceptibility curves, then one scoring.

    python scripts/blind_test2.py nulls     # raw files -> one row per file and one per composition, by the fixed rule
    python scripts/blind_test2.py papers    # the fixed search and filter -> batches of abstracts for the readers
    python scripts/blind_test2.py collect   # the readers' files -> checked records, and the second readers' inputs
    python scripts/blind_test2.py table     # nulls + confirmed records -> the table, with the rules that drop seen rows
    python scripts/blind_test2.py score     # once: cycle 4 against cycle 3, and the champion against the baseline

The raw files are the public dataset of Pogue et al. (2023), doi:10.34863/w3xm-w506, under
``data/raw/pogue-2023/files/`` (git ignores it). What is committed has no raw data: the per-file numbers the rule
used, and the per-composition verdicts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


# ------------------------------------------------------------------------------------------------ records

RECORD_FIELDS = ["record_id", "batch", "formula", "class", "temperature_k", "year", "identifier", "first_report"]


def cmd_collect() -> dict:
    """The readers' files -> one checked list of records, and the second readers' input files."""
    import blind_test as B1
    import scratchnet_common as C

    valid = set(C.load_dataset().table.symbols)
    papers = {p["identifier"]: p for p in json.loads((PAPERS / "papers.json").read_text(encoding="utf-8"))}
    batch_of = {}
    for path in sorted((PAPERS / "batches").glob("batch-*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("IDENTIFIER: "):
                batch_of[line[len("IDENTIFIER: "):].strip()] = path.stem
    records, dropped, evidence, seen, lines = [], Counter(), {}, set(), 0
    for path in sorted((PAPERS / "batches").glob("batch-*.txt")):
        reader = PAPERS / "readers" / f"{path.stem}.tsv"
        if not reader.is_file():
            raise SystemExit(f"missing reader file: {reader}")
        rows = list(csv.reader(reader.open(newline="", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
        for row in rows[1:]:
            lines += 1
            ident = B1.identifier(row[4]) if len(row) == 7 else None
            if ident is None or batch_of.get(ident) != path.stem:
                dropped["identifier_not_in_this_batch" if len(row) == 7 else "wrong_column_count"] += 1
                continue
            record, reason = B1.check_record(row, int(papers[ident]["year"]), valid)
            if record is None:
                dropped[reason] += 1
                continue
            mark = (record["key"], record["class"], record["identifier"])
            if mark in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(mark)
            record.update(record_id=f"s{len(records) + 1:03d}", batch=path.stem)
            evidence[record["record_id"]] = row[6].strip()
            records.append(record)
    C.write_table(OUT / "records.csv", records, RECORD_FIELDS)
    (PAPERS / "verify-in").mkdir(parents=True, exist_ok=True)
    for batch in sorted({r["batch"] for r in records}):
        with open(PAPERS / "verify-in" / f"{batch}.tsv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["record_id", "identifier", "formula", "claimed_class", "claimed_temperature_k"])
            for r in records:
                if r["batch"] == batch:
                    writer.writerow([r["record_id"], r["identifier"], r["formula"], r["class"], r["temperature_k"]])
    (PAPERS / "evidence.json").write_text(json.dumps(evidence, indent=0), encoding="utf-8")
    report = {"reader_lines": lines, "records": len(records), "by_class": dict(Counter(r["class"] for r in records)),
              "dropped": dict(dropped), "papers_with_a_record": len({r["identifier"] for r in records}),
              "papers_read": len(papers)}
    (DOCS / "collect.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def read_verdicts() -> dict[str, dict]:
    import blind_test as B1

    verdicts = {}
    for path in sorted((PAPERS / "verify-out").glob("batch-*.tsv")):
        rows = list(csv.reader(path.open(newline="", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
        for row in rows[1:]:
            if len(row) >= 2 and row[1].strip().lower() in B1.VERDICTS:
                verdicts[row[0].strip()] = {"verdict": row[1].strip().lower(),
                                            "note": (row[2].strip() if len(row) > 2 else "")[:200]}
    return verdicts


# ------------------------------------------------------------------------------------------------ table

MAGNETIC = {"Cr", "Mn", "Fe", "Co", "Ni"}
POWER_EACH_CLASS = 25
TABLE_FIELDS = ["composition", "formula", "label", "tc_k", "null_lowest_k", "family", "element_system", "source",
                "new_element_system", "has_magnetic_element", "first_report", "identifiers", "record_ids"]


def training_keys() -> tuple[set[str], set[str]]:
    """(every composition a model of record or a NIMS-trained model has seen, the first blind table's compositions)."""
    import gzip

    import children_knowledge as G
    import scratchnet_common as C

    nims = set(json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))["keys"])
    record = {row["composition"] for row in C.load_dataset().rows}
    with gzip.open(ROOT / "data/processed/learning-cycles/nims-additions.csv.gz", "rt", encoding="utf-8") as handle:
        added = {row["composition"] for row in csv.DictReader(handle)}
    with open(ROOT / "data/benchmarks/blind/table.csv", newline="", encoding="utf-8") as handle:
        spent = {row["composition"] for row in csv.DictReader(handle)}
    return nims | record | added, spent


def candidate_rows() -> tuple[list[dict], Counter]:
    """Every composition with a label, from the three sources, before the rules that drop seen compositions."""
    import blind_test as B1
    import build_mp_candidates as M
    import build_scratchnet_data as B

    by_key: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    with open(OUT / "pogue-compositions.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["verdict"] == "null":
                by_key[row["composition"]].append({"class": "null", "temperature_k": float(row["lowest_k"]),
                                                   "formula": row["formula"], "source": "dataset", "first_report": "",
                                                   "identifier": "10.34863/w3xm-w506", "record_id": ""})
                counts["dataset nulls"] += 1
    first = list(csv.DictReader(open(B1.OUT / "records.csv", newline="", encoding="utf-8")))
    first_verdicts = {row["record_id"]: row["verdict"] for row in
                      csv.DictReader(open(B1.OUT / "verdicts.csv", newline="", encoding="utf-8"))}
    second = list(csv.DictReader(open(OUT / "records.csv", newline="", encoding="utf-8")))
    second_verdicts = {key: value["verdict"] for key, value in read_verdicts().items()}
    missing = [r["record_id"] for r in second if r["record_id"] not in second_verdicts]
    if missing:
        raise SystemExit(f"{len(missing)} new records have no verdict yet, e.g. {missing[:5]}")
    for records, verdicts, source in ((first, first_verdicts, "first test, unscored"),
                                      (second, second_verdicts, "new harvest")):
        for r in records:
            if verdicts.get(r["record_id"]) != "confirmed":
                counts[f"{source}: not confirmed"] += 1
                continue
            if r["class"] == "null" and float(r["temperature_k"]) > B1.NULL_AT_MOST_K:
                counts[f"{source}: null not measured to 2 K"] += 1
                continue
            by_key[B.composition_key(M.fractions(r["formula"]))].append({**r, "source": source,
                                                                         "temperature_k": float(r["temperature_k"])})
            counts[f"{source}: confirmed records"] += 1
    rows = []
    for key, group in sorted(by_key.items()):
        fractions = B.parse_key(key)
        positive = [r for r in group if r["class"] == "superconductor"]
        chosen = positive or group
        if positive and len(positive) < len(group):
            counts["compositions with both a null and a superconducting record (kept as superconductors)"] += 1
        rows.append({"composition": key, "formula": chosen[0]["formula"], "label": int(bool(positive)),
                     "tc_k": max(r["temperature_k"] for r in positive) if positive else "",
                     "null_lowest_k": "" if positive else min(r["temperature_k"] for r in group),
                     "family": M.family(set(fractions)), "element_system": "-".join(sorted(fractions)),
                     "source": " + ".join(sorted({r["source"] for r in chosen})),
                     "has_magnetic_element": int(bool(set(fractions) & MAGNETIC)),
                     "first_report": "yes" if any(r["first_report"] == "yes" for r in chosen) else "no_or_unclear",
                     "identifiers": " ".join(sorted({r["identifier"] for r in group})),
                     "record_ids": " ".join(r["record_id"] for r in group if r["record_id"])})
    return rows, counts


def drop_seen(rows: list[dict], seen: set[str], spent: set[str], systems: set[frozenset]) -> tuple[list[dict], Counter]:
    """Keep a composition only if no model has seen it or a near twin of it, and the first table did not score it."""
    import build_scratchnet_data as B

    ordered = sorted(seen)
    fractions = [B.parse_key(k) for k in ordered] + [B.parse_key(r["composition"]) for r in rows]
    twins = set()
    for a, b in B.near_pairs(fractions, B.NEAR_TOLERANCE):
        if (a < len(ordered)) != (b < len(ordered)):
            twins.add(max(a, b) - len(ordered))
    kept, why = [], Counter()
    for i, row in enumerate(rows):
        if row["composition"] in spent:
            why["scored in the first blind table"] += 1
        elif row["composition"] in seen:
            why["in a training table"] += 1
        elif i in twins:
            why["a near twin of a training composition"] += 1
        else:
            kept.append({**row, "new_element_system": int(frozenset(B.parse_key(row["composition"])) not in systems)})
    return kept, why


def describe(rows: list[dict]) -> dict:
    return {"compositions": len(rows), "superconductors": sum(r["label"] for r in rows),
            "nulls": sum(1 - r["label"] for r in rows)}


def cmd_table() -> dict:
    import build_scratchnet_data as B
    import children_knowledge as G
    import scratchnet_common as C

    rows, counts = candidate_rows()
    seen, spent = training_keys()
    systems = {frozenset(B.parse_key(k)) for k in json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))["keys"]}
    kept, why = drop_seen(rows, seen, spent, systems)
    C.write_table(OUT / "table.csv", kept, TABLE_FIELDS)
    verdicts = read_verdicts()
    C.write_table(OUT / "verdicts.csv", [{"record_id": k, **v} for k, v in sorted(verdicts.items())],
                  ["record_id", "verdict", "note"])
    other = [r for r in kept if r["family"] == "Other"]
    report = {"records": dict(counts), "compositions_before_the_rules": len(rows), "dropped": dict(why),
              "second_reader_verdicts": dict(Counter(v["verdict"] for v in verdicts.values())),
              "table": {**describe(kept), "families": dict(Counter(r["family"] for r in kept)),
                        "sources": dict(Counter((r["source"], r["label"]) and f"{r['source']} / label {r['label']}"
                                                for r in kept))},
              "intermetallics": {**describe(other),
                                 "in_a_new_element_system": describe([r for r in other if r["new_element_system"]]),
                                 "without_magnetic_elements": describe([r for r in other
                                                                        if not r["has_magnetic_element"]])},
              "table_sha256": hashlib.sha256((OUT / "table.csv").read_bytes()).hexdigest()}
    report["power_rule_met"] = bool(report["intermetallics"]["nulls"] >= POWER_EACH_CLASS
                                    and report["intermetallics"]["superconductors"] >= POWER_EACH_CLASS)
    (DOCS / "table.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------------------------ score

LOCKBOX_CHAMPION = ("elements", "proxy", "alloy", "tc-size")
CYCLE_MODELS = ROOT / "data/processed/learning-cycles/models"


def gain(y, groups, first, second, resamples: int = 1000) -> dict:
    """AUC of ``first`` minus AUC of ``second`` on these rows, with intervals from resampling element systems."""
    import blind_test as B1
    import children_nims as N

    if len(set(y.tolist())) < 2:
        return {"compositions": int(len(y)), "note": "one class only"}
    out = {"compositions": int(len(y)), "superconductors": int(y.sum()), "nulls": int(len(y) - y.sum()),
           "auc_first": round(N.auc(y, first), 4), "auc_second": round(N.auc(y, second), 4)}
    out["difference"] = round(out["auc_first"] - out["auc_second"], 4)
    out.update(B1.interval(y, groups, first, second, resamples=resamples))
    return out


def cmd_score() -> dict:
    import joblib
    import numpy as np

    import children_knowledge as G
    import scratchnet_common as C

    out = DOCS / "score.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the second blind table is scored once.")
    table = OUT / "table.csv"
    summary = json.loads((DOCS / "table.json").read_text(encoding="utf-8"))
    if hashlib.sha256(table.read_bytes()).hexdigest() != summary["table_sha256"]:
        raise SystemExit("the table is not the one whose checksum was recorded")
    rows = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
    fractions = [C.fractions_of(r["composition"]) for r in rows]
    X, _ = C.formula_features([C.formula_text(f) for f in fractions])
    scores = {}
    for cycle in ("cycle3", "cycle4"):
        models = sorted((CYCLE_MODELS / cycle).glob("boosting-fold*.joblib"))
        if len(models) != 5:
            raise SystemExit(f"expected five {cycle} models, found {len(models)}")
        scores[cycle] = np.mean([C.boosting_apply(joblib.load(m), X)[0] for m in models], axis=0)
    scores["champion"], scores["baseline"] = G.score_new(fractions, LOCKBOX_CHAMPION, 0)
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["element_system"] for r in rows])
    other = np.array([r["family"] == G.PRIMARY_FAMILY for r in rows])
    from_dataset = np.array([r["label"] == "1" or "dataset" in r["source"] for r in rows])
    no_magnet = np.array([r["has_magnetic_element"] == "0" for r in rows])
    np.savez_compressed(ROOT / "data/processed/children-knowledge/blind2-scores.npz", y=y, **scores)

    def comparison(first: str, second: str) -> dict:
        block = {"first": first, "second": second,
                 "primary_intermetallics": gain(y[other], groups[other], scores[first][other], scores[second][other]),
                 "all_compositions": gain(y, groups, scores[first], scores[second]),
                 "sensitivity": {
                     "intermetallics_dataset_nulls_only": gain(y[other & from_dataset], groups[other & from_dataset],
                                                               scores[first][other & from_dataset],
                                                               scores[second][other & from_dataset]),
                     "intermetallics_without_magnetic_elements": gain(
                         y[other & no_magnet], groups[other & no_magnet], scores[first][other & no_magnet],
                         scores[second][other & no_magnet])}}
        primary = block["primary_intermetallics"]
        if "gain_95" in primary:
            low, high = primary["gain_95"]
            block["reading"] = (f"{first} is better" if low > 0 else f"{second} is better" if high < 0 else
                                "no difference shown")
        return block

    report = {"table_sha256": summary["table_sha256"],
              "A_cycle4_against_cycle3": comparison("cycle4", "cycle3"),
              "B_champion_against_baseline": comparison("champion", "baseline"),
              "champion_genes": list(LOCKBOX_CHAMPION),
              "warning": "the classes come from different sources, so only the differences are read, not the AUC values"}
    primary = report["A_cycle4_against_cycle3"]["primary_intermetallics"]
    report["power_rule_met"] = bool(primary.get("nulls", 0) >= POWER_EACH_CLASS
                                    and primary.get("superconductors", 0) >= POWER_EACH_CLASS)
    if not report["power_rule_met"]:
        for key in ("A_cycle4_against_cycle3", "B_champion_against_baseline"):
            if "reading" in report[key]:
                report[key]["reading"] = "underpowered: no claim (" + report[key]["reading"] + ")"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("nulls", "papers", "collect", "table", "score"))
    {"nulls": cmd_nulls, "papers": cmd_papers, "collect": cmd_collect, "table": cmd_table, "score": cmd_score}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
