#!/usr/bin/env python3
"""The blind test of the generation 7 champion on papers from 2021 on (``docs/blind-test.md``).

    python scripts/blind_test.py collect    # the readers' files -> one checked list of records, and verifier batches
    python scripts/blind_test.py table      # confirmed records -> one row per composition in a new element system
    python scripts/blind_test.py score      # once: the frozen baseline and champion on the committed table

The readers' files and the verifiers' answers live under ``data/raw/blind-2021/`` (git ignores it, and the readers'
quoted abstract sentences stay there). What is committed has no abstract text: the records, the verdicts, the table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scratchnet_common import ROOT

RAW = ROOT / "data/raw/blind-2021"
OUT = ROOT / "data/benchmarks/blind"
DOCS = ROOT / "docs/blind"
YEARS, THEMES = range(2021, 2027), range(1, 6)
EXTENSION_THEMES = (6, 7)            # two more null themes, added before any score (the plan's amendment)
FIRST_HARVEST_BATCHES = 8
NULL_AT_MOST_K = 2.0
STRICT_NULL_K = 0.5
BATCH = 30
POWER_COMPOSITIONS, POWER_NULLS = 80, 25
ANCHOR = ROOT / "data/benchmarks/rung4/shortlist2-second-opinion.csv"
ANCHOR_SHA256 = "4a63bf256b575afa4eed17fa0a3fa72ba2e8fb9c00cd81122c32c77825ded802"
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
ARXIV = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(\d{4}\.\d{4,5})$", re.IGNORECASE)
RECORD_FIELDS = ["record_id", "cell", "formula", "class", "temperature_k", "year", "identifier", "first_report"]
VERDICTS = ("confirmed", "wrong_value", "not_in_abstract", "not_bulk_ambient", "wrong_year", "not_found")


def identifier(text: str) -> str | None:
    """A DOI or an arXiv number in one spelling, or None if it is neither."""
    text = text.strip().removeprefix("https://doi.org/").removeprefix("doi:").strip()
    arxiv = ARXIV.match(text) or ARXIV_DOI.match(text)
    if arxiv:
        return f"arxiv:{arxiv.group(1)}"
    return text.lower() if DOI.match(text) else None


def check_record(row: list[str], year: int, valid: set[str]) -> tuple[dict | None, str]:
    """One reader line -> a record, or the reason it is dropped."""
    import build_mp_candidates as M
    import build_scratchnet_data as B

    if len(row) != 7:
        return None, "wrong_column_count"
    formula, kind, temperature, stated_year, ident, first, _evidence = (x.strip() for x in row)
    kind = kind.lower()
    if kind not in ("superconductor", "null"):
        return None, "bad_class"
    try:
        value = float(temperature)
    except ValueError:
        return None, "temperature_not_a_number"
    if not math.isfinite(value) or value < 0 or (kind == "superconductor" and value == 0):
        return None, "temperature_out_of_range"
    if stated_year != str(year):
        return None, "year_not_the_cell_s"
    ident = identifier(ident)
    if ident is None:
        return None, "no_doi_or_arxiv_number"
    fractions = M.fractions(formula)
    if not fractions or not set(fractions) <= valid or len(fractions) < 1:
        return None, "formula_not_usable"
    first = first.lower() if first.lower() in ("yes", "no") else "unclear"
    return {"formula": formula, "class": kind, "temperature_k": value, "year": year, "identifier": ident,
            "first_report": first, "key": B.composition_key(fractions)}, ""


def cmd_collect() -> dict:
    import scratchnet_common as C

    valid = set(C.load_dataset().table.symbols)
    records, dropped, per_cell, seen = [], Counter(), {}, set()
    evidence = {}
    cells = [(year, theme) for year in YEARS for theme in THEMES]
    cells += [(year, theme) for year in YEARS for theme in EXTENSION_THEMES]      # after, so earlier ids do not move
    for year, theme in cells:
        if True:
            cell = f"y{year}-t{theme}"
            path = RAW / "readers" / f"{cell}.tsv"
            if not path.is_file():
                raise SystemExit(f"missing reader file: {path}")
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
            kept = 0
            for row in rows[1:]:
                record, reason = check_record(row, year, valid)
                if record is None:
                    dropped[reason] += 1
                    continue
                mark = (record["key"], record["class"], record["identifier"])
                if mark in seen:
                    dropped["duplicate_of_another_reader_s_record"] += 1
                    continue
                seen.add(mark)
                record.update(record_id=f"r{len(records) + 1:03d}", cell=cell)
                evidence[record["record_id"]] = row[6].strip()
                records.append(record)
                kept += 1
            per_cell[cell] = {"lines": len(rows) - 1, "kept": kept}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "records.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    batches = RAW / "verify-in"
    batches.mkdir(parents=True, exist_ok=True)
    first = [r for r in records if int(r["cell"].split("-t")[1]) in THEMES]
    later = [r for r in records if int(r["cell"].split("-t")[1]) in EXTENSION_THEMES]
    order = np.random.default_rng(0).permutation(len(first))            # mix years and themes within a batch
    groups = [[first[i] for i in order[start:start + BATCH]] for start in range(0, len(first), BATCH)]
    if len(groups) != FIRST_HARVEST_BATCHES:
        raise SystemExit("the first harvest's batches have changed; its records must stay as verified")
    groups += [later[start:start + BATCH] for start in range(0, len(later), BATCH)]
    for number, group in enumerate(groups, start=1):
        with open(batches / f"batch-{number:02d}.tsv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["record_id", "identifier", "formula", "claimed_class", "claimed_temperature_k", "claimed_year"])
            for r in group:
                writer.writerow([r["record_id"], r["identifier"], r["formula"], r["class"], r["temperature_k"], r["year"]])
    with open(RAW / "evidence.json", "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=0)
    summary = {"reader_lines": sum(c["lines"] for c in per_cell.values()), "records": len(records),
               "by_class": dict(Counter(r["class"] for r in records)), "dropped": dict(dropped),
               "papers": len({r["identifier"] for r in records}), "verifier_batches": len(groups),
               "per_cell": per_cell}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "collect.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cell"}, indent=1))
    return summary


def read_verdicts() -> dict[str, dict]:
    verdicts = {}
    for path in sorted((RAW / "verify-out").glob("batch-*.tsv")):
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
        for row in rows[1:]:
            if len(row) >= 2 and row[1].strip().lower() in VERDICTS:
                verdicts[row[0].strip()] = {"verdict": row[1].strip().lower(),
                                            "note": (row[2].strip() if len(row) > 2 else "")[:200]}
    return verdicts


def build_table(records: list[dict], verdicts: dict[str, dict], nims_systems: set[frozenset]) -> tuple[list[dict], dict]:
    """Confirmed records -> one row per composition in an element system the NIMS data never held."""
    import build_mp_candidates as M
    import build_scratchnet_data as B

    steps = {"records": len(records)}
    confirmed = [r for r in records if verdicts.get(r["record_id"], {}).get("verdict") == "confirmed"]
    steps["confirmed_by_the_second_reader"] = len(confirmed)
    usable = [r for r in confirmed if r["class"] == "superconductor" or float(r["temperature_k"]) <= NULL_AT_MOST_K]
    steps["nulls_measured_to_2_k_or_lower"] = len(usable)
    by_key = defaultdict(list)
    for r in usable:
        by_key[B.composition_key(M.fractions(r["formula"]))].append(r)
    steps["compositions"] = len(by_key)
    rows = []
    for key, group in sorted(by_key.items()):
        fractions = B.parse_key(key)
        if frozenset(fractions) in nims_systems:
            continue
        positive = [r for r in group if r["class"] == "superconductor"]
        chosen = positive or group
        rows.append({"composition": key, "formula": chosen[0]["formula"], "label": int(bool(positive)),
                     "tc_k": max(float(r["temperature_k"]) for r in positive) if positive else "",
                     "null_lowest_k": "" if positive else min(float(r["temperature_k"]) for r in group),
                     "first_year": min(int(r["year"]) for r in group),
                     "first_report": "yes" if any(r["first_report"] == "yes" for r in chosen) else "no_or_unclear",
                     "family": M.family(set(fractions)), "element_system": "-".join(sorted(fractions)),
                     "first_harvest": "yes" if any(int(r.get("cell", "y0-t1").split("-t")[1]) in THEMES
                                                   for r in group) else "no",
                     "identifiers": " ".join(sorted({r["identifier"] for r in group})),
                     "record_ids": " ".join(r["record_id"] for r in group)})
    steps["in_a_new_element_system"] = len(rows)
    return rows, steps


TABLE_FIELDS = ["composition", "formula", "label", "tc_k", "null_lowest_k", "first_year", "first_report", "family",
                "element_system", "first_harvest", "identifiers", "record_ids"]


def cmd_table() -> dict:
    import children_knowledge as G

    with open(OUT / "records.csv", newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    verdicts = read_verdicts()
    missing = [r["record_id"] for r in records if r["record_id"] not in verdicts]
    if missing:
        raise SystemExit(f"{len(missing)} records have no verdict yet, e.g. {missing[:5]}")
    with open(OUT / "verdicts.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "verdict", "note"])
        for r in records:
            writer.writerow([r["record_id"], verdicts[r["record_id"]]["verdict"], verdicts[r["record_id"]]["note"]])
    nims = json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))["keys"]
    import scratchnet_common as C
    systems = {frozenset(C.fractions_of(k)) for k in nims}
    rows, steps = build_table(records, verdicts, systems)
    with open(OUT / "table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    other = [r for r in rows if r["family"] == "Other"]
    summary = {"steps": steps, "verdicts": dict(Counter(v["verdict"] for v in verdicts.values())),
               "rejected_fraction": round(1 - steps["confirmed_by_the_second_reader"] / steps["records"], 3),
               "table": {"compositions": len(rows), "superconductors": sum(r["label"] for r in rows),
                         "nulls": sum(1 - r["label"] for r in rows), "families": dict(Counter(r["family"] for r in rows)),
                         "element_systems": len({r["element_system"] for r in rows})},
               "intermetallics": {"compositions": len(other), "superconductors": sum(r["label"] for r in other),
                                  "nulls": sum(1 - r["label"] for r in other)},
               "power_rule": {"needs_compositions": POWER_COMPOSITIONS, "needs_nulls": POWER_NULLS}}
    summary["intermetallics"]["nulls_from_the_first_harvest"] = sum(1 - r["label"] for r in other
                                                                     if r["first_harvest"] == "yes")
    summary["power_rule"]["met"] = bool(len(other) >= POWER_COMPOSITIONS
                                        and summary["intermetallics"]["nulls"] >= POWER_NULLS)
    (DOCS / "table.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def interval(y, groups, family, a, b, which, resamples: int = 1000, seed: int = 0) -> dict:
    """Gain of scores ``a`` over ``b`` in AUC, with intervals from resampling element systems."""
    import children_knowledge as G

    fold = {"blind": {"y": y, "groups": groups, "family": family, "scores": {"baseline": b, "model": a}}}
    out = G.fold_bootstrap(fold, ["baseline", "model"], resamples=resamples, seed=seed)["model"]
    name = "primary" if which == G.PRIMARY_FAMILY else "all"
    return {"gain_95": out[f"{name}_gain_95"], "gain_99": out[f"{name}_gain_99"]}


def cmd_score() -> dict:
    import joblib

    import children_knowledge as G
    import rung4 as R
    import scratchnet_common as C

    out = DOCS / "score.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the blind table is scored once.")
    if hashlib.sha256(ANCHOR.read_bytes()).hexdigest() != ANCHOR_SHA256:
        raise SystemExit("the committed champion scores for the rung 4 survivors have changed")
    if G.pool_rows() != list(csv.reader(open(ANCHOR, newline="", encoding="utf-8"))):
        raise SystemExit("refitting the champion no longer reproduces its committed scores: the models have drifted")
    with open(OUT / "table.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fractions = [C.fractions_of(r["composition"]) for r in rows]
    spec = G.champion_spec()
    champion, baseline = G.score_new(fractions, tuple(spec["changes"]), spec["bags"])
    X, _ = C.formula_features([C.formula_text(f) for f in fractions])
    record = np.mean([C.boosting_apply(joblib.load(p), X)[0] for p in sorted(R.MODELS.glob("boosting-fold*.joblib"))],
                     axis=0)
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["element_system"] for r in rows])
    family = np.array([r["family"] for r in rows])
    np.savez_compressed(G.WORK / "blind-scores.npz", y=y, champion=champion, baseline=baseline, record=record)

    def block(mask: np.ndarray) -> dict:
        if len(set(y[mask].tolist())) < 2:
            return {"compositions": int(mask.sum()), "note": "one class only"}
        entry = {"compositions": int(mask.sum()), "superconductors": int(y[mask].sum()),
                 "nulls": int(mask.sum() - y[mask].sum()),
                 "auc": {"baseline": round(G.N.auc(y[mask], baseline[mask]), 4),
                         "champion": round(G.N.auc(y[mask], champion[mask]), 4),
                         "ranking_of_record": round(G.N.auc(y[mask], record[mask]), 4)}}
        entry["champion_minus_baseline"] = round(entry["auc"]["champion"] - entry["auc"]["baseline"], 4)
        everything = np.full(int(mask.sum()), "x")
        entry.update(interval(y[mask], groups[mask], everything, champion[mask], baseline[mask], None))
        return entry

    other = family == G.PRIMARY_FAMILY
    strict_null = np.array([r["label"] == "1" or float(r["null_lowest_k"]) <= STRICT_NULL_K for r in rows])
    first = np.array([r["label"] == "0" or r["first_report"] == "yes" for r in rows])
    harvest_one = np.array([r["first_harvest"] == "yes" for r in rows])
    report = {"champion": spec, "table_sha256": hashlib.sha256((OUT / "table.csv").read_bytes()).hexdigest(),
              "primary_intermetallics": block(other), "all_compositions": block(np.ones(len(y), dtype=bool)),
              "sensitivity": {"intermetallics_nulls_to_0.5_k_only": block(other & strict_null),
                              "intermetallics_first_reports_only": block(other & first),
                              "intermetallics_first_harvest_only": block(other & harvest_one)},
              "ranking_of_record_warning": "its training table has no dates and may contain some of these compounds"}
    primary = report["primary_intermetallics"]
    report["power_rule_met"] = bool(primary.get("compositions", 0) >= POWER_COMPOSITIONS
                                    and primary.get("nulls", 0) >= POWER_NULLS)
    if "gain_95" in primary:
        low, gain = primary["gain_95"][0], primary["champion_minus_baseline"]
        report["reading"] = ("confirmed on new data" if low > 0 else
                             "consistent with the selection evidence, not confirmed" if gain > 0 else
                             "the selection gain did not carry over")
        if not report["power_rule_met"]:
            report["reading"] = "underpowered: no claim (" + report["reading"] + ")"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("collect", "table", "score"))
    {"collect": cmd_collect, "table": cmd_table, "score": cmd_score}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
