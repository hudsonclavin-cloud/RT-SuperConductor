#!/usr/bin/env python3
"""Scan the training table for probable duplicates and labels that look wrong, for a person to check.

    python scripts/label_audit.py                       # writes data/processed/label-audit/ (not committed)
    python scripts/label_audit.py --out DIR --summary FILE   # as the GitHub Action runs it

It decides nothing and changes no label. Every flag is a lead for the owner's correction rule (two independent readers
and a sceptical verifier, recorded in data/training/negative_evidence.csv); see docs/label-audit.md.

What it flags, among the development and train-only rows only:
* **probable duplicates**: two rows with the same elements that differ by at most 0.005 on every element (a rounding or
  notation twin) but carry opposite labels;
* **possible false negatives** (label 0), with one point for each independent signal:
  - a near twin with the same elements (within 0.02 on every element, so the same group) is a superconductor. Twins
    that differ by an added element, and cuprates, are left out: a dopant (often a magnetic one) can really switch
    superconductivity on or off, and a cuprate's doping series really crosses from insulator to superconductor;
  - another source reports it superconducting: the NIMS SuperCon file, NEMAD's usable subset (flagged experimental, not
    at a known high pressure), or a SuperCon2 record with a Tc at unstated pressure (text-mined, so the weakest);
  - the ranking of record (the 50/50 blend of cycle 6 and TabICLv2) gives it, out of fold, at least 0.9;
* **possible false positives** (label 1): a near twin is a null; another source holds only nulls for it; the blend
  gives it at most 0.1 out of fold; or its Tc is above its family's ambient-pressure record.

Locked rows are never read: their labels are kept for one-time scoring. Near twins always share a group and a group is
locked whole, so no development or train-only row has a locked twin; the scanner checks that anyway. Compositions
already reviewed (the correction records, the cycle evidence files, the review queues) are counted, not listed.
Sources that are not present are skipped and named in the report, so the same script runs here and in CI.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_scratchnet_data as B  # noqa: E402
import scratchnet_common as C  # noqa: E402

OUT = ROOT / "data/processed/label-audit"
NIMS_TABLE = ROOT / "data/processed/children-knowledge/table.json"
NEMAD_TABLE = ROOT / "data/processed/nemad/superconductor.csv"
SUPERCON2 = ROOT / "data/raw/supercon2/supercon2_v22.12.03.csv"
# NIMS and NEMAD labels for the scanned compositions only (no locked row), for runs without the raw files (CI).
# Both sources are CC BY 4.0: NIMS SuperCon, doi:10.48505/nims.4487; NEMAD, Itani et al., J. Magn. Magn. Mater. 2026.
EXTRACT = ROOT / "data/training/label-audit-external.csv.gz"
OOF = {"cycle6": [ROOT / "data/benchmarks/learning-cycles/cycle6-oof.npz", ROOT / "data/processed/learning-cycles/cycle6-oof.npz"],
       "tabicl": [ROOT / "data/benchmarks/local-gpu/fm-tabicl-20261123/oof.npz"]}
REVIEWED_FILES = [ROOT / "data/training/negative_evidence.csv", ROOT / "data/training/reviewed/reviewed_records.csv",
                  ROOT / "data/training/label_audit_queue.csv", ROOT / "data/training/label-audit-reviewed.csv"]
REVIEWED_GLOBS = ["data/benchmarks/learning-cycles/*evidence*.json", "data/benchmarks/learning-cycles/*triage-leads.json"]
DUPLICATE = 0.005                 # rows this close on every element are the same material written two ways
HIGH, LOW = 0.9, 0.1              # out-of-fold blend probabilities that count as a signal
# Highest Tc at ambient pressure that this project accepts without a second look, by build_mp_candidates.family.
FAMILY_RECORD_K = {"Cuprates": 140.0, "Ferrites": 60.0, "Oxides": 40.0, "Other": 40.0}


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a.get(e, 0.0) - b.get(e, 0.0)) for e in set(a) | set(b))


def table_rows(ds) -> list[dict]:
    """Development and train-only rows with a label. Locked rows are dropped before any label is looked at."""
    out = []
    for row, split, family, label, tc in zip(ds.rows, ds.split, ds.family, ds.arrays.label, ds.arrays.tc_k):
        if split == "lockbox" or not np.isfinite(label):
            continue
        out.append({"key": row["composition"], "label": int(label), "tc": float(tc) if np.isfinite(tc) else None,
                    "family": str(family), "split": str(split), "group": row["group"]})
    return out


def twin_conflicts(rows: list[dict]) -> list[tuple[int, int, float]]:
    """Index pairs of non-cuprate near twins (within B.NEAR_TOLERANCE) with the same elements and opposite labels."""
    fractions = [C.fractions_of(r["key"]) for r in rows]
    pairs = []
    for i, j in B.near_pairs(fractions, B.NEAR_TOLERANCE):
        same = {e for e, v in fractions[i].items() if v > 0} == {e for e, v in fractions[j].items() if v > 0}
        if same and rows[i]["label"] != rows[j]["label"] and "Cuprates" not in (rows[i]["family"], rows[j]["family"]):
            if rows[i]["group"] != rows[j]["group"]:
                raise SystemExit(f"near twins in different groups: {rows[i]['key']} / {rows[j]['key']}")
            pairs.append((i, j, distance(fractions[i], fractions[j])))
    return pairs


def key_of_formula(formula: str) -> str | None:
    import build_mp_candidates as M

    try:
        fractions = M.fractions(formula)
    except Exception:
        return None
    return B.composition_key(fractions) if fractions else None


def nims_labels(path: Path = NIMS_TABLE) -> dict[str, int] | None:
    if not path.is_file():
        return None
    nims = json.loads(path.read_text(encoding="utf-8"))
    return {key: int(any(label for _, label, _ in nims["records"][key])) for key in nims["keys"]}


def nemad_labels(path: Path = NEMAD_TABLE) -> dict[str, int] | None:
    if not path.is_file():
        return None
    import build_nemad_additions as BA

    with open(path, newline="", encoding="utf-8") as handle:
        return {k: v[0] for k, v in BA.nemad_labels(list(csv.DictReader(handle))).items()}


def supercon2_positive(path: Path = SUPERCON2) -> dict[str, int] | None:
    """Compositions SuperCon2 reports with a Tc above zero at unstated pressure. It holds no reliable nulls."""
    if not path.is_file():
        return None
    import nemad_download as ND

    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("appliedPressure", "").strip():
                continue
            match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", row.get("criticalTemperature", ""))
            name = re.sub(r"\s+", "", row.get("formula", ""))
            if not match or float(match.group(1)) <= 0 or not name:
                continue
            key = key_of_formula(ND.clean_name(name))
            if key:
                out[key] = 1
    return out


def blend_oof(paths: dict[str, list[Path]] = OOF) -> dict[str, float] | None:
    """The ranking of record's out-of-fold probability per development composition (mean of the two halves)."""
    parts = {}
    for name, options in paths.items():
        found = next((p for p in options if p.is_file()), None)
        if found is None:
            return None
        with np.load(found, allow_pickle=False) as d:
            parts[name] = dict(zip((str(k) for k in d["compositions"]), d["p"].tolist()))
    keys = set.intersection(*(set(p) for p in parts.values()))
    return {k: float(np.mean([p[k] for p in parts.values()])) for k in keys}


def reviewed_keys(files: list[Path] = REVIEWED_FILES, globs: list[str] = REVIEWED_GLOBS) -> set[str]:
    """Compositions a person has already looked at, from any formula or composition field in the review records."""
    texts: list[str] = []
    for path in files:
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    texts += [row.get(c, "") for c in ("formula", "composition") if row.get(c)]

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("formula", "composition", "compound", "key") and isinstance(v, str):
                    texts.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            walk(json.loads(path.read_text(encoding="utf-8")))
    out = set()
    for text in texts:
        if ":" in text and ";" in text:
            out.add(text)
        else:
            key = key_of_formula(re.sub(r"\s+", "", text))
            if key:
                out.add(key)
    return out


def audit(rows: list[dict], sources: dict[str, dict[str, int] | None], oof: dict[str, float] | None,
          reviewed: set[str]) -> dict:
    """The flags, ranked: most independent signals first, then the strongest out-of-fold disagreement."""
    signals: dict[int, list[str]] = defaultdict(list)
    duplicates = []
    twins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i, j, d in twin_conflicts(rows):
        twins[i].append((d, j))
        twins[j].append((d, i))
        if d <= DUPLICATE:
            duplicates.append({"first": rows[i]["key"], "second": rows[j]["key"], "distance": round(d, 4),
                               "labels": [rows[i]["label"], rows[j]["label"]]})
    for i, found in twins.items():                      # one signal per row, however many twins disagree
        d, j = min(found)
        signals[i].append(f"{len(found)} near twin(s) with the same elements carry the other label "
                          f"(closest: {C.formula_text(C.fractions_of(rows[j]['key']))}, {d:.3f} apart)")
    for i, r in enumerate(rows):
        for name, labels in sources.items():
            if labels is None or r["key"] not in labels:
                continue
            if r["label"] == 0 and labels[r["key"]] == 1:
                signals[i].append(f"{name} reports it superconducting")
            elif r["label"] == 1 and labels[r["key"]] == 0 and name != "SuperCon2":
                signals[i].append(f"{name} holds only nulls for it")
        p = oof.get(r["key"]) if oof else None
        if p is not None and r["label"] == 0 and p >= HIGH:
            signals[i].append(f"the blend gives it {p:.2f} out of fold")
        if p is not None and r["label"] == 1 and p <= LOW:
            signals[i].append(f"the blend gives it {p:.2f} out of fold")
        if r["label"] == 1 and r["tc"] is not None and r["tc"] > FAMILY_RECORD_K.get(r["family"], 40.0):
            signals[i].append(f"Tc {r['tc']:.1f} K is above its family's ambient record "
                              f"({FAMILY_RECORD_K.get(r['family'], 40.0):.0f} K)")
    flagged = {"possible false negatives": [], "possible false positives": []}
    skipped = Counter()
    for i, why in signals.items():
        r = rows[i]
        kind = "possible false negatives" if r["label"] == 0 else "possible false positives"
        if r["key"] in reviewed:
            skipped[kind] += 1
            continue
        p = oof.get(r["key"]) if oof else None
        flagged[kind].append({"composition": r["key"], "formula": C.formula_text(C.fractions_of(r["key"])),
                              "label": r["label"], "tc_k": r["tc"], "family": r["family"], "split": r["split"],
                              "blend_oof": None if p is None else round(p, 3), "signals": len(why), "why": why})
    for kind, items in flagged.items():
        strength = (lambda x: x["blend_oof"] or 0.0) if kind == "possible false negatives" else \
                   (lambda x: 1.0 - (x["blend_oof"] if x["blend_oof"] is not None else 1.0))
        items.sort(key=lambda x: (-x["signals"], -strength(x), x["composition"]))
    return {"rows_scanned": len(rows), "sources": {n: (None if s is None else len(s)) for n, s in sources.items()},
            "out_of_fold_rows": None if oof is None else len(oof), "already_reviewed_skipped": dict(skipped),
            "probable duplicates": duplicates, **flagged}


def markdown(report: dict, top: int = 25) -> str:
    lines = ["# Label audit", "",
             f"Rows scanned: {report['rows_scanned']:,} (development and train-only; locked rows are never read).", "",
             "| Source | Compositions |", "| --- | --- |"]
    for name, n in report["sources"].items():
        lines.append(f"| {name} | {'not available here' if n is None else f'{n:,}'} |")
    lines.append(f"| out-of-fold scores (ranking of record) | "
                 f"{'not available here' if report['out_of_fold_rows'] is None else f'{report['out_of_fold_rows']:,}'} |")
    lines += ["", f"Already reviewed, so not listed: {report['already_reviewed_skipped'] or 'none'}.", ""]
    lines += [f"## Probable duplicates with opposite labels ({len(report['probable duplicates'])})", ""]
    for d in report["probable duplicates"][:top]:
        lines.append(f"- `{d['first']}` ({d['labels'][0]}) and `{d['second']}` ({d['labels'][1]}), "
                     f"{d['distance']} apart")
    for kind in ("possible false negatives", "possible false positives"):
        items = report[kind]
        by = Counter(x["signals"] for x in items)
        lines += ["", f"## {kind.capitalize()} ({len(items)}; by number of signals: "
                  + ", ".join(f"{k}: {by[k]}" for k in sorted(by, reverse=True)) + ")", "",
                  "| Formula | Label | Tc (K) | Family | Blend out of fold | Signals | Why |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for x in items[:top]:
            lines.append(f"| {x['formula']} | {x['label']} | {'' if x['tc_k'] is None else round(x['tc_k'], 2)} | "
                         f"{x['family']} | {'' if x['blend_oof'] is None else x['blend_oof']} | {x['signals']} | "
                         f"{'; '.join(x['why'])} |")
    lines += ["", "A flag is a lead, not a correction. A label changes only under the owner's rule "
              "(docs/label-audit.md). To stop a checked composition being listed again, add it to "
              "`data/training/label-audit-reviewed.csv`."]
    return "\n".join(lines) + "\n"


def write(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "label-audit.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with open(out / "flags.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["kind", "formula", "composition", "label", "tc_k", "family", "split", "blend_oof", "signals", "why"])
        for kind in ("possible false negatives", "possible false positives"):
            for x in report[kind]:
                w.writerow([kind, x["formula"], x["composition"], x["label"], x["tc_k"], x["family"], x["split"],
                            x["blend_oof"], x["signals"], " | ".join(x["why"])])
    (out / "label-audit.md").write_text(markdown(report), encoding="utf-8")


def load_extract(path: Path = EXTRACT) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if path.is_file():
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out.setdefault(row["source"], {})[row["composition"]] = int(row["label"])
    return out


def write_extract(rows: list[dict], sources: dict[str, dict[str, int] | None], path: Path = EXTRACT) -> int:
    """Keep only the scanned (unlocked) compositions, so no locked row's outside label is ever committed."""
    keys = {r["key"] for r in rows}
    n = 0
    with gzip.GzipFile(path, "wb", mtime=0) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["source", "composition", "label"])
        for name in ("NIMS SuperCon", "NEMAD (usable subset)"):
            for key, label in sorted((sources.get(name) or {}).items()):
                if key in keys:
                    w.writerow([name, key, label])
                    n += 1
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summary", type=Path, default=None, help="append the Markdown report here (CI job summary)")
    parser.add_argument("--include-reviewed", action="store_true",
                        help="list compositions already reviewed or triaged too, instead of only counting them")
    parser.add_argument("--write-extract", action="store_true",
                        help=f"write the NIMS and NEMAD labels of the scanned compositions to {EXTRACT.name} and stop")
    args = parser.parse_args(argv)
    ds = C.load_dataset()        # refuses a table whose checksum differs from docs/scratchnet-data.json: the locked
    rows = table_rows(ds)        # rows would then differ from the real ones, and their labels must never be printed
    sources = {"NIMS SuperCon": nims_labels(), "NEMAD (usable subset)": nemad_labels(), "SuperCon2": supercon2_positive()}
    if args.write_extract:
        print(f"wrote {write_extract(rows, sources)} rows to {EXTRACT}")
        return 0
    extract = load_extract()
    for name in ("NIMS SuperCon", "NEMAD (usable subset)"):
        if sources[name] is None and name in extract:        # no raw file here: use the committed extract
            sources[name] = extract[name]
    report = audit(rows, sources, blend_oof(), set() if args.include_reviewed else reviewed_keys())
    report["table_sha256"] = ds.sha256
    write(report, args.out)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(markdown(report))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
