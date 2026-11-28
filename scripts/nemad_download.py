#!/usr/bin/env python3
"""Download the NEMAD superconductor database (Itani et al., J. Magn. Magn. Mater. 652, 174189, 2026) and turn it into a
table this project can read.

    python scripts/nemad_download.py fetch     # one request per element, saved under data/raw/nemad/ (ignored by git)
    python scripts/nemad_download.py table     # data/processed/nemad/superconductor.csv + data/processed/nemad/table.json

The API (https://nemad.org/backend/openapi.json) needs a key in the ``X-API-Key`` header. In the cloud session the network
proxy adds it (a network secret for nemad.org); elsewhere set the environment variable NEMAD_API_KEY. The key is never
written to disk or logged.

The API searches by element, so ``fetch`` asks once per element for every superconductor record containing it and
``table`` merges them, dropping exact duplicates. Records whose names do not parse as formulas are counted and kept out.
The site describes its data as CC BY 4.0; cite the paper and NEMAD. Every record was extracted from papers by language
models, so treat each label as a lead to check, as with SuperCon2 (docs/data-sources.md).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nemad/superconductor"
OUT = ROOT / "data/processed/nemad"
BASE = "https://nemad.org/backend"
FIELDS = ["formula_text", "label", "tc_k", "tc_text", "pressure_text", "ambient", "experimental_flag", "material_type",
          "superconductor_type", "material_class", "doi", "material_name"]


def elements() -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from pymatgen.core.periodic_table import Element

    return [Element.from_Z(z).symbol for z in range(1, 104)]


def get(path: str, params: dict, retries: int = 4) -> dict:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/json", "User-Agent": "rt-superconductor-data-check/1.0"}
    if os.environ.get("NEMAD_API_KEY"):
        headers["X-API-Key"] = os.environ["NEMAD_API_KEY"]
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise SystemExit(f"{error.code} from {BASE}{path}: the API key is missing or not accepted") from error
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(5 * 2 ** attempt)
    raise AssertionError("unreachable")


def cmd_fetch(pause: float = 1.0) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    counts = {}
    for symbol in elements():
        path = RAW / f"{symbol}.json.gz"
        if path.exists():                                   # resumable
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                counts[symbol] = len(json.load(handle).get("results", []))
            continue
        data = get("/api/superconductor/search", {"elements": symbol, "exact_match": "false", "limit": -1})
        if "results" not in data:
            raise SystemExit(f"unexpected answer for {symbol}: {str(data)[:200]}")
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(data, handle)
        tmp.replace(path)
        counts[symbol] = len(data["results"])
        print(f"{symbol}: {counts[symbol]}", flush=True)
        time.sleep(pause)
    return {"per_element": counts}


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def tc_value(text) -> float | None:
    """A single Tc in kelvin from text such as '30 K' or '4.2'; None for ranges, inequalities, other units or nothing."""
    if text is None:
        return None
    text = str(text).strip()
    if not text or re.search(r"[<>~≈±]|-\s*\d|–|to\b|mK|°C|\bC\b", text):
        return None
    numbers = _NUMBER.findall(text)
    if len(numbers) != 1:
        return None
    value = float(numbers[0])
    return value if 0 <= value < 400 else None


def label_of(record: dict) -> int | None:
    kind = str(record.get("Material_Type") or "").strip().upper()
    if kind in ("SC", "SUPERCONDUCTOR"):
        return 1
    if kind in ("NSC", "NON-SC", "NON-SUPERCONDUCTOR", "NONSUPERCONDUCTOR", "NOT SC"):
        return 0
    return None


def ambient(pressure) -> bool | None:
    if pressure in (None, ""):
        return None
    text = str(pressure).strip().lower()
    if text in ("ambient", "0", "0 gpa", "1 atm", "atmospheric", "ambient pressure", "101325", "101325 pa"):
        return True
    return False


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")


def clean_name(name: str) -> str:
    """A record name made readable as a formula, conservatively:
    * subscript digits to plain digits, and Unicode minus signs to '-';
    * trailing notes in brackets dropped ('HfV2 (polycrystalline)'), and polytype prefixes ('2H-', '1T-');
    * oxygen off-stoichiometry set to its ideal value ('YBa2Cu3O7-δ' -> 'YBa2Cu3O7');
    * a doping variable replaced only when the name gives it exactly one value ('La2-xSrxCuO4 (x=0.15)').
    Anything else (words, ranges, unknown x) is left as it is and fails to parse later."""
    t = str(name).translate(_SUBSCRIPTS).replace("\u2212", "-").replace("\u2013", "-").replace("\u2009", "").strip()
    t = re.sub(r"\(.*?Tc.*?\)", "", t).strip()                                   # 'X (Tc~30K)'
    values = dict(re.findall(r"\b([xyz])\s*=\s*(\d*\.?\d+)\b(?![\s,]*(?:\d|and|to|-))", t))
    while re.search(r"\s+\([^()]*\)\s*$", t):
        note = re.search(r"\s+\(([^()]*)\)\s*$", t).group(1)
        if re.search(r"(?i)dop|substitut|intercal|added|addition|with|\+|annealed in|irradiat", note):
            return ""                                    # the note changes the composition: not usable
        t = re.sub(r"\s+\([^()]*\)\s*$", "", t)
    t = re.sub(r"^\d+[A-Za-z]{1,2}-(?=[A-Z(])", "", t)
    t = re.sub(r"[-+±]\s*δ", "", t)                       # δ marks a small off-stoichiometry: use the ideal value
    for var, val in values.items():
        v = float(val)
        t = re.sub(rf"(\d*\.?\d+)\s*-\s*{var}(?![a-z])", lambda m: f"{float(m.group(1)) - v:g}", t)
        t = re.sub(rf"(\d*\.?\d+)\s*\+\s*{var}(?![a-z])", lambda m: f"{float(m.group(1)) + v:g}", t)
        t = re.sub(rf"(?<=[A-Za-z)]){var}(?![a-z])", f"{v:g}", t)
    return t.strip()


def formula_of(record: dict) -> tuple[str | None, dict | None]:
    """The record's formula as atomic fractions, from Chemical_Composition, else Material_Name; None if it does not parse."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_mp_candidates as M

    for field in ("Chemical_Composition", "Material_Name"):
        text = record.get(field)
        if not text:
            continue
        text = clean_name(text)
        if not text or re.search(r"[a-z]{3,}|\s", text):        # words or spaces left: not a formula
            continue
        fractions = M.fractions(text)
        if fractions and len(fractions) == 1 and re.search(r"\d{2,}", text):
            continue                                             # nicknames such as 'Cu1234', 'Sm123', 'Bi2212'
        if fractions:
            return text, fractions
    return None, None


def cmd_table() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from rt_superconductor.scratchnet.data import composition_key, write_table

    seen, rows, counts = set(), [], Counter()
    files = sorted(RAW.glob("*.json.gz"))
    if not files:
        raise SystemExit("nothing fetched yet: run `python scripts/nemad_download.py fetch`")
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            results = json.load(handle).get("results", [])
        for record in results:
            digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            counts["records"] += 1
            text, fractions = formula_of(record)
            if fractions is None:
                counts["formula_not_usable"] += 1
                continue
            label = label_of(record)
            counts[f"label_{label}"] += 1
            tc = tc_value(record.get("Superconducting_Transition_Temperature"))
            rows.append({"formula_text": text, "composition": composition_key(fractions), "label": "" if label is None else label,
                         "tc_k": "" if tc is None else tc, "tc_text": record.get("Superconducting_Transition_Temperature") or "",
                         "pressure_text": record.get("Pressure") or "", "ambient": ambient(record.get("Pressure")),
                         "experimental_flag": record.get("Experimental"), "material_type": record.get("Material_Type") or "",
                         "superconductor_type": record.get("Superconductor_Type") or "",
                         "material_class": record.get("Material_Class") or "", "doi": record.get("DOI") or "",
                         "material_name": record.get("Material_Name") or ""})
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(OUT / "superconductor.csv", rows, ["composition"] + FIELDS)
    reported = get("/materials/superconductor_number", {})
    report = {"source": "NEMAD superconductor database, https://nemad.org (Itani et al. 2026)",
              "files": len(files), "unique_records": counts["records"], "reported_total": reported,
              "formula_not_usable": counts["formula_not_usable"], "usable_rows": len(rows),
              "unique_compositions": len({r["composition"] for r in rows}),
              "labels": {k: v for k, v in counts.items() if k.startswith("label_")},
              "material_types": dict(Counter(r["material_type"] for r in rows).most_common(10)),
              "experimental_flag": dict(Counter(str(r["experimental_flag"]) for r in rows)),
              "ambient": dict(Counter(str(r["ambient"]) for r in rows)),
              "tc_parsed": sum(r["tc_k"] != "" for r in rows)}
    (OUT / "table.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    sub.add_parser("table")
    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(cmd_fetch(args.pause)["per_element"]))
    else:
        cmd_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
