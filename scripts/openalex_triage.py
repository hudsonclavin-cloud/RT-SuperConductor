#!/usr/bin/env python3
"""Literature triage of training negatives through the OpenAlex API (cycle 3 of docs/learning-cycles.md).

For each chosen negative (a composition labelled "not a superconductor"), search OpenAlex once and keep the works
whose title or abstract mentions superconductivity and names every element of the composition. A row with such a
work is a *lead*: it goes to readers and a verifier under the owner's rule. A lead is not evidence, and a row
with no lead is not a verified null.

Which negatives: development rows that are not copper oxides, ordered by the previous cycle's out-of-fold
probability of superconducting (the model's suspicion), the top ``--top``. Locked rows are not used.

One query per element system, by element names ("superconductivity indium thallium alloy"; "alloy" is added
when the atom counts are not small whole numbers). Rows of the same system share the query and its answer.

Search costs 0.001 USD of the key's daily budget per query (1 USD a day on the free tier). Answers are cached under
``data/processed/learning-cycles/openalex-cache.json`` so a rerun spends nothing on queries it has made.

    python scripts/openalex_triage.py --previous cycle1 --top 500 --out data/benchmarks/learning-cycles/cycle3-triage.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT

WORK = ROOT / "data/processed/learning-cycles"
CACHE = WORK / "openalex-cache.json"
API = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_year,abstract_inverted_index"
ELEMENT_NAMES = {
    "H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium", "B": "boron", "C": "carbon", "N": "nitrogen",
    "O": "oxygen", "F": "fluorine", "Na": "sodium", "Mg": "magnesium", "Al": "aluminum", "Si": "silicon",
    "P": "phosphorus", "S": "sulfur", "Cl": "chlorine", "K": "potassium", "Ca": "calcium", "Sc": "scandium",
    "Ti": "titanium", "V": "vanadium", "Cr": "chromium", "Mn": "manganese", "Fe": "iron", "Co": "cobalt",
    "Ni": "nickel", "Cu": "copper", "Zn": "zinc", "Ga": "gallium", "Ge": "germanium", "As": "arsenic",
    "Se": "selenium", "Br": "bromine", "Rb": "rubidium", "Sr": "strontium", "Y": "yttrium", "Zr": "zirconium",
    "Nb": "niobium", "Mo": "molybdenum", "Tc": "technetium", "Ru": "ruthenium", "Rh": "rhodium", "Pd": "palladium",
    "Ag": "silver", "Cd": "cadmium", "In": "indium", "Sn": "tin", "Sb": "antimony", "Te": "tellurium", "I": "iodine",
    "Cs": "cesium", "Ba": "barium", "La": "lanthanum", "Ce": "cerium", "Pr": "praseodymium", "Nd": "neodymium",
    "Pm": "promethium", "Sm": "samarium", "Eu": "europium", "Gd": "gadolinium", "Tb": "terbium", "Dy": "dysprosium",
    "Ho": "holmium", "Er": "erbium", "Tm": "thulium", "Yb": "ytterbium", "Lu": "lutetium", "Hf": "hafnium",
    "Ta": "tantalum", "W": "tungsten", "Re": "rhenium", "Os": "osmium", "Ir": "iridium", "Pt": "platinum",
    "Au": "gold", "Hg": "mercury", "Tl": "thallium", "Pb": "lead", "Bi": "bismuth", "Po": "polonium",
    "Th": "thorium", "Pa": "protactinium", "U": "uranium", "Np": "neptunium", "Pu": "plutonium", "Am": "americium"}


def small_formula(fractions: dict[str, float], max_atoms: int = 12) -> str | None:
    """'YRu3B2' if the fractions are small whole-number ratios (at most ``max_atoms`` atoms), else None."""
    for total in range(1, max_atoms + 1):
        counts = {el: value * total for el, value in fractions.items()}
        if all(abs(c - round(c)) <= 0.0015 * total and round(c) >= 1 for c in counts.values()):
            ints = {el: int(round(c)) for el, c in counts.items()}
            return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(ints.items(), key=lambda kv: kv[0]))
    return None


def query_for(fractions: dict[str, float]) -> tuple[str, str]:
    """(kind, query), one per element system, by element names: "superconductivity indium thallium alloy".

    Names, not symbols: several symbols are English words that a search engine drops (In, As, Be), and a formula
    has many spellings (YRu3B2, YB2Ru3, Y(Ru3B2)). Compounds and alloys of one system share the query, so each
    system costs one search; a reader then checks the exact composition.
    """
    names = " ".join(ELEMENT_NAMES.get(el, el) for el in sorted(fractions))
    kind = "compound" if small_formula(fractions) is not None else "alloy"
    return kind, f"superconductivity {names}" + (" alloy" if kind == "alloy" else "")


def abstract_text(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, places in inverted.items() for pos in places]
    return " ".join(word for _, word in sorted(positions))


def is_lead(work: dict, elements: set[str]) -> bool:
    """Mentions superconductivity, and names every element: by name, or by symbol inside a formula-like token
    (one with a digit or several symbols, such as In0.95Tl0.05 or YRu3B2). Bare symbols are not trusted
    because several are English words (In, As, Be, No)."""
    text = f"{work.get('title') or ''} {abstract_text(work.get('abstract_inverted_index'))}"
    lower = text.lower()
    if "superconduct" not in lower:
        return False
    in_formulas: set[str] = set()
    for token in re.split(r"[\s,;:()\[\]/]+", text):
        symbols = re.findall(r"[A-Z][a-z]?", token)
        if len(symbols) >= 2 or (symbols and any(ch.isdigit() for ch in token)):
            in_formulas.update(symbols)
    aliases = {"aluminum": ("aluminium",), "sulfur": ("sulphur",), "cesium": ("caesium",)}
    for el in elements:
        name = ELEMENT_NAMES.get(el, "")
        named = bool(name) and (name in lower or any(a in lower for a in aliases.get(name, ())))
        if not (named or el in in_formulas):
            return False
    return True


def search(query: str, cache: dict, per_page: int = 10) -> tuple[list[dict], float | None]:
    if query in cache:
        return cache[query], None
    url = f"{API}?{urllib.parse.urlencode({'search': query, 'per-page': per_page, 'select': SELECT})}"
    request = urllib.request.Request(url, headers={"User-Agent": "rt-superconductor-triage/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                remaining = response.headers.get("X-RateLimit-Remaining-USD")
                data = json.loads(response.read())
            cache[query] = data.get("results", [])
            return cache[query], float(remaining) if remaining else None
        except Exception:            # noqa: BLE001 - network: retry, then give up on this query
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return [], None


def choose_rows(previous: str, top: int) -> list[dict]:
    ds = C.load_dataset()
    with np.load(WORK / f"{previous}-oof.npz") as saved:
        p = dict(zip((str(k) for k in saved["compositions"]), saved["p"]))
    keep = [row for row in ds.rows if row["split"] == "dev" and row["label"] == "0" and row["family"] != "Cuprates"]
    keep.sort(key=lambda row: -p[row["composition"]])
    return [{**row, "oof_p": round(float(p[row["composition"]]), 4)} for row in keep[:top]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--previous", required=True, help="the cycle whose out-of-fold probabilities rank the rows")
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--stop-below-usd", type=float, default=0.02, help="stop when the key's budget falls this low")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = choose_rows(args.previous, args.top)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    results, spent_out = [], False
    for n, row in enumerate(rows, 1):
        fractions = C.fractions_of(row["composition"])
        kind, query = query_for(fractions)
        if spent_out and query not in cache:
            results.append({**_row(row), "query_kind": kind, "query": query, "status": "not_searched_budget"})
            continue
        works, remaining = search(query, cache)
        if remaining is not None and remaining < args.stop_below_usd:
            spent_out = True
        leads = [{"openalex_id": w.get("id"), "doi": w.get("doi"), "title": w.get("title"),
                  "year": w.get("publication_year"),
                  "abstract": abstract_text(w.get("abstract_inverted_index"))[:1200]}
                 for w in works if is_lead(w, set(fractions))]
        results.append({**_row(row), "query_kind": kind, "query": query, "status": "searched",
                        "works_returned": len(works), "leads": leads})
        if n % 25 == 0:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"{n}/{len(rows)} rows; budget left {remaining}", flush=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    summary = {"previous": args.previous, "top": args.top, "rows": len(results),
               "searched": sum(r["status"] == "searched" for r in results),
               "rows_with_leads": sum(bool(r.get("leads")) for r in results),
               "queries": len({r["query"] for r in results}), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    return 0


def _row(row: dict) -> dict:
    return {"composition": row["composition"], "system": row["system"], "fold": int(row["fold"]),
            "oof_p": row["oof_p"], "n_reports": row["n_reports"], "lineages": row["lineages"]}


if __name__ == "__main__":
    raise SystemExit(main())
