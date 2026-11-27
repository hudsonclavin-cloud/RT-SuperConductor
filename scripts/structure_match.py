#!/usr/bin/env python3
"""Match compounds that have no structure in the training data to the Crystallography Open Database (COD) and to the
Materials Project data held here (plan and results: docs/structure-match.md).

    python scripts/structure_match.py fetch      # COD, one query per element set; resumable, cached in data/raw/cod/
    python scripts/structure_match.py match      # match compositions, write the tables

Input: docs/perplexity/structures.csv (3,345 ordered non-cuprate compounds without a 3DSC structure; no locked row).
A COD or Materials Project entry matches a compound when every element's atomic fraction is within 0.02 ("close");
within 0.005 it is "exact". COD entries flagged as duplicates or on hold are skipped. A compound's status is the best of:
* "experimental (COD)": a COD entry matches (a diffraction-refined structure from a paper);
* "experimental via MP": a Materials Project entry that cites ICSD records matches;
* "calculated only (MP)": only a Materials Project entry with no experimental source matches;
* "none".
More than one space group among the experimental matches is reported as polymorphs. No label, Tc or score is read.
COD data are CC0; Materials Project data are CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402

INPUT = ROOT / "docs/perplexity/structures.csv"
CACHE = ROOT / "data/raw/cod"
MP = ROOT / "data/candidates/mp-stable-summary.csv.gz"
MATCHES = ROOT / "data/processed/structures/matches.csv.gz"
SUMMARY_CSV = ROOT / "docs/structure-match/summary.csv"
SUMMARY_JSON = ROOT / "docs/structure-match/summary.json"
COD_URL = "https://www.crystallography.net/cod/result"
CLOSE, EXACT = 0.02, 0.005
KEEP = ("file", "formula", "calcformula", "sg", "sgNumber", "a", "b", "c", "alpha", "beta", "gamma", "Z", "vol", "year",
        "doi", "journal", "diffrtemp", "diffrpressure", "cellpressure", "method", "flags", "status", "duplicateof",
        "onhold", "mineral")


def fractions_of_key(key: str) -> dict[str, float]:
    return {e: float(v) for e, v in (part.split(":") for part in key.split(";"))}


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a.get(e, 0.0) - b.get(e, 0.0)) for e in set(a) | set(b))


def compounds() -> list[dict]:
    with open(INPUT, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["fractions"] = fractions_of_key(r["composition_key"])
        r["elements"] = tuple(sorted(r["fractions"]))
    return rows


def cache_path(elements: tuple[str, ...]) -> Path:
    return CACHE / ("-".join(elements) + ".json")


def fetch_one(elements: tuple[str, ...]) -> list[dict]:
    import subprocess

    params = {f"el{i + 1}": e for i, e in enumerate(elements)}
    params.update({"strictmin": len(elements), "strictmax": len(elements), "format": "json"})
    url = f"{COD_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        result = subprocess.run(["curl", "-sf", "--max-time", "180", url], capture_output=True)
        if result.returncode == 0:
            text = result.stdout.decode("utf-8", "replace").strip()
            data = json.loads(text) if text else []
            return [{k: e.get(k) for k in KEEP} for e in data]
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"COD query failed: {url}")


def cmd_fetch(pause: float = 1.0) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    sets = sorted({r["elements"] for r in compounds()})
    todo = [s for s in sets if not cache_path(s).is_file()]
    print(f"{len(sets)} element sets, {len(todo)} still to fetch", flush=True)
    for i, elements in enumerate(todo, 1):
        try:
            entries = fetch_one(elements)
        except RuntimeError as error:
            print(f"skip {'-'.join(elements)}: {error}", flush=True)
            continue
        cache_path(elements).write_text(json.dumps(entries), encoding="utf-8")
        if i % 50 == 0:
            print(f"{i}/{len(todo)} fetched", flush=True)
        time.sleep(pause)


def cod_fractions(entry: dict) -> dict[str, float] | None:
    text = (entry.get("calcformula") or entry.get("formula") or "").strip().strip("-").strip()
    try:
        return M.fractions(text) if text else None
    except Exception:
        return None


def mp_index() -> dict[tuple[str, ...], list[dict]]:
    index = defaultdict(list)
    with gzip.open(MP, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                fractions = M.fractions(row["formula"])
            except Exception:
                fractions = None
            if fractions:
                index[tuple(sorted(fractions))].append({**row, "fractions": fractions})
    return index


def best_cod(matches: list[dict]) -> dict:
    """Prefer exact composition, ambient pressure, the most common space group among the matches, then the newest."""
    common = Counter(m["sgNumber"] for m in matches)

    def ambient(m):
        p = m.get("diffrpressure") or m.get("cellpressure")
        try:
            return float(p) <= 101.4          # COD gives pressure in kPa
        except (TypeError, ValueError):
            return True                       # not stated: taken as ambient
    return sorted(matches, key=lambda m: (m["distance"] > EXACT, not ambient(m), -common[m["sgNumber"]],
                                          -int(m.get("year") or 0)))[0]


def cmd_match() -> dict:
    rows = compounds()
    mp = mp_index()
    all_matches, summary = [], []
    for r in rows:
        cod = []
        path = cache_path(r["elements"])
        fetched = path.is_file()
        for entry in json.loads(path.read_text(encoding="utf-8")) if fetched else []:
            if entry.get("duplicateof") or entry.get("onhold"):
                continue
            fractions = cod_fractions(entry)
            if not fractions or tuple(sorted(fractions)) != r["elements"]:
                continue
            d = distance(fractions, r["fractions"])
            if d <= CLOSE:
                cod.append({**entry, "distance": round(d, 4)})
        mps = []
        for entry in mp.get(r["elements"], []):
            d = distance(entry["fractions"], r["fractions"])
            if d <= CLOSE:
                mps.append({**entry, "distance": round(d, 4)})
        mp_exp = [m for m in mps if m.get("icsd_ids") or m.get("theoretical") == "False"]
        for m in cod:
            all_matches.append({"id": r["id"], "source": "COD", "entry": m["file"], "formula": m["formula"],
                                "space_group": m["sg"], "sg_number": m["sgNumber"], "distance": m["distance"],
                                "a": m["a"], "b": m["b"], "c": m["c"], "year": m["year"], "doi": m["doi"]})
        for m in mps:
            all_matches.append({"id": r["id"], "source": "MP", "entry": m["mp_id"], "formula": m["formula"],
                                "space_group": m["spacegroup_symbol"], "sg_number": m["spacegroup_number"],
                                "distance": m["distance"], "a": "", "b": "", "c": "", "year": "",
                                "doi": "ICSD" if m in mp_exp else "theoretical"})
        experimental_groups = {str(m["sgNumber"]) for m in cod} | {str(m["spacegroup_number"]) for m in mp_exp}
        status = ("experimental (COD)" if cod else "experimental via MP" if mp_exp
                  else "calculated only (MP)" if mps else "none" if fetched else "not fetched")
        best = best_cod(cod) if cod else None
        mp_best = sorted(mp_exp or mps, key=lambda m: (m["distance"], float(m["energy_above_hull_ev_per_atom"] or 9)))[0] \
            if mps else None
        summary.append({"id": r["id"], "formula": r["formula"], "composition_key": r["composition_key"], "status": status,
                        "cod_matches": len(cod), "cod_id": best["file"] if best else "",
                        "cod_space_group": best["sg"] if best else "", "cod_sg_number": best["sgNumber"] if best else "",
                        "cod_a": best["a"] if best else "", "cod_b": best["b"] if best else "",
                        "cod_c": best["c"] if best else "", "cod_distance": best["distance"] if best else "",
                        "cod_doi": (best["doi"] or "") if best else "",
                        "mp_id": mp_best["mp_id"] if mp_best else "",
                        "mp_space_group": mp_best["spacegroup_symbol"] if mp_best else "",
                        "mp_e_above_hull": mp_best["energy_above_hull_ev_per_atom"] if mp_best else "",
                        "mp_experimental_source": bool(mp_best and mp_best in mp_exp),
                        "experimental_space_groups": len(experimental_groups),
                        "polymorphs": len(experimental_groups) > 1})
    MATCHES.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(MATCHES, "wt", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(all_matches[0]) if all_matches else ["id"])
        w.writeheader()
        w.writerows(all_matches)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    report = {"compounds": len(rows), "status": dict(Counter(s["status"] for s in summary)),
              "with_an_experimental_structure": sum(s["status"].startswith("experimental") for s in summary),
              "polymorphs": sum(s["polymorphs"] for s in summary),
              "cod_exact": sum(1 for s in summary if s["cod_distance"] != "" and float(s["cod_distance"]) <= EXACT),
              "element_sets_fetched": sum(cache_path(e).is_file() for e in {r["elements"] for r in rows}),
              "element_sets": len({r["elements"] for r in rows}),
              "sources": {"COD": COD_URL + " (CC0)", "Materials Project": str(MP.relative_to(ROOT)) + " (CC BY 4.0)"}}
    SUMMARY_JSON.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("match")
    args = parser.parse_args(argv)
    cmd_fetch() if args.command == "fetch" else cmd_match()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
