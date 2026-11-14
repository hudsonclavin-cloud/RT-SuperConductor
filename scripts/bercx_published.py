"""Bercx et al.'s published electron-phonon results as a fallback check on the made-metals ranking.

Source: arXiv:2503.10943 (supplementary section 7, one LaTeX file per material) and Materials Cloud Archive
record 2025.39 (doi:10.24435/materialscloud:n9-a7, CC BY 4.0, Tables A to D). The usage rules are in
docs/bercx-published-check.md and were written before any candidate was matched.

    python scripts/bercx_published.py download   # tables and LaTeX into data/raw/bercx-2025/ (git-ignored)
    python scripts/bercx_published.py join        # writes docs/bercx-made-metals-join.csv and -summary.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bercx-2025"
ARCHIVE = "https://archive.materialscloud.org/records/0wv1s-yqg46/files/{}?download=1"
EPRINT = "https://arxiv.org/e-print/2503.10943v2"
EPRINT_SHA256 = "e2355f7a79b984c819b52cdb22e3ec50f7a6f308fccb0e1c8b0e5abc5a7243f0"
SI_MATERIALS = 250
TABLES = {
    "Table_A.csv": "c497700e6295bd2267f204666fe0978fb6fe248f700a1f1ebea135306bc5eef2",
    "Table_B.csv": "1d643a42ae14fe76ff53248e0b26d7402ef02ed0a0318e784ec4b935b689f323",
    "Table_C.csv": "4604d9753e3cb3eabd1987a3e79a35689fd820266fe8000f3efdac5628299856",
    "Table_D.csv": "e71a430645bc9aee72de2baf20b30a451f0a3c1f5120e7e465d054f211ea9e6b",
}
CANDIDATES = ROOT / "data" / "processed" / "scratchnet" / "round2-candidates-boosting.csv.gz"
MP_SUMMARY = ROOT / "data" / "raw" / "mp-wide" / "mp-wide-summary.csv.gz"
OUT_JOIN = ROOT / "docs" / "bercx-made-metals-join.csv"
OUT_SUMMARY = ROOT / "docs" / "bercx-made-metals-summary.json"
TOP = 500
T4_MIN_TC_K = 5.0
TIERS = ("T4", "T4-low", "T3", "T2", "T1", "T0")


# ---------------------------------------------------------------- parsing the supplementary LaTeX

def _grab(pattern: str, text: str, flags: int = 0) -> str:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _number(text: str) -> float | None:
    try:
        return float(text.replace("~", "").strip())
    except ValueError:
        return None


def _cell(label: str) -> str:
    """Regex for a value cell after a label, allowing both layouts used in the files ("[K]" inside or outside the bold)."""
    return label + r"\s*\}?\s*\[K\]\}?\s*&\s*(\$<\$~?\s*[\d.]+|[-\d.~]+)"


def parse_material(text: str) -> dict | None:
    """One material's supplementary .tex file -> a flat record, or None for files that are not material pages."""
    if r"\subsection{" not in text or "Allen-Dynes" not in text:
        return None
    formula_tex = _grab(r"\\subsection\{(.+?)\}\s*\n", text)
    sg = re.search(r"Space group\}\s*&\s*(.+?)\((\d+)\)", text)
    iso = _grab(_cell(r"Isotropic \$T_c\$"), text)
    aniso = _grab(_cell(r"Anisotropic \$T_c\$"), text)
    return {
        "formula": re.sub(r"[$\{\}_\\ ]|mathrm", "", formula_tex),
        "spacegroup": sg.group(2) if sg else "",
        "source_id": _grab(r"Source database\}\s*&\s*([A-Za-z0-9]+-[A-Za-z0-9]+)", text),
        "mc3d_id": _grab(r"\{(mc3d-\d+)\}", text),
        "category": _grab(r"Category\}\s*&\s*(?:\\multicolumn\{\d\}\{l\}\{)?\s*([A-Za-z\-]+)", text),
        "lambda": _number(_grab(r"boldsymbol\{\\lambda\}\$\}\s*&\s*([-\d.~]+)", text)),
        "omega_log_mev": _number(_grab(r"omega_\{\\rm log\}\}\$\s*\[meV\]\}\s*&\s*([-\d.~]+)", text)),
        "tc_allen_dynes_k": _number(_grab(_cell(r"Allen-Dynes \$T_c\$"), text)),
        "tc_iso_k": _number(iso),
        "tc_iso_text": iso.replace("$<$", "<").replace("~", "").replace(" ", ""),
        "tc_aniso_k": _number(aniso),
        "tc_experimental": " ".join(re.sub(r"\\cite\{[^}]*\}", "",
                                           _grab(r"Experimental \$T_c\$\s*\}?\s*\[K\]\}?\s*&\s*(.+?)\\\\", text)
                                           ).replace("~", " ").replace("$<$", "<").split()),
        "note": re.sub(r"\s+", " ", _grab(r"Note\}\s*&\s*\\multicolumn\{3\}\{p\{11cm\}\}\{(.+?)\}\s*\\\\", text, re.S))[:300],
    }


# ---------------------------------------------------------------- matching and tiers

def composition_key(formula: str) -> str:
    from pymatgen.core import Composition
    return Composition(formula).reduced_formula


def made_metal(row: dict) -> bool:
    """Same rule as scripts/run_scratchnet.py::made_metal (tests/test_bercx_published.py checks they agree)."""
    return (row["is_metal"] == "True" and row["theoretical"] == "False" and row["family"] in ("Other", "Oxides")
            and "Cu" not in row["element_system"].split("-"))


def _si_tier(m: dict) -> str:
    if m["category"] == "problematic":
        return "T3"   # problematic entries carry no verdict, whatever their Tc
    if m["tc_iso_k"] is not None and m["tc_iso_k"] >= T4_MIN_TC_K:
        return "T4"
    return "T4-low"


def tier(key: tuple[str, str], si: dict, table_b: set, table_c: set, table_d: set) -> str:
    """The pre-registered evidence tier for a (reduced formula, space group) key.

    ``si`` maps a key to a list of supplementary records (the same structure can appear more than once). If the
    records disagree on the tier, the compound gets no verdict (T3).
    """
    if key in si:
        tiers = {_si_tier(m) for m in (si[key] if isinstance(si[key], list) else [si[key]])}
        return tiers.pop() if len(tiers) == 1 else "T3"
    if key in table_d:
        return "T3"
    if key in table_c:
        return "T2"
    if key in table_b:
        return "T1"
    return "T0"


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for [[a, b], [c, d]] (sum of tables no more likely than the observed one)."""
    from math import comb
    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)
    p_obs = comb(n1, a) * comb(n2, c) / total
    p = 0.0
    for x in range(max(0, k - n2), min(k, n1) + 1):
        px = comb(n1, x) * comb(n2, k - x) / total
        if px <= p_obs * (1 + 1e-9):
            p += px
    return min(1.0, p)


# ---------------------------------------------------------------- commands

def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in TABLES.items():
        data = urllib.request.urlopen(ARCHIVE.format(name), timeout=120).read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            sys.exit(f"{name}: checksum {got} does not match the recorded {digest}")
        (RAW / name).write_bytes(data)
    blob = urllib.request.urlopen(EPRINT, timeout=300).read()
    got = hashlib.sha256(blob).hexdigest()
    if got != EPRINT_SHA256:
        sys.exit(f"arXiv e-print: checksum {got} does not match the recorded {EPRINT_SHA256}")
    si = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".tex") and "manuscript" not in member.name:
                record = parse_material(tar.extractfile(member).read().decode("utf-8", "replace"))
                if record:
                    si.append(record)
    if len(si) != SI_MATERIALS:
        sys.exit(f"parsed {len(si)} supplementary materials, expected {SI_MATERIALS}")
    with open(RAW / "si-materials.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(si[0]))
        writer.writeheader()
        writer.writerows(sorted(si, key=lambda r: (r["formula"], r["spacegroup"])))
    print(f"{len(si)} materials parsed; tables verified")


def table_d_by_source() -> dict:
    """Source id -> (reduced formula, space group) as Table D gives it."""
    with open(RAW / "Table_D.csv", newline="") as handle:
        return {f"{r['Source Database'].strip()}-{r[' Source ID'].strip()}".lower():
                (composition_key(r["Formula"].strip()), r[" Spacegroup"].strip()) for r in csv.DictReader(handle)}


def si_index(si_rows: list[dict], d_by_source: dict) -> dict:
    """Key -> list of supplementary records. Each record is filed under its own (formula, space group) and, when its
    source id is in Table D with another space group, under Table D's too: 13 supplementary pages print a lower
    symmetry than Table D gives for the same structure."""
    index: dict = {}
    for r in si_rows:
        keys = {(composition_key(r["formula"]), r["spacegroup"])}
        if r["source_id"].lower() in d_by_source:
            keys.add(d_by_source[r["source_id"].lower()])
        for key in keys:
            index.setdefault(key, []).append(r)
    return index


def load_keys(name: str) -> set:
    with open(RAW / name, newline="") as handle:
        return {(composition_key(r["Formula"].strip()), r[" Spacegroup"].strip()) for r in csv.DictReader(handle)}


def join() -> dict:
    with open(RAW / "si-materials.csv", newline="") as handle:
        si_rows = list(csv.DictReader(handle))
    for r in si_rows:
        for k in ("tc_iso_k", "tc_aniso_k", "tc_allen_dynes_k", "lambda", "omega_log_mev"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
    table_b, table_c, table_d = load_keys("Table_B.csv"), load_keys("Table_C.csv"), load_keys("Table_D.csv")
    si = si_index(si_rows, table_d_by_source())
    bercx_formulas = {k[0] for k in table_b} | {k[0] for k in si}
    with gzip.open(MP_SUMMARY, "rt") as handle:
        spacegroup = {r["mp_id"]: r["spacegroup_number"] for r in csv.DictReader(handle)}
    with gzip.open(CANDIDATES, "rt") as handle:
        made = [r for r in csv.DictReader(handle) if made_metal(r)]
    made.sort(key=lambda r: -float(r["p_superconductor"]))   # the made-metals order (stable, as in run_scratchnet)
    rows, counts = [], {"top": {t: 0 for t in TIERS}, "rest": {t: 0 for t in TIERS}}
    formula_only = {"top": 0, "rest": 0}
    for position, r in enumerate(made, start=1):
        key = (composition_key(r["formula"]), spacegroup.get(r["mp_id"], ""))
        t = tier(key, si, table_b, table_c, table_d)
        group = "top" if position <= TOP else "rest"
        counts[group][t] += 1
        if t == "T0" and key[0] in bercx_formulas:
            formula_only[group] += 1
        if t != "T0":
            found = si.get(key, [])
            m = found[0] if len(found) == 1 else {}
            rows.append({"made_metals_rank": position, "mp_id": r["mp_id"], "formula": r["formula"],
                         "spacegroup": key[1], "boosting_p": r["p_superconductor"], "tier": t,
                         "bercx_category": m.get("category", ""), "bercx_lambda": m.get("lambda", ""),
                         "bercx_omega_log_mev": m.get("omega_log_mev", ""),
                         "bercx_tc_allen_dynes_k": m.get("tc_allen_dynes_k", ""),
                         "bercx_tc_iso": m.get("tc_iso_text", ""), "bercx_tc_aniso_k": m.get("tc_aniso_k", ""),
                         "bercx_tc_experimental": m.get("tc_experimental", "")})
    matched = {g: sum(v for t, v in counts[g].items() if t != "T0") for g in counts}
    a, c = counts["top"]["T4"], counts["rest"]["T4"]
    b, d = matched["top"] - a, matched["rest"] - c
    odds = (a * d) / (b * c) if b * c else None
    summary = {"made_metals": len(made), "top_n": TOP, "t4_min_tc_iso_k": T4_MIN_TC_K,
               "tier_counts": counts, "structure_matched": matched,
               "formula_only_not_structure": formula_only,
               "t4_share_among_matched": {"top": round(a / matched["top"], 4) if matched["top"] else None,
                                          "rest": round(c / matched["rest"], 4) if matched["rest"] else None},
               "fisher_t4_top_vs_rest": {"table": [[a, b], [c, d]],
                                         "odds_ratio": round(odds, 3) if odds is not None else None,
                                         "p_two_sided": round(fisher_two_sided(a, b, c, d), 5)}}
    with open(OUT_JOIN, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download", "join"))
    args = parser.parse_args()
    download() if args.command == "download" else join()


if __name__ == "__main__":
    main()
