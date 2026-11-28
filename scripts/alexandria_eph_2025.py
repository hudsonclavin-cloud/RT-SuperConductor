#!/usr/bin/env python3
"""The Alexandria PBEsol phonon and electron-phonon release of 2025.08.11, reduced to a slim table (owner approved the
5 GB download on 2026-11-27; survey: docs/training-data-sources.md).

    python scripts/alexandria_eph_2025.py            # resumable: files already reduced are skipped
    python scripts/alexandria_eph_2025.py --combine  # only rebuild the combined table and summary

Source: https://alexandria.icams.rub.de/data/pbesol_ph/3d/alexandria_ph_000..092.json.bz2 (CC BY 4.0; Gao et al.,
Nat. Commun. 2025, 10.1038/s41467-025-63702-w; Cerqueira et al.). Each file is downloaded, reduced and deleted, one at a
time, so the raw 5 GB never sits on disk. Kept per entry: mat_id, formula, composition key, number of sites, the
imaginary-mode flag, lambda, omega_log, omega_2, and Tc by McMillan, Allen-Dynes and isotropic Eliashberg at mu* = 0.10
and 0.13 (0.13 interpolated between the 0.12 and 0.14 grid points), plus the relaxed structure in a separate file.
Every value is calculated (QE DFPT, PBEsol, harmonic); none is a measurement, and none enters the ranking's labels.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

BASE = "https://alexandria.icams.rub.de/data/pbesol_ph/3d/alexandria_ph_{:03d}.json.bz2"
FILES = 93
WORK = ROOT / "data/raw/alexandria-eph-2025"          # git-ignored; raw files are deleted after reduction
OUT = ROOT / "data/processed/alexandria-eph-2025"
SUMMARY = ROOT / "docs/alexandria-eph-2025.json"


def at_mu(values: list[float], grid: list[float], mu: float) -> float | None:
    """Linear interpolation on the mu* grid."""
    if not values or not grid:
        return None
    for i in range(len(grid) - 1):
        if grid[i] <= mu <= grid[i + 1]:
            w = (mu - grid[i]) / (grid[i + 1] - grid[i])
            lo, hi = values[i], values[i + 1]
            if lo is None or hi is None:            # a gap in the source's Tc array: only an exact grid point is usable
                exact = lo if w < 1e-9 else hi if w > 1 - 1e-9 else None
                return None if exact is None else round(exact, 4)
            return round(lo * (1 - w) + hi * w, 4)
    return None


def reduce_entry(entry: dict) -> tuple[dict, dict]:
    import build_mp_candidates as M
    import build_scratchnet_data as B

    data = entry["data"]
    fractions = M.fractions(data["formula"])
    row = {"mat_id": data["mat_id"], "formula": data["formula"],
           "composition": B.composition_key(fractions) if fractions else "", "nsites": data.get("nsites"),
           "imag": bool(data.get("imag"))}
    tc = data.get("tc") or {}
    if tc:
        grid = tc.get("mustr") or []
        row.update({"lambda": round(tc["lambda"], 4), "wlog_k": round(tc["wlog[K]"], 2), "w2av_k": round(tc["w2av[K]"], 2)})
        for name, key in (("mcmillan", "TcMcMillan"), ("allen_dynes", "TcAllenDynes"), ("eliashberg", "TcEliashberg")):
            row[f"tc_{name}_mu010"] = at_mu(tc.get(key), grid, 0.10)
            row[f"tc_{name}_mu013"] = at_mu(tc.get(key), grid, 0.13)
    return row, {"mat_id": data["mat_id"], "structure": entry["structure"]}


def reduce_file(index: int) -> dict:
    part = OUT / f"part-{index:03d}.jsonl.gz"
    if part.is_file():
        return {"file": index, "skipped": True}
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = WORK / f"alexandria_ph_{index:03d}.json.bz2"
    for attempt in range(4):
        result = subprocess.run(["curl", "-sfL", "--max-time", "900", "-o", str(raw), BASE.format(index)])
        if result.returncode == 0:
            break
        time.sleep(2 ** (attempt + 2))
    else:
        raise SystemExit(f"download failed for file {index:03d}")
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    entries = json.load(bz2.open(raw, "rt"))["entries"]
    rows, structures = zip(*(reduce_entry(e) for e in entries)) if entries else ((), ())
    tmp = part.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with gzip.open(OUT / f"structures-{index:03d}.jsonl.gz", "wt", encoding="utf-8") as handle:
        for s in structures:
            handle.write(json.dumps(s) + "\n")
    tmp.rename(part)
    size = raw.stat().st_size
    raw.unlink()
    info = {"file": index, "entries": len(rows), "with_tc": sum("lambda" in r for r in rows), "bytes": size, "sha256": sha}
    with open(OUT / "files.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(info) + "\n")
    return info


def combine() -> dict:
    import csv

    rows = []
    for part in sorted(OUT.glob("part-*.jsonl.gz")):
        with gzip.open(part, "rt", encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle]
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("mat_id", "formula", "composition"), k))
    with gzip.open(OUT / "table.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    rows = list({r["mat_id"]: r for r in rows}.values())       # a few entries appear in two files
    with_tc = [r for r in rows if "lambda" in r and not r["imag"]]
    held = ROOT / "data/raw/alexandria-epc"
    held_ids, held_keys = set(), set()
    if held.is_dir():
        import re

        import build_mp_candidates as M
        import build_scratchnet_data as B
        import physics_proxy2 as P

        for r in P.alexandria_rows()[0]:
            found = re.search(r"agm\d+", r["id"])
            if found:
                held_ids.add(found.group(0))
            fractions = M.fractions(r["formula"])
            if fractions:
                held_keys.add(B.composition_key(fractions))
    # lambda above 5 or omega_log below 100 K is outside what harmonic DFPT describes reliably (e.g. a soft mode at the
    # edge of instability); such rows are kept but flagged
    suspect = [r for r in with_tc if r["lambda"] > 5 or r["wlog_k"] < 100]
    summary = {"source": BASE.format(0).rsplit("/", 1)[0], "release": "2025.08.11", "licence": "CC BY 4.0",
               "files_reduced": len(list(OUT.glob("part-*.jsonl.gz"))), "files_expected": FILES,
               "entries": len(rows), "stable_with_tc": len(with_tc),
               "unique_compositions_with_tc": len({r["composition"] for r in with_tc if r["composition"]}),
               "held_before": {"rows": len(held_ids), "compositions": len(held_keys)},
               "new_mat_ids_with_tc": sum(r["mat_id"] not in held_ids for r in with_tc),
               "new_compositions_with_tc": len({r["composition"] for r in with_tc if r["composition"]} - held_keys),
               "flagged_lambda_above_5_or_wlog_below_100K": len(suspect),
               "tc_allen_dynes_mu010_at_least": {str(t): sum((r.get("tc_allen_dynes_mu010") or 0) >= t for r in with_tc)
                                                 for t in (10, 20, 40, 77)},
               "tc_allen_dynes_mu010_at_least_77K_not_flagged": sorted(
                   ({"formula": r["formula"], "mat_id": r["mat_id"], "nsites": r["nsites"], "lambda": r["lambda"],
                     "wlog_k": r["wlog_k"], "tc_allen_dynes_mu010": r["tc_allen_dynes_mu010"],
                     "tc_eliashberg_mu010": r["tc_eliashberg_mu010"]}
                    for r in with_tc if (r.get("tc_allen_dynes_mu010") or 0) >= 77 and r not in suspect),
                   key=lambda x: -x["tc_allen_dynes_mu010"]),
               "held_alexandria_dir_present": held.is_dir(),
               "note": "Calculated values only (QE DFPT, PBEsol, harmonic). Not labels."}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=FILES - 1)
    args = parser.parse_args(argv)
    if not args.combine:
        for index in range(args.first, args.last + 1):
            started = time.time()
            info = reduce_file(index)
            print(json.dumps({**info, "seconds": round(time.time() - started, 1)}), flush=True)
    print(json.dumps(combine(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
