#!/usr/bin/env python3
"""Download the pinned UCI and 3DSC inputs used by the active-learning loop.

Raw datasets are ignored by git because they are large and have source-specific
redistribution terms. The UCI archive and 3DSC table are hash-checked. CIFs are
downloaded from the pinned 3DSC Git commit; CI uses a deterministic stratified
subset, while ``--all-cifs`` is available for a complete local structure run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


UCI_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
UCI_SHA256 = "87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e"
THREEDSC_COMMIT = "8d69ca9c94b83d387378549225d3b7c3af85ca42"
THREEDSC_TABLE_URL = f"https://raw.githubusercontent.com/aimat-lab/3DSC/{THREEDSC_COMMIT}/superconductors_3D/data/final/MP/3DSC_MP.csv"
THREEDSC_TABLE_SHA256 = "353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24"
THREEDSC_LICENSE = "CC BY 4.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None = None) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected_sha256 is None or _sha256(destination) == expected_sha256):
        return False
    with tempfile.NamedTemporaryFile(prefix=destination.name, dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "RT-SuperConductor/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected_sha256 is not None and _sha256(temporary) != expected_sha256:
            raise ValueError(f"sha256 mismatch for {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def _read_3dsc_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        if not first:
            return []
        if not first.lstrip().startswith("#"):
            handle.seek(0)
        return list(csv.DictReader(handle, skipinitialspace=True))


def _select_cifs(rows: list[dict[str, str]], limit: int | None) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = (row.get("cif") or "").replace("\\", "/").strip()
        if not value:
            continue
        filename = Path(value).name
        grouped[(row.get("sc_class") or "Unknown")].append(filename)
    for values in grouped.values():
        values[:] = list(dict.fromkeys(values))
    selected: list[str] = []
    keys = sorted(grouped)
    index = 0
    while keys and (limit is None or len(selected) < limit):
        key = keys[index % len(keys)]
        values = grouped[key]
        if values:
            selected.append(values.pop(0))
        keys = [item for item in keys if grouped[item]]
        index += 1
    return selected


def _download_cif(name: str, destination: Path) -> tuple[str, str | None]:
    url = f"https://raw.githubusercontent.com/aimat-lab/3DSC/{THREEDSC_COMMIT}/superconductors_3D/data/final/MP/cifs/{name}"
    try:
        _download(url, destination)
        return name, None
    except Exception as exc:
        return name, f"{type(exc).__name__}: {exc}"


def download_training_data(
    *,
    data_root: Path = Path("data/raw"),
    cif_limit: int | None = 128,
    all_cifs: bool = False,
    report_path: Path | None = None,
    workers: int = 8,
) -> dict[str, object]:
    uci_archive = data_root / "uci" / "superconductivity-data.zip"
    uci_extract = data_root / "uci" / "extracted"
    table = data_root / "3dsc" / "MP" / "3DSC_MP.csv"
    cif_root = data_root / "3dsc" / "MP" / "cifs"
    uci_downloaded = _download(UCI_URL, uci_archive, UCI_SHA256)
    if not (uci_extract / "unique_m.csv").is_file():
        _safe_extract_zip(uci_archive, uci_extract)
    table_downloaded = _download(THREEDSC_TABLE_URL, table, THREEDSC_TABLE_SHA256)
    rows = _read_3dsc_rows(table)
    selected = _select_cifs(rows, None if all_cifs else cif_limit)
    failures: dict[str, str] = {}
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_download_cif, name, cif_root / name): name
            for name in selected
        }
        for future in as_completed(futures):
            name, error = future.result()
            if error is None:
                downloaded += 1
            else:
                failures[name] = error
    report: dict[str, object] = {
        "status": "ready" if not failures else "ready_with_cif_failures",
        "uci": {
            "archive": str(uci_archive),
            "sha256": UCI_SHA256,
            "downloaded": uci_downloaded,
            "unique_m": str(uci_extract / "unique_m.csv"),
        },
        "3dsc": {
            "table": str(table),
            "commit": THREEDSC_COMMIT,
            "sha256": THREEDSC_TABLE_SHA256,
            "downloaded": table_downloaded,
            "license": THREEDSC_LICENSE,
            "rows": len(rows),
            "cif_root": str(cif_root),
            "cif_requested": len(selected),
            "cif_downloaded": downloaded,
            "cif_failures": failures,
            "all_cifs": all_cifs,
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--cif-limit", type=int, default=128)
    parser.add_argument("--all-cifs", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("data/processed/download-report.json"))
    args = parser.parse_args()
    report = download_training_data(
        data_root=args.data_root,
        cif_limit=args.cif_limit,
        all_cifs=args.all_cifs,
        report_path=args.report,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
