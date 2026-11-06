#!/usr/bin/env python3
"""Prepare a JARVIS JVASP-1014 tantalum QE/EPW benchmark hand-off.

The script writes source-verified templates only. It never runs DFT and leaves
status blocked until reviewed pseudopotentials and a scheduler/HPC allocation
are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from generate_qe_inputs import render_epw, render_matdyn, render_nscf, render_ph, render_q2r, render_relax, render_scf, render_vc_relax  # noqa: E402

JID = "JVASP-1014"
FORMULA = "Ta"
TARGET_LAMBDA = 0.7572
TARGET_OMEGA_LOG_K = 164.99
SOURCE_OMEGA_LOG_K = 164.9922
DATASET_DOI = "10.6084/m9.figshare.21370572.v4"
PAPER_DOI = "10.1038/s41524-022-00933-1"
DATASET_URL = "https://figshare.com/articles/dataset/JARVIS-SuperconDB/21370572"
DATASET_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/38307921"
DATASET_SHA256 = "4213651cee9b4c13a376dd74ac1f8d3bd9efcd451f1b8da615255c8df0e1487f"
DATASET_MD5 = "bb4a91e899458f34cbb3b9bf03e9bd56"
DATASET_MEMBER = "jarvis_epc_data_figshare_1058.json"
DEFAULT_DATASET = Path("data/raw/jarvis-supercond/jarvis_epc_data_figshare_1058.json.zip")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tantalum_row(dataset_path: Path) -> tuple[dict[str, Any], str]:
    """Load JVASP-1014 only after matching the pinned published dataset file."""
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, dataset_path)
    payload = dataset_path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()
    if sha != DATASET_SHA256 or md5 != DATASET_MD5:
        raise ValueError(f"JARVIS-SuperconDB checksum mismatch (sha256={sha}, md5={md5})")
    with zipfile.ZipFile(dataset_path) as archive:
        rows = json.load(archive.open(DATASET_MEMBER))
    row = next((item for item in rows if str(item.get("jid")) == JID), None)
    if row is None:
        raise ValueError(f"{JID} is absent from the pinned JARVIS-SuperconDB export")
    return row, sha


def _structure_from_row(row: dict[str, Any]):
    from pymatgen.core import Structure

    atoms = row.get("atoms") or {}
    structure = Structure(
        np.asarray(atoms["lattice_mat"], dtype=float),
        atoms["elements"],
        atoms["coords"],
        coords_are_cartesian=bool(atoms.get("cartesian", False)),
    )
    if structure.composition.reduced_formula != FORMULA:
        raise ValueError("JVASP-1014 structure is not elemental Ta")
    return structure


def _set_pseudo_path(text: str, pseudo_dir: Path, output_dir: Path) -> str:
    import os
    relative = os.path.relpath(pseudo_dir.resolve(), output_dir.resolve())
    return text.replace("pseudo_dir = './pseudo'", f"pseudo_dir = '{relative}'")


def prepare_tantalum_epw_benchmark(
    *, output_dir: Path = Path("data/processed/epw-benchmark/tantalum-jvasp-1014"),
    dataset_path: Path = DEFAULT_DATASET,
    pseudopotential_dir: Path = Path("data/pseudopotentials/upf"),
    scheduler_confirmed: bool = False,
    source_row: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Write inputs and a provenance manifest from the exact JARVIS entry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if source_row is None:
            source_row, source_sha256 = load_tantalum_row(dataset_path)
        if str(source_row.get("jid")) != JID:
            raise ValueError(f"Expected {JID} source row")
        lam = float(source_row["lamb"])
        omega = float(source_row["wlog"])
        if not np.isclose(lam, TARGET_LAMBDA, atol=1e-8) or not np.isclose(omega, SOURCE_OMEGA_LOG_K, atol=1e-4):
            raise ValueError("JARVIS electron-phonon targets do not match the pinned JVASP-1014 record")
        structure = _structure_from_row(source_row)
    except Exception as exc:
        report = {"status": "blocked_source_unavailable_or_invalid", "jid": JID,
                  "reason": f"{type(exc).__name__}: {exc}", "dft_executed": False,
                  "input_files_written": []}
        (output_dir / "generation-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    expected = {"Ta": "Ta.UPF"}
    pseudo = pseudopotential_dir / expected["Ta"]
    pseudo_status = {"Ta": {"filename": expected["Ta"], "path": str(pseudo),
                             "present": pseudo.is_file(), "sha256": _sha256(pseudo) if pseudo.is_file() else None}}
    cif = output_dir / "ta-jvasp-1014-reference.cif"
    cif.write_text(structure.to(fmt="cif"), encoding="utf-8")
    inputs = {
        "relax.in": render_relax("ta_jvasp_1014", FORMULA, structure=structure, pseudopotential_catalog=expected),
        "vc-relax.in": render_vc_relax("ta_jvasp_1014", FORMULA, structure=structure, pseudopotential_catalog=expected),
        "scf.in": render_scf("ta_jvasp_1014", FORMULA, structure=structure, pseudopotential_catalog=expected),
        "nscf.in": render_nscf("ta_jvasp_1014", FORMULA, structure=structure, pseudopotential_catalog=expected),
        "ph.in": render_ph("ta_jvasp_1014"),
        "q2r.in": render_q2r("ta_jvasp_1014"),
        "matdyn.in": render_matdyn("ta_jvasp_1014"),
        "epw.in": render_epw("ta_jvasp_1014"),
    }
    for name, text in inputs.items():
        text = "! Template only: inspect convergence, pseudopotentials, and workflow before use.\n" + text
        if name in {"relax.in", "vc-relax.in", "scf.in", "nscf.in"}:
            text = _set_pseudo_path(text, pseudopotential_dir, output_dir)
        (output_dir / name).write_text(text, encoding="utf-8")
    blocked = []
    if not pseudo.is_file():
        blocked.append("reviewed Ta UPF pseudopotential file is missing")
    if not scheduler_confirmed:
        blocked.append("HPC scheduler/access has not been confirmed")
    manifest = {
        "benchmark": "JARVIS tantalum electron-phonon target reproduction",
        "jid": JID, "formula": FORMULA,
        "structure_source": "JARVIS-SuperconDB published DFT structure; not an experimental structure claim",
        "dataset_doi": DATASET_DOI, "dataset_landing_page": DATASET_URL,
        "dataset_download_url": DATASET_DOWNLOAD_URL,
        "dataset_license": "CC BY 4.0", "dataset_sha256": source_sha256,
        "paper_doi": PAPER_DOI,
        "target_lambda": TARGET_LAMBDA,
        "target_omega_log_k": TARGET_OMEGA_LOG_K,
        "source_omega_log_k_unrounded": SOURCE_OMEGA_LOG_K,
        "source_tc_k": float(source_row.get("Tc")),
        "source_pressure": str(source_row.get("press", "")),
        "pseudopotentials": pseudo_status,
        "scheduler_confirmed_by_operator": bool(scheduler_confirmed),
        "status": "blocked_prerequisites" if blocked else "prepared_for_manual_review",
        "blocked_reasons": blocked, "dft_executed": False,
        "structure_file": str(cif), "structure_sha256": _sha256(cif),
        "input_files": list(inputs),
        "input_sha256": {name: _sha256(output_dir / name) for name in inputs},
    }
    (output_dir / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/epw-benchmark/tantalum-jvasp-1014"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pseudopotential-dir", type=Path, default=Path("data/pseudopotentials/upf"))
    parser.add_argument("--scheduler-confirmed", action="store_true", help="operator confirms scheduler and compute access")
    args = parser.parse_args()
    report = prepare_tantalum_epw_benchmark(output_dir=args.output_dir, dataset_path=args.dataset,
                                             pseudopotential_dir=args.pseudopotential_dir,
                                             scheduler_confirmed=args.scheduler_confirmed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "blocked_source_unavailable_or_invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
