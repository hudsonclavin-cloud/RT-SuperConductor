#!/usr/bin/env python3
"""Read a finished phonon chain (scripts/hpc/submit_phonon_chain.sh) and print its verdict as JSON.

    python3 scripts/phonon_chain_report.py <run_dir>/run > report.json

Standard library only, so it runs on a compute node. It reads:
- ``<prefix>.dyn*``: the phonon frequencies at every calculated q-point (imaginary modes show as negative). The
  three whole-crystal translations at the zone centre are left out: they are zero by symmetry;
- ``lambda``: matdyn.x's table of lambda and omega_log for each of the ten broadenings;
- ``scf.out``: the residual pressure and the largest force at the relaxed geometry.
- every step's output: which Quantum ESPRESSO program and version wrote it, on how many MPI processes, and whether
  it printed "JOB DONE" (the ``quantum_espresso`` section).

The rung 4 rule (``docs/rung4-plan.md``, step 5): a finalist needs no imaginary mode and an isotropic Tc of at least
2 K. Here Tc is Allen-Dynes at mu* = 0.10 and 0.13 from the headline broadening, and is read as an upper bound.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

IMAGINARY_BELOW_CM1 = -5.0        # small negative acoustic values at Gamma are numerical noise
HEADLINE_BROADENING_RY = 0.020
TC_RULE_K = 2.0


def allen_dynes(lam: float, omega_log_k: float, mu_star: float) -> float:
    denominator = lam - mu_star * (1.0 + 0.62 * lam)
    if lam <= 0 or omega_log_k <= 0 or denominator <= 0:
        return 0.0
    return omega_log_k / 1.2 * math.exp(-1.04 * (1.0 + lam) / denominator)


MODE = re.compile(r"freq\s*\(\s*\d+\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]")
Q_POINT = re.compile(r"q\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)")
VECTOR = re.compile(r"^\s*\(\s*([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s*\)\s*$")


TRANSLATION_OVERLAP = 0.9         # acoustic modes at Gamma measure 0.99 or more, optical modes 0.1 or less


def atom_masses(text: str) -> list[float] | None:
    """The mass of each atom, in the order of the file, from a .dyn file's header. None if there is no header."""
    lines = text.splitlines()
    for start, line in enumerate(lines[:8]):
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit() and parts[1].isdigit():      # ntyp, nat, ibrav, celldm(1..6)
            ntyp, nat = int(parts[0]), int(parts[1])
            break
    else:
        return None
    start += 1
    if start < len(lines) and lines[start].strip().lower().startswith("basis vectors"):
        start += 4
    species: dict[int, float] = {}
    for line in lines[start:start + ntyp]:
        found = re.match(r"\s*(\d+)\s+'[^']*'\s+([-+\d.EeDd]+)", line)
        if not found:
            return None
        species[int(found.group(1))] = float(found.group(2).replace("D", "E").replace("d", "e"))
    masses = []
    for line in lines[start + ntyp:start + ntyp + nat]:
        parts = line.split()
        if len(parts) < 5 or not parts[1].isdigit() or int(parts[1]) not in species:
            return None
        masses.append(species[int(parts[1])])
    return masses if len(masses) == nat else None


def translation_overlap(vectors: list[tuple[float, float, float]], masses: list[float]) -> float:
    """How much of a mode is the whole crystal sliding: 1 for a pure translation, 0 for a mode with no net momentum.

    The .dyn file prints displacements u. In mass-weighted form a translation is sqrt(M) times one direction for
    every atom, so the overlap is |sum M u| / sqrt(sum M |u|^2 * sum M). Optical modes at the zone centre carry no
    net momentum (sum M u = 0), whatever the masses are, so the two kinds are far apart.
    """
    net = [sum(m * v[k] for m, v in zip(masses, vectors)) for k in range(3)]
    weight = sum(m * sum(x * x for x in v) for m, v in zip(masses, vectors)) * sum(masses)
    return math.sqrt(sum(x * x for x in net) / weight) if weight > 0 else 0.0


def is_translation(vectors: list[tuple[float, float, float]], masses: list[float] | None = None) -> bool:
    """True if the mode is the crystal sliding as a whole, which costs no energy.

    With the atoms' masses (read from the file's header) this is the overlap test above. Without them it falls back
    to "every atom moves the same way by nearly the same amount", which is right only for atoms of similar mass.
    """
    if masses is not None and len(masses) == len(vectors) and len(vectors) >= 2:
        return translation_overlap(vectors, masses) >= TRANSLATION_OVERLAP
    if len(vectors) < 2:
        return False
    first = vectors[0]
    size = math.sqrt(sum(x * x for x in first))
    if size == 0:
        return False
    for other in vectors[1:]:
        other_size = math.sqrt(sum(x * x for x in other))
        if other_size == 0 or not 0.9 < other_size / size < 1.1:
            return False
        if sum(a * b for a, b in zip(first, other)) / (size * other_size) < 0.99:
            return False
    return True


def modes(text: str) -> list[dict]:
    """Every mode in one .dyn file: its frequency in cm-1 and whether it is a whole-crystal translation at q = 0.

    The three translations at the zone centre have zero frequency by symmetry. ph.x does not impose that, so the
    raw file shows them a few cm-1 either side of zero (-10.7 for SbTe2 in the pre-check). They are numerical
    noise, not an instability; q2r.x and matdyn.x set them to zero with the acoustic sum rule.
    """
    found: list[dict] = []
    at_gamma = False
    current: dict | None = None
    for line in text.splitlines():
        q = Q_POINT.search(line)
        if q:
            at_gamma = all(abs(float(x)) < 1e-8 for x in q.groups())
            continue
        mode = MODE.search(line)
        if mode:
            current = {"cm1": float(mode.group(1)), "at_gamma": at_gamma, "vectors": []}
            found.append(current)
            continue
        vector = VECTOR.match(line)
        if vector and current is not None:
            current["vectors"].append(tuple(float(x) for x in vector.groups()))
    masses = atom_masses(text)
    for item in found:
        vectors = item.pop("vectors")
        item["translation"] = item["at_gamma"] and is_translation(vectors, masses)
    return found


def frequencies(text: str) -> list[float]:
    """Mode frequencies in cm-1 from one .dyn file, without the zone-centre translations."""
    return [item["cm1"] for item in modes(text) if not item["translation"]]


def lambda_table(text: str) -> list[dict]:
    rows = []
    pattern = re.compile(r"Broadening\s+([\d.]+)\s+lambda\s+([-\d.]+)\s+dos\(Ef\)\s*([-\d.]+)\s+omega_ln \[K\]\s+([-\d.]+)")
    for match in pattern.finditer(text):
        rows.append({"broadening_ry": float(match.group(1)), "lambda": float(match.group(2)),
                     "dos_ef": float(match.group(3)), "omega_log_k": float(match.group(4))})
    return rows


VALIDATED_QE_VERSION = "7.6"      # the version every benchmark in this repository was run with
CHAIN_OUTPUTS = ("scf_fit.out", "scf.out", "ph.out", "q2r.out", "matdyn.out")
PROGRAM = re.compile(r"Program\s+(\S+)\s+v\.(\S+)\s+starts")
MPI_PROCESSES = re.compile(r"Number of MPI processes:\s+(\d+)")


def programs(run: Path) -> dict:
    """Which Quantum ESPRESSO program wrote each output of the chain, which version, on how many MPI processes,
    and whether it finished. Every QE program prints its name and version first and "JOB DONE." last."""
    found = {}
    for name in CHAIN_OUTPUTS:
        path = run / name
        if not path.is_file():
            found[name] = {"present": False}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        program, processes = PROGRAM.search(text), MPI_PROCESSES.search(text)
        found[name] = {"present": True, "program": program.group(1) if program else None,
                       "version": program.group(2) if program else None,
                       "mpi_processes": int(processes.group(1)) if processes else 1,
                       "finished": "JOB DONE" in text}
    present = [entry for entry in found.values() if entry["present"]]
    versions = sorted({entry["version"] for entry in present if entry["version"]})
    return {"outputs": found, "versions": versions,
            "is_quantum_espresso": bool(present) and all(entry["program"] for entry in present),
            "validated_version": VALIDATED_QE_VERSION,
            "matches_validated_version": versions == [VALIDATED_QE_VERSION],
            "all_finished": len(present) == len(CHAIN_OUTPUTS) and all(entry["finished"] for entry in present)}


def report(run: Path) -> dict:
    run = run.resolve()  # the job passes ".", whose parent would otherwise be "." again
    package = json.loads((run.parent / "package.json").read_text(encoding="utf-8")) if (run.parent / "package.json").is_file() else {}
    prefix = package.get("name") or next(run.glob("*.dyn0")).name[:-5]
    mode = (run.parent / "CHAIN_MODE.txt").read_text(encoding="utf-8").splitlines()[0] if (run.parent / "CHAIN_MODE.txt").is_file() else "mode=unknown"
    out: dict = {"name": prefix, "formula": package.get("formula"), "mode": mode.split("=")[-1]}
    lowest = None
    q_files = sorted(p for p in run.glob(f"{prefix}.dyn*") if not p.name.endswith("dyn0"))
    translations: list[float] = []
    for path in q_files:
        found = modes(path.read_text(encoding="utf-8", errors="replace"))
        translations += [item["cm1"] for item in found if item["translation"]]
        values = [item["cm1"] for item in found if not item["translation"]]
        if values:
            lowest = min(values) if lowest is None else min(lowest, min(values))
    out["q_point_files"] = len(q_files)
    out["lowest_frequency_cm1"] = lowest
    out["zone_centre_translations_cm1"] = translations  # should be near zero; not counted as modes
    out["imaginary_mode"] = None if lowest is None else bool(lowest < IMAGINARY_BELOW_CM1)
    table = lambda_table((run / "lambda").read_text(encoding="utf-8")) if (run / "lambda").is_file() else []
    out["lambda_by_broadening"] = table
    if table:
        headline = min(table, key=lambda row: abs(row["broadening_ry"] - HEADLINE_BROADENING_RY))
        lambdas = [row["lambda"] for row in table]
        out["headline"] = {**headline,
                           "tc_allen_dynes_k_mu_0.10": round(allen_dynes(headline["lambda"], headline["omega_log_k"], 0.10), 2),
                           "tc_allen_dynes_k_mu_0.13": round(allen_dynes(headline["lambda"], headline["omega_log_k"], 0.13), 2),
                           "lambda_range_over_broadenings": [min(lambdas), max(lambdas)]}
    scf = (run / "scf.out").read_text(encoding="utf-8", errors="replace") if (run / "scf.out").is_file() else ""
    pressure = re.findall(r"P=\s*([-\d.]+)", scf)
    force = re.findall(r"Total force =\s*([-\d.]+)", scf)
    out["scf_pressure_kbar"] = float(pressure[-1]) if pressure else None
    out["scf_total_force_ry_bohr"] = float(force[-1]) if force else None
    out["quantum_espresso"] = programs(run)
    if out["mode"] != "full":
        out["verdict"] = "smoke run: checks the tools only, no verdict"
    elif out["imaginary_mode"] is None or "headline" not in out:
        out["verdict"] = "incomplete"
    elif out["imaginary_mode"]:
        out["verdict"] = "fails: imaginary phonon mode"
    elif out["headline"]["tc_allen_dynes_k_mu_0.13"] >= TC_RULE_K:
        out["verdict"] = "passes the rung 4 rule (no imaginary mode, Tc at least 2 K at mu* = 0.13); read Tc as an upper bound"
    else:
        out["verdict"] = "stable, but calculated Tc is below 2 K"
    return out


if __name__ == "__main__":
    print(json.dumps(report(Path(sys.argv[1])), indent=1))
