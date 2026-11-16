#!/usr/bin/env python3
"""Read a finished phonon chain (scripts/hpc/submit_phonon_chain.sh) and print its verdict as JSON.

    python3 scripts/phonon_chain_report.py <run_dir>/run > report.json

Standard library only, so it runs on a compute node. It reads:
- ``<prefix>.dyn*``: the phonon frequencies at every calculated q-point (imaginary modes show as negative);
- ``lambda``: matdyn.x's table of lambda and omega_log for each of the ten broadenings;
- ``scf.out``: the residual pressure and the largest force at the relaxed geometry.

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


def frequencies(text: str) -> list[float]:
    """Mode frequencies in cm-1 from one .dyn file (every q-point in it)."""
    return [float(x) for x in re.findall(r"freq\s*\(\s*\d+\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]", text)]


def lambda_table(text: str) -> list[dict]:
    rows = []
    pattern = re.compile(r"Broadening\s+([\d.]+)\s+lambda\s+([-\d.]+)\s+dos\(Ef\)\s*([-\d.]+)\s+omega_ln \[K\]\s+([-\d.]+)")
    for match in pattern.finditer(text):
        rows.append({"broadening_ry": float(match.group(1)), "lambda": float(match.group(2)),
                     "dos_ef": float(match.group(3)), "omega_log_k": float(match.group(4))})
    return rows


def report(run: Path) -> dict:
    package = json.loads((run.parent / "package.json").read_text(encoding="utf-8")) if (run.parent / "package.json").is_file() else {}
    prefix = package.get("name") or next(run.glob("*.dyn0")).name[:-5]
    mode = (run.parent / "CHAIN_MODE.txt").read_text(encoding="utf-8").splitlines()[0] if (run.parent / "CHAIN_MODE.txt").is_file() else "mode=unknown"
    out: dict = {"name": prefix, "formula": package.get("formula"), "mode": mode.split("=")[-1]}
    lowest = None
    q_files = sorted(p for p in run.glob(f"{prefix}.dyn*") if not p.name.endswith("dyn0"))
    for path in q_files:
        values = frequencies(path.read_text(encoding="utf-8", errors="replace"))
        if values:
            lowest = min(values) if lowest is None else min(lowest, min(values))
    out["q_point_files"] = len(q_files)
    out["lowest_frequency_cm1"] = lowest
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
