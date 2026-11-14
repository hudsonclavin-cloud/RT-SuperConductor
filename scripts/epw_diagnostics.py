#!/usr/bin/env python3
"""Three quick checks on a finished EPW run, none of which needs a new calculation.

``moments``   Read the Eliashberg function (the ``<prefix>.a2f.*`` file) and report lambda, the frequency
              averages, and eta = M <omega^2> lambda. Eta depends on the electrons and how strongly they
              couple, not on how stiff the lattice is. Comparing eta with a published value tells you
              whether a wrong lambda comes from the phonon frequencies or from the electronic side.

``symmetry``  Points in the Brillouin zone that the crystal's symmetry maps onto each other must give the
              same phonon frequencies and the same coupling. This groups the q-points of an EPW output
              into such sets ("stars") and reports how far the values inside each set disagree. A large
              disagreement, or any negative coupling, means the number is not to be trusted.

``stars``     Write a q-point file (for EPW's ``filqf``) holding every member of the stars of a few chosen
              points, so the symmetry check can be run in a minute or two instead of over a whole grid.

The structure is read from the Quantum ESPRESSO XML file (``<prefix>.xml`` or ``data-file-schema.xml``).
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MEV_PER_KELVIN = 0.0861733326          # Boltzmann constant in meV/K
RY_TO_KELVIN = 157887.5
AMU_KG, HBAR_JS, EV_J = 1.66053907e-27, 1.054571817e-34, 1.602176634e-19
IQ_LINE = re.compile(r"iq =\s+(\d+)\s+coord\.:\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+wt:\s+([\d.]+)")
MODE_LINE = re.compile(r"lambda___\(\s*(\d+)\s*\)=\s+(-?[\d.]+)\s+gamma___=\s+(-?[\d.]+) meV\s+omega=\s+(-?[\d.]+) meV")
TOTAL_LINE = re.compile(r"lambda___\(\s*tot\s*\)=\s+(-?[\d.]+)")


# ---------------------------------------------------------------------------
# moments

def read_a2f(path: Path, column: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(frequency in meV, a2F) from an EPW a2f file; ``column`` picks the phonon smearing (1 = the smallest)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith("Integrated"):
            break
        parts = line.split()
        if len(parts) > column:
            try:
                rows.append((float(parts[0]), float(parts[column])))
            except ValueError:
                continue
    if len(rows) < 10:
        raise ValueError(f"{path} does not look like an EPW a2f file")
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def moments(omega_mev: np.ndarray, a2f: np.ndarray, mass_amu: float | None = None,
            lowest_mev: float = 0.2) -> dict:
    """lambda, omega_log, the first and second frequency averages, and eta when a mass is given.

    All averages are weighted by 2 a2F / omega, the weight that defines lambda. Frequencies below
    ``lowest_mev`` are left out: a2F is divided by omega, so noise near zero would dominate.
    """
    keep = omega_mev > lowest_mev
    w, f = omega_mev[keep], a2f[keep]
    step = np.gradient(w)
    weight = 2.0 * f / w * step
    lam = float(weight.sum())
    if lam <= 0:
        raise ValueError("the Eliashberg function integrates to zero or less")
    log_average = float(np.exp((weight * np.log(w)).sum() / lam))
    first = float((weight * w).sum() / lam)
    second = float(np.sqrt((weight * w * w).sum() / lam))
    cumulative = np.cumsum(weight)
    out = {"lambda": lam, "omega_log_k": log_average / MEV_PER_KELVIN, "omega_1_k": first / MEV_PER_KELVIN,
           "omega_2_k": second / MEV_PER_KELVIN, "omega_log_mev": log_average,
           "half_of_lambda_below_mev": float(w[int(np.searchsorted(cumulative, lam / 2.0))]),
           "highest_frequency_mev": float(w[f > 1e-4 * f.max()].max())}
    if mass_amu:
        angular = second * 1e-3 * EV_J / HBAR_JS                       # rad/s
        out["eta_ev_per_a2"] = float(mass_amu * AMU_KG * angular ** 2 * lam / EV_J * 1e-20)
    return out


def eta_from(lam: float, omega_2_k: float, mass_amu: float) -> float:
    """eta = M <omega^2> lambda in eV per square angstrom, for comparing with a published lambda and frequency."""
    angular = omega_2_k * MEV_PER_KELVIN * 1e-3 * EV_J / HBAR_JS
    return float(mass_amu * AMU_KG * angular ** 2 * lam / EV_J * 1e-20)


# ---------------------------------------------------------------------------
# symmetry

def read_structure(xml_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(lattice vectors as rows, fractional positions, species) from a Quantum ESPRESSO XML file."""
    structure = ET.parse(xml_path).getroot().find("output/atomic_structure")
    cell = structure.find("cell")
    lattice = np.array([[float(x) for x in cell.find(tag).text.split()] for tag in ("a1", "a2", "a3")])
    atoms = structure.find("atomic_positions").findall("atom")
    cartesian = np.array([[float(x) for x in atom.text.split()] for atom in atoms])
    return lattice, cartesian @ np.linalg.inv(lattice), [atom.get("name") for atom in atoms]


def reciprocal_operations(lattice: np.ndarray, positions: np.ndarray, species: list[str]) -> list[np.ndarray]:
    """How each symmetry operation of the crystal acts on a q-point given in crystal coordinates."""
    import spglib

    numbers = [sorted(set(species)).index(name) + 1 for name in species]
    found = spglib.get_symmetry((lattice, positions, numbers), symprec=1e-5)
    unique = {tuple(np.round(np.linalg.inv(rotation).T).astype(int).ravel()) for rotation in found["rotations"]}
    return [np.array(op).reshape(3, 3) for op in sorted(unique)]


def grid_key(q: np.ndarray, grid: tuple[int, int, int]) -> tuple[int, int, int]:
    """A q-point as whole-number grid coordinates, folded into the first cell."""
    scaled = np.asarray(q, float) * np.asarray(grid)
    whole = np.round(scaled)
    if np.abs(scaled - whole).max() > 2e-3:
        raise ValueError(f"q-point {tuple(q)} is not on the {grid} grid")
    return tuple(int(v) % n for v, n in zip(whole, grid))


def star_of(q: np.ndarray, operations: list[np.ndarray], grid: tuple[int, int, int],
            time_reversal: bool = True) -> list[tuple[int, int, int]]:
    """Every grid point that symmetry (and q -> -q) maps ``q`` onto, sorted."""
    signs = (1, -1) if time_reversal else (1,)
    return sorted({grid_key(sign * (op @ np.asarray(q, float)), grid) for op in operations for sign in signs})


def parse_epw_output(path: Path) -> list[dict]:
    """Per q-point: coordinates, weight, the frequency and coupling of every mode, and the total coupling."""
    points, current = [], None
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = IQ_LINE.search(line)
        if match:
            current = {"q": [float(match.group(i)) for i in (2, 3, 4)], "weight": float(match.group(5)),
                       "omega_mev": [], "lambda_modes": [], "lambda": None}
            points.append(current)
            continue
        if current is None:
            continue
        match = MODE_LINE.search(line)
        if match:
            current["lambda_modes"].append(float(match.group(2)))
            current["omega_mev"].append(float(match.group(4)))
            continue
        match = TOTAL_LINE.search(line)
        if match:
            current["lambda"] = float(match.group(1))
    return [point for point in points if point["lambda"] is not None]


def fine_grid(epw_text: str) -> tuple[int, int, int] | None:
    match = re.search(r"Using uniform q-mesh:\s+(\d+)\s+(\d+)\s+(\d+)", epw_text)
    return tuple(int(match.group(i)) for i in (1, 2, 3)) if match else None


def symmetry_report(points: list[dict], operations: list[np.ndarray], grid: tuple[int, int, int],
                    max_spread: float = 0.05) -> dict:
    """Group the q-points into stars and measure the disagreement inside each."""
    by_key = {grid_key(point["q"], grid): point for point in points}
    seen, stars = set(), []
    for key, point in by_key.items():
        if key in seen:
            continue
        members = [k for k in star_of(np.array(key) / np.array(grid), operations, grid) if k in by_key]
        seen.update(members)
        if len(members) < 2:
            continue
        lam = np.array([by_key[k]["lambda"] for k in members])
        omega = np.array([sorted(by_key[k]["omega_mev"]) for k in members])
        mean = float(lam.mean())
        stars.append({"q": point["q"], "members_present": len(members), "lambda_mean": mean,
                      "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                      "lambda_spread_over_mean": float((lam.max() - lam.min()) / abs(mean)) if mean else None,
                      "omega_spread_mev": float((omega.max(axis=0) - omega.min(axis=0)).max()) if omega.size else 0.0})
    all_lambda = np.array([point["lambda"] for point in points])
    spreads = [star["lambda_spread_over_mean"] for star in stars if star["lambda_spread_over_mean"] is not None]
    report = {"q_points": len(points), "stars_with_two_or_more_members": len(stars),
              "negative_coupling_points": int((all_lambda < 0).sum()),
              "lowest_coupling": float(all_lambda.min()) if len(all_lambda) else None,
              "median_spread_over_mean": float(np.median(spreads)) if spreads else None,
              "largest_spread_over_mean": float(max(spreads)) if spreads else None,
              "largest_frequency_spread_mev": max((star["omega_spread_mev"] for star in stars), default=0.0),
              "worst_stars": sorted(stars, key=lambda s: -(s["lambda_spread_over_mean"] or 0))[:5]}
    problems = []
    if not stars:
        problems.append("no star has two members in this output: use a full grid or a file written by 'stars'")
    if report["negative_coupling_points"]:
        problems.append("negative coupling at some q-points: this cannot be physical; check that the input has "
                        "delta_approx = .true.")
    if report["largest_frequency_spread_mev"] > 1e-2:
        problems.append("phonon frequencies differ inside a star: the phonon interpolation is not symmetric")
    if spreads and report["median_spread_over_mean"] > max_spread:
        problems.append(f"coupling differs inside stars by {report['median_spread_over_mean']:.0%} (median), above the "
                        f"{max_spread:.0%} limit: the interpolated electrons or matrix elements are not symmetric")
    report["problems"] = problems
    report["verdict"] = "pass" if not problems else "fail"
    return report


def write_star_file(path: Path, seeds: list[list[float]], operations: list[np.ndarray],
                    grid: tuple[int, int, int]) -> list[int]:
    """Write every member of the star of each seed point in EPW's filqf format; returns the star sizes."""
    members, sizes = [], []
    for seed in seeds:
        star = star_of(np.array(seed, float), operations, grid)
        members += star
        sizes.append(len(star))
    lines = [f"{len(members)} crystal"]
    lines += [f"{k[0] / grid[0]:.10f} {k[1] / grid[1]:.10f} {k[2] / grid[2]:.10f} {1.0 / len(members):.10f}"
              for k in members]
    Path(path).write_text("\n".join(lines) + "\n")
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("moments", help="lambda, frequency averages and eta from an a2f file")
    m.add_argument("--a2f", type=Path, required=True)
    m.add_argument("--mass-amu", type=float, help="atomic mass, for eta (one kind of atom only)")
    m.add_argument("--compare", nargs=2, type=float, metavar=("LAMBDA", "OMEGA_LOG_K"),
                   help="a published lambda and omega_log; its eta is estimated with this run's ratio of "
                        "second-moment to log-average frequency")
    s = sub.add_parser("symmetry", help="do symmetry-equivalent q-points agree?")
    s.add_argument("--epw-out", type=Path, required=True)
    s.add_argument("--xml", type=Path, required=True, help="the Quantum ESPRESSO XML file of the same structure")
    s.add_argument("--grid", type=int, nargs=3, help="fine q grid, when the output does not state it (filqf runs)")
    s.add_argument("--max-spread", type=float, default=0.05)
    s.add_argument("--report", type=Path)
    t = sub.add_parser("stars", help="write a filqf file with every member of the stars of the given points")
    t.add_argument("--xml", type=Path, required=True)
    t.add_argument("--grid", type=int, nargs=3, required=True)
    t.add_argument("--point", type=float, nargs=3, action="append", required=True, metavar=("Q1", "Q2", "Q3"))
    t.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "moments":
        result = moments(*read_a2f(args.a2f), mass_amu=args.mass_amu)
        if args.compare and args.mass_amu:
            ratio = result["omega_2_k"] / result["omega_log_k"]
            result["published"] = {"lambda": args.compare[0], "omega_log_k": args.compare[1],
                                   "eta_ev_per_a2_estimate": eta_from(args.compare[0], ratio * args.compare[1],
                                                                      args.mass_amu),
                                   "assumed_ratio_omega_2_to_omega_log": ratio}
        print(json.dumps(result, indent=1))
        return 0
    operations = reciprocal_operations(*read_structure(args.xml))
    if args.command == "stars":
        sizes = write_star_file(args.out, args.point, operations, tuple(args.grid))
        print(f"wrote {args.out}: {sum(sizes)} q-points in stars of {sizes}")
        return 0
    grid = tuple(args.grid) if args.grid else fine_grid(args.epw_out.read_text(errors="ignore"))
    if grid is None:
        raise SystemExit("the output does not state a uniform fine q grid; pass --grid")
    report = symmetry_report(parse_epw_output(args.epw_out), operations, grid, args.max_spread)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "worst_stars"}, indent=1))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
