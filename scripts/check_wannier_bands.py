#!/usr/bin/env python3
"""Check EPW's Wannier interpolation against directly calculated bands.

EPW computes the electron-phonon coupling on a coarse k-grid and interpolates
it everywhere else through Wannier functions. If the interpolated bands are
wrong near the Fermi level, the wrong states count as being on the Fermi
surface and lambda comes out wrong, even though every program finishes
normally. On the tantalum benchmark, six Wannier functions missed the true
bands by 137 meV on average near E_F and lambda came out 91% too high; nine
functions missed by 36 meV and lambda dropped by a fifth. Run this on every
new material before trusting its lambda. It needs no phonons.

Three steps, from a finished EPW run folder (the one holding scf.in, nscf.in,
epw.in, <prefix>.save, <prefix>_hr.dat and <prefix>_wsvec.dat):

1. prepare: write the direct calculation into a new folder.

     python scripts/check_wannier_bands.py prepare --epw-dir <run>/epw --out <check_dir>

   It copies only the charge density and pseudopotentials, so it is small.
   It picks random k-points (off the coarse grid) plus a few coarse-grid
   points, which must come out exact and so check the bookkeeping.

2. run the two short Quantum ESPRESSO calculations it wrote (minutes):

     sbatch --ntasks=4 --time=00:30:00 --partition=compute --qos=normal <check_dir>/run.sh   # Hazel
     LAUNCHER="mpirun -np 4" bash <check_dir>/run.sh                                         # elsewhere

   Not on a Hazel login node: these are calculations.

3. analyze: compare, print a verdict and write a JSON report.

     python scripts/check_wannier_bands.py analyze --epw-dir <run>/epw --check-dir <check_dir>

   Exit status 0 means pass, 1 means fail, 2 means something is missing.

The verdict looks at every directly calculated state within the window around
E_F that EPW itself uses (fsthick in epw.in), and passes when the mean error
is at most 50 meV, 95% of errors are at most EPW's smearing (degaussw), and
the density of states at E_F agrees within 5%. The thresholds are options.
They come from one material (tantalum) and are a starting point, not a
standard.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

HARTREE_EV = 27.211386245988
DEFAULT_RANDOM_POINTS = 300
DEFAULT_GRID_POINTS = 12
DEFAULT_DENSE_GRID = 24
DEFAULT_MAX_MEAN_MEV = 50.0
DEFAULT_MAX_DOS_DEVIATION = 0.05
GRID_EXACTNESS_MEV = 5.0          # coarse-grid points must agree to within this
MISSING_STATE_EV = 0.1            # a direct state this far from every interpolated one is "missing"


# ---------------------------------------------------------------------------
# Reading the run's inputs

def read_namelist_value(text: str, key: str) -> str | None:
    """A value from a Fortran namelist, e.g. read_namelist_value(epw, "degaussw")."""
    pattern = rf"^\s*{re.escape(key)}\s*=\s*([^,!\n]+)"
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def read_float(text: str, key: str, default: float | None = None) -> float | None:
    value = read_namelist_value(text, key)
    if value is None:
        return default
    return float(value.lower().replace("d", "e"))


def coarse_grid(epw_text: str) -> tuple[int, int, int]:
    grid = tuple(int(read_float(epw_text, f"nk{i}") or 0) for i in (1, 2, 3))
    if min(grid) < 1:
        raise ValueError("epw.in does not give the coarse k-grid (nk1, nk2, nk3)")
    return grid


def species_files(text: str) -> list[str]:
    """Pseudopotential file names listed under ATOMIC_SPECIES."""
    block = text.split("ATOMIC_SPECIES", 1)[1].splitlines()[1:]
    names = []
    for line in block:
        parts = line.split()
        if len(parts) < 3:
            break
        names.append(parts[2])
    return names


def run_settings(epw_dir: Path) -> dict[str, Any]:
    """What the analysis needs from the run folder's inputs."""
    epw = (epw_dir / "epw.in").read_text()
    prefix = read_namelist_value(epw, "prefix")
    if not prefix:
        raise ValueError("epw.in has no prefix")
    return {
        "prefix": prefix,
        "coarse_grid": coarse_grid(epw),
        "fsthick_ev": read_float(epw, "fsthick", 0.4),
        "degaussw_ev": read_float(epw, "degaussw", 0.05),
        "frozen_window_ev": (read_float(epw, "dis_froz_min"), read_float(epw, "dis_froz_max")),
        "num_wannier": int(read_float(epw, "nbndsub", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Reading Quantum ESPRESSO and Wannier90 output

def read_qe_xml(path: Path) -> dict[str, Any]:
    """Cell (in units of alat), k-points (cartesian, 2pi/alat), weights, eigenvalues (eV), E_F (eV)."""
    root = ET.parse(path).getroot()
    structure = root.find("output/atomic_structure")
    alat = float(structure.get("alat"))
    cell = structure.find("cell")
    lattice = np.array([[float(x) for x in cell.find(tag).text.split()] for tag in ("a1", "a2", "a3")]) / alat
    bands = root.find("output/band_structure")
    states = bands.findall("ks_energies")
    kpoints = np.array([[float(x) for x in s.find("k_point").text.split()] for s in states])
    weights = np.array([float(s.find("k_point").get("weight")) for s in states])
    energies = np.array([[float(x) for x in s.find("eigenvalues").text.split()] for s in states]) * HARTREE_EV
    fermi = bands.find("fermi_energy")
    return {"lattice": lattice, "kpoints": kpoints, "weights": weights, "energies": energies,
            "fermi_ev": float(fermi.text) * HARTREE_EV if fermi is not None else None}


def read_hr(path: Path) -> dict[str, Any]:
    """Wannier90's <prefix>_hr.dat: H_mn(R) in eV and the degeneracy of each R."""
    lines = path.read_text().splitlines()
    num_wann, num_r = int(lines[1]), int(lines[2])
    degeneracy: list[int] = []
    row = 3
    while len(degeneracy) < num_r:
        try:
            degeneracy.extend(int(x) for x in lines[row].split())
        except (ValueError, IndexError):
            raise ValueError(f"{path}: header says {num_r} lattice vectors, but the degeneracy list "
                             f"has only {len(degeneracy)} entries") from None
        row += 1
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for line in lines[row:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        r = (int(parts[0]), int(parts[1]), int(parts[2]))
        m, n = int(parts[3]) - 1, int(parts[4]) - 1
        blocks.setdefault(r, np.zeros((num_wann, num_wann), complex))[m, n] = float(parts[5]) + 1j * float(parts[6])
    vectors = list(blocks)
    if len(vectors) != num_r:
        raise ValueError(f"{path}: header says {num_r} lattice vectors, found {len(vectors)}")
    return {"num_wann": num_wann, "vectors": vectors, "degeneracy": degeneracy[:num_r], "blocks": blocks}


def read_wsvec(path: Path | None) -> dict[tuple[tuple[int, int, int], int, int], list[tuple[int, int, int]]]:
    """Wannier90's <prefix>_wsvec.dat (use_ws_distance): shifts T per (R, m, n). Empty if absent."""
    if path is None or not path.is_file():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    shifts, row = {}, 0
    while row < len(lines):
        head = [int(x) for x in lines[row].split()]
        count = int(lines[row + 1])
        vectors = [tuple(int(x) for x in lines[row + 2 + j].split()) for j in range(count)]
        shifts[((head[0], head[1], head[2]), head[3], head[4])] = vectors
        row += 2 + count
    return shifts


class WannierModel:
    """Fourier interpolation of a Wannier Hamiltonian, as Wannier90 and EPW do it.

    H_mn(k) = sum_R sum_T exp(2 pi i k.(R+T)) H_mn(R) / (ndegen(R) * N_T(R, m, n)),
    with k and R in fractional (lattice) coordinates. A line "R m n" in both
    _hr.dat and _wsvec.dat refers to element (m, n).
    """

    def __init__(self, hr: dict[str, Any], wsvec: dict | None = None):
        wsvec = wsvec or {}
        self.num_wann = hr["num_wann"]
        vectors, rows, cols, values = [], [], [], []
        for r, degeneracy in zip(hr["vectors"], hr["degeneracy"]):
            block = hr["blocks"][r]
            for m in range(self.num_wann):
                for n in range(self.num_wann):
                    shifts = wsvec.get((r, m + 1, n + 1), [(0, 0, 0)])
                    for t in shifts:
                        vectors.append((r[0] + t[0], r[1] + t[1], r[2] + t[2]))
                        rows.append(m)
                        cols.append(n)
                        values.append(block[m, n] / (degeneracy * len(shifts)))
        self._vectors = np.array(vectors, float)
        self._rows = np.array(rows)
        self._cols = np.array(cols)
        self._values = np.array(values)

    def hamiltonian(self, kfrac: np.ndarray) -> np.ndarray:
        terms = np.exp(2j * np.pi * self._vectors @ np.asarray(kfrac, float)) * self._values
        h = np.zeros((self.num_wann, self.num_wann), complex)
        np.add.at(h, (self._rows, self._cols), terms)
        return 0.5 * (h + h.conj().T)

    def bands(self, kfrac: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.eigvalsh(self.hamiltonian(k)) for k in np.atleast_2d(kfrac)])


def cartesian_to_fractional(kpoints: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """k in cartesian 2pi/alat units -> fractional reciprocal coordinates (k . a_i, a_i in alat)."""
    return np.asarray(kpoints) @ np.asarray(lattice).T


# ---------------------------------------------------------------------------
# Comparison

def nearest_state_errors(direct: np.ndarray, interpolated: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    """For every direct state within centre +/- half_width, the distance to the closest interpolated state."""
    errors = []
    for direct_k, interp_k in zip(direct, interpolated):
        for energy in direct_k:
            if abs(energy - centre) <= half_width:
                errors.append(float(np.min(np.abs(interp_k - energy))))
    return np.array(errors)


def gaussian_dos(energies: np.ndarray, weights: np.ndarray, at: float, sigma: float) -> float:
    """Density of states per spin at energy `at` (states/eV/cell), weights summing to 1."""
    gauss = np.exp(-((energies - at) / sigma) ** 2) / (sigma * np.sqrt(np.pi))
    return float(np.sum(weights[:, None] * gauss))


def uniform_grid(n: int) -> np.ndarray:
    steps = np.arange(n) / n
    return np.array(np.meshgrid(steps, steps, steps, indexing="ij")).reshape(3, -1).T


def summarize(errors: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {"states": 0}
    return {
        "states": int(errors.size),
        "mean_mev": round(float(errors.mean()) * 1000, 1),
        "p95_mev": round(float(np.percentile(errors, 95)) * 1000, 1),
        "max_mev": round(float(errors.max()) * 1000, 1),
        "missing_states": int((errors > MISSING_STATE_EV).sum()),
    }


def analyze(epw_dir: Path, check_dir: Path, *, max_mean_mev: float = DEFAULT_MAX_MEAN_MEV,
            max_p95_mev: float | None = None, max_dos_deviation: float = DEFAULT_MAX_DOS_DEVIATION,
            dense_grid: int | None = None) -> dict[str, Any]:
    settings = run_settings(epw_dir)
    prefix = settings["prefix"]
    meta = json.loads((check_dir / "check.json").read_text()) if (check_dir / "check.json").is_file() else {}
    direct = read_qe_xml(check_dir / "bands.xml")
    dense = read_qe_xml(check_dir / "dense" / f"{prefix}.save" / "data-file-schema.xml")
    if dense["fermi_ev"] is None:
        raise ValueError("the dense calculation has no Fermi energy; it must be an nscf run with smearing")
    model = WannierModel(read_hr(epw_dir / f"{prefix}_hr.dat"), read_wsvec(epw_dir / f"{prefix}_wsvec.dat"))
    fermi = dense["fermi_ev"]
    window = settings["fsthick_ev"]
    smearing = settings["degaussw_ev"]
    max_p95 = max_p95_mev if max_p95_mev is not None else smearing * 1000

    kfrac = cartesian_to_fractional(direct["kpoints"], direct["lattice"])
    interpolated = model.bands(kfrac)
    on_grid = np.array(meta.get("on_grid", [False] * len(kfrac)), bool)
    if on_grid.size != len(kfrac):
        on_grid = np.zeros(len(kfrac), bool)
    off = ~on_grid

    near = nearest_state_errors(direct["energies"][off], interpolated[off], fermi, window)
    wide = nearest_state_errors(direct["energies"][off], interpolated[off], fermi, 1.0)
    low, high = settings["frozen_window_ev"]
    grid_check = None
    if on_grid.any() and low is not None and high is not None:
        centre, half = 0.5 * (low + high), 0.5 * (high - low) - 0.05
        grid_check = summarize(nearest_state_errors(direct["energies"][on_grid], interpolated[on_grid], centre, half))

    n = dense_grid or int(meta.get("dense_grid", DEFAULT_DENSE_GRID))
    grid = uniform_grid(n)
    interp_dense = model.bands(grid)
    dense_weights = dense["weights"] / dense["weights"].sum()
    dos = {}
    for sigma in sorted({smearing, smearing / 2}, reverse=True):
        reference = gaussian_dos(dense["energies"], dense_weights, fermi, sigma)
        wannier = gaussian_dos(interp_dense, np.full(len(grid), 1 / len(grid)), fermi, sigma)
        dos[f"{sigma:g}"] = {"direct": round(reference, 4), "wannier": round(wannier, 4),
                             "ratio": round(wannier / reference, 4) if reference > 0 else None}

    near_summary = summarize(near)
    checks = {
        "mean_error_within_fsthick": near_summary.get("mean_mev", float("inf")) <= max_mean_mev,
        "p95_error_within_fsthick": near_summary.get("p95_mev", float("inf")) <= max_p95,
        "dos_at_fermi_level": all(v["ratio"] is not None and abs(v["ratio"] - 1) <= max_dos_deviation
                                  for v in dos.values()),
    }
    problems = []
    if near_summary["states"] == 0:
        problems.append("no directly calculated state fell within the window around E_F")
    if grid_check and grid_check.get("max_mev", 0) > GRID_EXACTNESS_MEV:
        problems.append(f"coarse-grid points disagree by up to {grid_check['max_mev']} meV; they should be exact. "
                        "The _hr.dat probably does not belong to this calculation.")
    if low is not None and high is not None and not (low <= fermi - window and fermi + window <= high):
        problems.append("the window around E_F extends outside the frozen window, where the Wannier bands "
                        "are not meant to match exactly")
    passed = all(checks.values()) and not problems
    return {
        "verdict": "pass" if passed else "fail",
        "checks": checks,
        "problems": problems,
        "thresholds": {"max_mean_mev": max_mean_mev, "max_p95_mev": round(max_p95, 1),
                       "max_dos_deviation": max_dos_deviation},
        "fermi_energy_ev": round(fermi, 4),
        "window_ev": window, "degaussw_ev": smearing,
        "frozen_window_ev": [low, high],
        "num_wannier": model.num_wann,
        "coarse_grid": list(settings["coarse_grid"]),
        "test_kpoints": int(off.sum()),
        "errors_within_fsthick": near_summary,
        "errors_within_1ev": summarize(wide),
        "coarse_grid_exactness": grid_check,
        "dos_at_fermi_level_per_spin": dos,
        "dense_grid": n,
    }


# ---------------------------------------------------------------------------
# Preparing the direct calculation

def _replace_pseudo_dir(text: str) -> str:
    if re.search(r"pseudo_dir\s*=", text):
        return re.sub(r"pseudo_dir\s*=\s*'[^']*'", "pseudo_dir  = './pp/'", text)
    return text.replace("&control", "&control\n    pseudo_dir  = './pp/'", 1)


def _set_calculation(text: str, kind: str) -> str:
    return re.sub(r"calculation\s*=\s*'[^']*'", f"calculation = '{kind}'", text)


def _set_nbnd(text: str, nbnd: int) -> str:
    if re.search(r"^\s*nbnd\s*=", text, re.MULTILINE):
        return re.sub(r"^(\s*)nbnd\s*=\s*\d+", rf"\g<1>nbnd        = {nbnd}", text, flags=re.MULTILINE)
    return re.sub(r"(&system[^\n]*\n)", rf"\1    nbnd        = {nbnd}\n", text, count=1, flags=re.IGNORECASE)


def _strip(text: str, key: str) -> str:
    return re.sub(rf"^\s*{key}\s*=.*\n", "", text, flags=re.MULTILINE | re.IGNORECASE)


def prepare(epw_dir: Path, out: Path, *, random_points: int = DEFAULT_RANDOM_POINTS,
            grid_points: int = DEFAULT_GRID_POINTS, dense_grid: int = DEFAULT_DENSE_GRID,
            seed: int = 2026) -> dict[str, Any]:
    settings = run_settings(epw_dir)
    prefix = settings["prefix"]
    scf = (epw_dir / "scf.in").read_text()
    nscf_path = epw_dir / "nscf.in"
    nbnd = int(read_float(nscf_path.read_text(), "nbnd") or 0) if nscf_path.is_file() else 0
    if nbnd < 1:
        raise ValueError("cannot find nbnd in nscf.in; the band step's band count is needed")
    save = epw_dir / f"{prefix}.save"
    density = sorted(save.glob("charge-density*"))
    if not density or not (save / "data-file-schema.xml").is_file():
        raise ValueError(f"{save} lacks the charge density or data-file-schema.xml")
    pseudo_text = read_namelist_value(scf, "pseudo_dir") or "./"
    pseudo_dir = (epw_dir / pseudo_text).resolve()
    names = species_files(scf)
    missing = [name for name in names if not (pseudo_dir / name).is_file()]
    if missing:
        raise ValueError(f"pseudopotentials not found in {pseudo_dir}: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    for folder in (out, out / "dense"):
        (folder / "pp").mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(pseudo_dir / name, folder / "pp" / name)
        target = folder / f"{prefix}.save"
        target.mkdir(exist_ok=True)
        for item in density + [save / "data-file-schema.xml"]:
            shutil.copy2(item, target / item.name)

    head = scf.split("K_POINTS", 1)[0]
    head = _replace_pseudo_dir(head)
    head = re.sub(r"outdir\s*=\s*'[^']*'", "outdir      = './'", head)
    for key in ("tprnfor", "tstress", "nosym", "noinv"):
        head = _strip(head, key)
    head = _set_nbnd(head, nbnd)

    rng = np.random.default_rng(seed)
    test = rng.random((random_points, 3))
    nk = settings["coarse_grid"]
    grid = np.array([[i / nk[0], j / nk[1], k / nk[2]] for i in range(nk[0]) for j in range(nk[1]) for k in range(nk[2])])
    picks = grid[rng.choice(len(grid), size=min(grid_points, len(grid)), replace=False)]
    points = np.vstack([test, picks])
    on_grid = [False] * len(test) + [True] * len(picks)
    bands_in = _set_calculation(head, "bands") + f"K_POINTS crystal\n{len(points)}\n" + "".join(
        f"  {a:.10f} {b:.10f} {c:.10f} 1.0\n" for a, b, c in points)
    (out / "bands.in").write_text(bands_in)
    dense_in = _set_calculation(head, "nscf") + f"K_POINTS automatic\n {dense_grid} {dense_grid} {dense_grid} 0 0 0\n"
    (out / "dense" / "nscf.in").write_text(dense_in)
    run = out / "run.sh"
    run.write_text(f"""#!/usr/bin/env bash
# Written by scripts/check_wannier_bands.py prepare. Two short pw.x runs:
# bands at {len(points)} k-points, and an nscf on a {dense_grid}^3 grid for the
# density of states. LAUNCHER defaults to srun (a Slurm job); elsewhere use
#   LAUNCHER="mpirun -np 4" bash run.sh
# Not on a Hazel login node: these are calculations.
#SBATCH --job-name=rtsc-bandcheck
#SBATCH --nodes=1
#SBATCH --time=00:30:00
set -euo pipefail
cd "{out.resolve()}"
launcher="${{LAUNCHER:-srun}}"
$launcher pw.x -in bands.in > bands.out
cp {prefix}.save/data-file-schema.xml bands.xml
cd dense
$launcher pw.x -in nscf.in > nscf.out
echo "done; now run: python scripts/check_wannier_bands.py analyze --epw-dir {epw_dir.resolve()} --check-dir {out.resolve()}"
""")
    run.chmod(0o755)
    meta = {"prefix": prefix, "random_points": random_points, "grid_points": int(len(picks)),
            "dense_grid": dense_grid, "nbnd": nbnd, "seed": seed, "on_grid": on_grid,
            "epw_dir": str(epw_dir.resolve())}
    (out / "check.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------

def _print_report(report: dict[str, Any]) -> None:
    near, dos = report["errors_within_fsthick"], report["dos_at_fermi_level_per_spin"]
    print(f"Wannier band check: {report['verdict'].upper()}  ({report['num_wannier']} Wannier functions, "
          f"coarse grid {'x'.join(map(str, report['coarse_grid']))}, {report['test_kpoints']} test k-points)")
    if near.get("states"):
        print(f"  within {report['window_ev']} eV of E_F ({near['states']} states): mean {near['mean_mev']} meV, "
              f"95th percentile {near['p95_mev']} meV, worst {near['max_mev']} meV "
              f"(limits {report['thresholds']['max_mean_mev']} / {report['thresholds']['max_p95_mev']} meV)")
    for sigma, values in dos.items():
        print(f"  DOS at E_F, smearing {sigma} eV: direct {values['direct']}, Wannier {values['wannier']} "
              f"(ratio {values['ratio']})")
    if report["coarse_grid_exactness"]:
        print(f"  coarse-grid points (should be exact): worst {report['coarse_grid_exactness'].get('max_mev')} meV")
    for problem in report["problems"]:
        print(f"  problem: {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="write the direct calculation")
    p.add_argument("--epw-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--random-points", type=int, default=DEFAULT_RANDOM_POINTS)
    p.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    p.add_argument("--dense-grid", type=int, default=DEFAULT_DENSE_GRID)
    p.add_argument("--seed", type=int, default=2026)
    a = sub.add_parser("analyze", help="compare and give a verdict")
    a.add_argument("--epw-dir", type=Path, required=True)
    a.add_argument("--check-dir", type=Path, required=True)
    a.add_argument("--output", type=Path, default=None, help="JSON report (default <check_dir>/report.json)")
    a.add_argument("--max-mean-mev", type=float, default=DEFAULT_MAX_MEAN_MEV)
    a.add_argument("--max-p95-mev", type=float, default=None, help="default: EPW's degaussw")
    a.add_argument("--max-dos-deviation", type=float, default=DEFAULT_MAX_DOS_DEVIATION)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            meta = prepare(args.epw_dir, args.out, random_points=args.random_points, grid_points=args.grid_points,
                           dense_grid=args.dense_grid, seed=args.seed)
            print(f"wrote {args.out}/bands.in ({meta['random_points']} random + {meta['grid_points']} grid k-points), "
                  f"{args.out}/dense/nscf.in ({meta['dense_grid']}^3) and {args.out}/run.sh")
            return 0
        report = analyze(args.epw_dir, args.check_dir, max_mean_mev=args.max_mean_mev,
                         max_p95_mev=args.max_p95_mev, max_dos_deviation=args.max_dos_deviation)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output = args.output or args.check_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
