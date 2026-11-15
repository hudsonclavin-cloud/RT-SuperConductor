#!/usr/bin/env python3
"""Write EPW a fine k-point list holding only the points near the Fermi level.

Why: EPW 6.1's memory for the phonon self-energy grows with the number of fine
k-points, about 0.31 MB per point for niobium with 9 Wannier functions, however
many processes share the work. The full 36x36x36 grid (46,656 points) needs
about 14 GB, more than a 15 GB machine can spare. EPW's own fix for this,
`etf_mem = 3` (lfast_kmesh), keeps only points within `fsthick` of the Fermi
level, but EPW refuses it for the phonon self-energy.

This script does the same by hand. It interpolates the bands on the full
uniform grid (the same Gamma-centred grid EPW builds for nkf1..3) from
Wannier90's <prefix>_hr.dat and <prefix>_wsvec.dat, and keeps every point
with at least one band within `window` eV of the Fermi level. Every kept
point gets the full grid's weight 1/N.

Why the result should be the same as the full grid:
- EPW drops every state further than `fsthick` from E_F from the coupling sums.
  With window > fsthick, every state EPW would use is kept.
- The density of states at E_F is a Gaussian sum with width `degaussw`. A state
  further than `window` from E_F has weight exp(-(window/degaussw)^2): e^-144
  for 0.6 eV at 0.05 eV.
- If EPW rescales the weights to a fixed total, it rescales the coupling sum
  and the density of states by the same factor, and lambda is their ratio.
  The validation run (full grid against list on 24x24x24) checks this.
- E_F must be fixed (`efermi_read`, `fermi_energy`) to the full grid's value,
  because EPW would otherwise compute it from the kept points only.

Usage:

    python scripts/epw_fermi_klist.py --epw-dir <run>/epw --prefix nb --n 36 \
        --fermi 17.819144 --window 0.6 --out fermi36.kf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_wannier_bands import WannierModel, read_hr, read_wsvec, uniform_grid  # noqa: E402


def dense_terms(model: WannierModel) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the model's terms into one matrix per distinct lattice vector: H(k) = sum_v e^{2 pi i k.v} M_v."""
    nw = model.num_wann
    keys, index = np.unique(model._vectors, axis=0, return_inverse=True)
    index = np.asarray(index).reshape(-1)
    matrices = np.zeros((len(keys), nw * nw), complex)
    np.add.at(matrices, (index, model._rows * nw + model._cols), model._values)
    return keys, matrices


def bands_on(model: WannierModel, kfrac: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Interpolated eigenvalues (eV) at many fractional k-points, in chunks to bound memory."""
    vectors, matrices = dense_terms(model)
    nw = model.num_wann
    kfrac = np.atleast_2d(np.asarray(kfrac, float))
    out = np.empty((len(kfrac), nw))
    for start in range(0, len(kfrac), chunk):
        k = kfrac[start:start + chunk]
        h = (np.exp(2j * np.pi * k @ vectors.T) @ matrices).reshape(len(k), nw, nw)
        out[start:start + chunk] = np.linalg.eigvalsh(0.5 * (h + np.conj(np.transpose(h, (0, 2, 1)))))
    return out


def select(bands: np.ndarray, fermi: float, window: float) -> np.ndarray:
    """True where at least one band lies within `window` of the Fermi level."""
    return np.any(np.abs(bands - fermi) <= window, axis=1)


def write_klist(path: Path, kfrac: np.ndarray, weight: float) -> None:
    lines = [f"{len(kfrac)} crystal"]
    lines += [f"{k[0]:.12f} {k[1]:.12f} {k[2]:.12f} {weight:.12e}" for k in kfrac]
    path.write_text("\n".join(lines) + "\n")


def build(epw_dir: Path, prefix: str, n: int, fermi: float, window: float, sigma: float, out: Path) -> dict:
    model = WannierModel(read_hr(epw_dir / f"{prefix}_hr.dat"), read_wsvec(epw_dir / f"{prefix}_wsvec.dat"))
    grid = uniform_grid(n)
    bands = bands_on(model, grid)
    keep = select(bands, fermi, window)
    weight = 1.0 / len(grid)
    write_klist(out, grid[keep], weight)
    gauss = np.exp(-((bands - fermi) / sigma) ** 2) / (sigma * np.sqrt(np.pi))
    dos_full = float(gauss.sum() * weight)
    dos_kept = float(gauss[keep].sum() * weight)
    excluded = np.abs(bands[~keep] - fermi)
    return {
        "grid": n, "points": int(len(grid)), "kept": int(keep.sum()), "kept_fraction": round(float(keep.mean()), 4),
        "fermi_ev": fermi, "window_ev": window, "degaussw_ev": sigma,
        "closest_excluded_state_ev": None if excluded.size == 0 else round(float(excluded.min()), 4),
        "dos_full_per_spin": dos_full, "dos_kept_per_spin": dos_kept,
        "dos_relative_difference": abs(dos_full - dos_kept) / dos_full if dos_full else None,
        "out": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epw-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--n", type=int, required=True, help="fine grid size per direction (nkf1 = nkf2 = nkf3)")
    p.add_argument("--fermi", type=float, required=True, help="E_F in eV from the full fine grid")
    p.add_argument("--window", type=float, default=0.6, help="keep points with a band within this many eV of E_F")
    p.add_argument("--degaussw", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    summary = build(a.epw_dir, a.prefix, a.n, a.fermi, a.window, a.degaussw, a.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
