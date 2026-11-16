#!/usr/bin/env python3
"""A quick phonon-stability screen with a machine-learned force field (``docs/rung4-shortlist2.md``, step 7).

The force constants come from displacing each atom of the primitive cell in a supercell (central differences), and the
frequencies are evaluated at the wave vectors that supercell holds exactly. With a 2x2x2 supercell that is the zone
centre and the seven zone-boundary points with coordinates 0 or 1/2. No interpolation is involved.

``frequencies`` takes any function that returns forces, so the lattice dynamics can be tested without a force field.
``chgnet_forces`` and ``chgnet_relax`` supply CHGNet (``pip install chgnet``; its weights ship inside the package).
"""
from __future__ import annotations

import itertools

import numpy as np

CM1_PER_ROOT_EV_A2_AMU = 15.633302 * 33.35641      # sqrt(eV / A^2 / amu) -> THz -> cm-1
DELTA_A = 0.02
SUPERCELL = (2, 2, 2)
UNSTABLE_BELOW_CM1 = -20.0


def supercell_map(primitive, supercell, n) -> list[tuple[int, np.ndarray]]:
    """For every supercell site: which primitive atom it is, and the lattice vector (in primitive cells) it sits at."""
    inverse = np.linalg.inv(primitive.lattice.matrix)
    out = []
    for site in supercell:
        fractional = site.coords @ inverse
        for atom in range(len(primitive)):
            shift = fractional - primitive.frac_coords[atom]
            if primitive[atom].specie == site.specie and np.allclose(shift, np.round(shift), atol=1e-4):
                out.append((atom, np.round(shift).astype(int)))
                break
        else:
            raise ValueError("a supercell site matches no primitive atom")
    return out


def frequencies(primitive, forces, n: tuple[int, int, int] = SUPERCELL, delta: float = DELTA_A) -> dict:
    """Phonon frequencies (cm-1, imaginary shown as negative) at every wave vector the supercell holds exactly.

    ``forces(structures)`` returns one (atoms, 3) array of forces in eV/A per structure. The result maps each wave
    vector (in reciprocal-lattice units) to its sorted frequencies; at the zone centre the three whole-crystal
    translations are removed, because their frequency is zero by symmetry and a force field only gets close to it.
    """
    supercell = primitive.copy()
    supercell.make_supercell(np.diag(n))
    where = supercell_map(primitive, supercell, n)
    atoms = len(primitive)
    home = [next(i for i, (atom, _) in enumerate(where) if atom == k) for k in range(atoms)]
    displaced = []
    for k, axis, sign in itertools.product(range(atoms), range(3), (1, -1)):
        moved = supercell.copy()
        step = np.zeros(3)
        step[axis] = sign * delta
        moved.translate_sites([home[k]], step, frac_coords=False, to_unit_cell=False)
        displaced.append(moved)
    result = [np.asarray(f, dtype=float) for f in forces(displaced)]
    constants = np.zeros((atoms, 3, len(supercell), 3))
    for index, (k, axis) in enumerate(itertools.product(range(atoms), range(3))):
        constants[k, axis] = -(result[2 * index] - result[2 * index + 1]) / (2.0 * delta)
    masses = np.array([float(site.specie.atomic_mass) for site in primitive])
    out = {}
    for q in itertools.product(*(np.arange(m) / m for m in n)):
        q = np.array(q)
        dynamical = np.zeros((3 * atoms, 3 * atoms), dtype=complex)
        for j, (other, lattice_vector) in enumerate(where):
            for k in range(atoms):
                phase = np.exp(2j * np.pi * (q @ (lattice_vector - where[home[k]][1])))
                dynamical[3 * k:3 * k + 3, 3 * other:3 * other + 3] += (constants[k, :, j, :] * phase
                                                                         / np.sqrt(masses[k] * masses[other]))
        dynamical = (dynamical + dynamical.conj().T) / 2.0
        squared, vectors = np.linalg.eigh(dynamical)
        values = np.sign(squared) * np.sqrt(np.abs(squared)) * CM1_PER_ROOT_EV_A2_AMU
        if not q.any():
            values = np.delete(values, translation_modes(vectors, masses))
        out[tuple(float(x) for x in q)] = np.sort(values)
    return out


def translation_modes(vectors: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """The three eigenvectors that overlap most with a rigid shift of the whole crystal."""
    shifts = np.zeros((3, 3 * len(masses)))
    for axis in range(3):
        shifts[axis, axis::3] = np.sqrt(masses / masses.sum())
    overlap = (np.abs(shifts @ vectors) ** 2).sum(axis=0)
    return np.argsort(overlap)[-3:]


def verdict(by_q: dict, threshold: float = UNSTABLE_BELOW_CM1) -> dict:
    lowest = {q: float(values.min()) for q, values in by_q.items() if len(values)}   # one atom: no mode at the centre
    worst = min(lowest, key=lowest.get)
    centre = lowest.get((0.0, 0.0, 0.0))
    return {"lowest_cm1": round(lowest[worst], 1), "at": list(worst),
            "lowest_at_zone_centre_cm1": None if centre is None else round(centre, 1),
            "unstable_points": int(sum(v < threshold for v in lowest.values())), "points": len(lowest),
            "unstable": bool(lowest[worst] < threshold)}


# ------------------------------------------------------------------------------------------------ CHGNet

_MODEL = None


def chgnet_model():
    global _MODEL
    if _MODEL is None:
        import warnings

        from chgnet.model import CHGNet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODEL = CHGNet.load(verbose=False, use_device="cpu")
    return _MODEL


def chgnet_forces(structures: list, chunk: int = 8) -> list[np.ndarray]:
    """Forces for each structure, a few at a time: building every graph at once ran a 12-atom cell out of memory."""
    import gc

    out = []
    for start in range(0, len(structures), chunk):
        rows = chgnet_model().predict_structure(structures[start:start + chunk], task="ef", batch_size=chunk)
        out += [np.array(row["f"], dtype=float) for row in rows]
        gc.collect()
    return out


def chgnet_relax(structure, fmax: float = 0.005, steps: int = 500):
    import contextlib
    import io
    import warnings

    from chgnet.model import StructOptimizer
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        result = StructOptimizer(model=chgnet_model(), use_device="cpu").relax(structure, fmax=fmax, steps=steps,
                                                                              verbose=False)
    return result["final_structure"]


def screen(structure) -> dict:
    """Relax with CHGNet, then the phonon verdict. ``structure`` is a pymatgen Structure."""
    relaxed = chgnet_relax(structure)
    out = verdict(frequencies(relaxed, chgnet_forces))
    out["volume_change_percent"] = round(100.0 * (relaxed.volume / structure.volume - 1.0), 2)
    return out
