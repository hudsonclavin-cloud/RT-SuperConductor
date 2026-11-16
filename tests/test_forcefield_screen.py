"""The force-field phonon screen: lattice dynamics checked against springs with a known answer (no force field)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402

import forcefield_screen as F  # noqa: E402

A, SPRING, MASS = 3.0, 2.0, 92.90637        # angstrom, eV/A^2, niobium's mass


def spring_forces(reference, constant: float = SPRING):
    """Forces of a simple-cubic crystal whose atoms are tied to their six neighbours by isotropic springs."""
    def forces(structures):
        out = []
        for structure in structures:
            shift = structure.cart_coords - reference.cart_coords
            total = np.zeros_like(shift)
            for i, site in enumerate(reference):
                for neighbour in reference.get_neighbors(site, A * 1.01):
                    total[i] += constant * (shift[neighbour.index] - shift[i])
            out.append(total)
        return out
    return forces


@pytest.fixture(scope="module")
def cubic():
    primitive = Structure(Lattice.cubic(A), ["Nb"], [[0, 0, 0]])
    supercell = primitive.copy()
    supercell.make_supercell(np.diag(F.SUPERCELL))
    return primitive, F.frequencies(primitive, spring_forces(supercell))


def analytic(q) -> float:
    """omega^2 = (2K/M) * sum over axes of (1 - cos(2 pi q)), in cm-1."""
    return float(np.sqrt(2 * SPRING / MASS * sum(1 - np.cos(2 * np.pi * x) for x in q)) * F.CM1_PER_ROOT_EV_A2_AMU)


def test_spring_crystal_matches_the_analytic_frequencies_at_every_point(cubic):
    _, by_q = cubic
    assert len(by_q) == 8
    for q, values in by_q.items():
        if not any(q):
            assert len(values) == 0                       # one atom: the zone centre holds only the three translations
        else:
            assert values == pytest.approx([analytic(q)] * 3, rel=1e-6)


def test_verdict_reports_the_lowest_mode_and_where():
    by_q = {(0.0, 0.0, 0.0): np.array([50.0, 60.0]), (0.5, 0.5, 0.5): np.array([-62.5, 80.0]),
            (0.5, 0.0, 0.0): np.array([-10.0, 70.0])}
    out = F.verdict(by_q)
    assert out["unstable"] is True and out["lowest_cm1"] == -62.5 and out["at"] == [0.5, 0.5, 0.5]
    assert out["unstable_points"] == 1 and out["lowest_at_zone_centre_cm1"] == 50.0
    assert F.verdict({(0.0, 0.0, 0.0): np.array([5.0]), (0.5, 0.0, 0.0): np.array([-19.0])})["unstable"] is False


def test_negative_springs_give_imaginary_modes_shown_as_negative():
    primitive = Structure(Lattice.cubic(A), ["Nb"], [[0, 0, 0]])
    supercell = primitive.copy()
    supercell.make_supercell(np.diag(F.SUPERCELL))
    by_q = F.frequencies(primitive, spring_forces(supercell, constant=-SPRING))
    assert F.verdict(by_q)["unstable"] is True
    assert by_q[(0.5, 0.5, 0.5)] == pytest.approx([-analytic((0.5, 0.5, 0.5))] * 3, rel=1e-6)


def test_two_atom_cell_keeps_its_optical_modes_at_the_zone_centre():
    """CsCl-type cell with springs between unlike neighbours: three translations removed, three optical modes left."""
    primitive = Structure(Lattice.cubic(A), ["Nb", "Sn"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    supercell = primitive.copy()
    supercell.make_supercell(np.diag(F.SUPERCELL))

    def forces(structures):
        out = []
        for structure in structures:
            shift = structure.cart_coords - supercell.cart_coords
            total = np.zeros_like(shift)
            for i, site in enumerate(supercell):
                for neighbour in supercell.get_neighbors(site, A * 0.9):
                    total[i] += SPRING * (shift[neighbour.index] - shift[i])
            out.append(total)
        return out

    by_q = F.frequencies(primitive, forces)
    optical = by_q[(0.0, 0.0, 0.0)]
    m1, m2 = 92.90637, 118.71
    expected = np.sqrt(8 * SPRING * (1 / m1 + 1 / m2)) * F.CM1_PER_ROOT_EV_A2_AMU     # eight unlike neighbours
    assert optical == pytest.approx([expected] * 3, rel=1e-3)
