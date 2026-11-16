"""The second physics proxy: reading the two calculated sets and combining them, without any model."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pymatgen")

import physics_proxy2 as P  # noqa: E402

MCMILLAN = """ #        McMillan
 # integral a2F  =  0.64658903032150039
 # lambda        =   8.2211586940981074E-002
 # wlog[K]       =   170.41239889016805
 # w2av[K]       =   193.17756176676400
 # mustr - TcMcMillan - TcAllenDynes
     0.000     0.000     0.000
"""
CIF = "data_HfPd3\\n_chemical_formula_structural   HfPd3\\n_chemical_formula_sum   'Hf1 Pd3'\\n_cell_volume   61.77\\n"


def test_reads_lambda_and_omega_log_from_an_alexandria_file():
    """Lines copied from a real McMillan.dat (Pd3Hf)."""
    lam, wlog = P.read_mcmillan(MCMILLAN)
    assert lam == pytest.approx(0.0822116) and wlog == pytest.approx(170.4124)
    assert P.read_mcmillan("# nothing here") is None


def test_reads_the_formula_from_a_cif():
    assert P.cif_formula(CIF) == "Hf1Pd3" and P.cif_formula("no formula") is None


def test_usable_rejects_values_that_mark_an_unstable_or_broken_calculation():
    assert P.usable(0.7, 160.0)
    assert not P.usable(4.5, 160.0) and not P.usable(0.0, 160.0) and not P.usable(0.7, 0.0)
    assert not P.usable(float("nan"), 160.0)


def test_allen_dynes_matches_a_hand_calculation():
    # lambda 0.7572, omega_log 164.9922 K, mu* 0.10: 164.9922 / 1.2 * exp(-1.04 * 1.7572 / (0.7572 - 0.1 * 1.469464))
    expected = 164.9922 / 1.2 * math.exp(-1.04 * 1.7572 / (0.7572 - 0.1 * (1 + 0.62 * 0.7572)))
    assert P.allen_dynes(0.7572, 164.9922) == pytest.approx(expected) == pytest.approx(6.882, abs=0.001)
    assert P.allen_dynes(0.08, 170.0) == 0.0                         # coupling too weak for the formula


def test_structures_of_one_formula_are_averaged_in_logarithms():
    rows = [{"source": "alexandria", "id": "a", "formula": "Nb3Sn", "lambda": 1.0, "omega_log_k": 100.0},
            {"source": "jarvis", "id": "b", "formula": "Nb6Sn2", "lambda": 4.0, "omega_log_k": 400.0},
            {"source": "jarvis", "id": "c", "formula": "LaPb3", "lambda": 0.5, "omega_log_k": 80.0},
            {"source": "jarvis", "id": "d", "formula": "Xx2", "lambda": 0.5, "omega_log_k": 80.0}]
    table, counts = P.by_composition(rows, {"Nb", "Sn", "La", "Pb"})
    assert counts["formula_not_usable"] == 1 and len(table) == 2
    nb = next(r for r in table if r["element_system"] == "Nb-Sn")
    assert nb["lambda"] == pytest.approx(2.0) and nb["omega_log_k"] == pytest.approx(200.0)
    assert nb["structures"] == 2 and nb["sources"] == "alexandria+jarvis" and nb["tc_above_1k"] == 1
    la = next(r for r in table if r["element_system"] == "La-Pb")
    assert la["has_f_element"] == 1 and la["tc_k"] == pytest.approx(P.allen_dynes(0.5, 80.0), abs=1e-4)
