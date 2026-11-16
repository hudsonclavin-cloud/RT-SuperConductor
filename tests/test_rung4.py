"""Rung 4 steps 1 and 2: prototypes, pool filters, crystal systems and folds."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import rung4 as R  # noqa: E402


def row(formula, sg="221", theoretical="False", metal="True", known="", tc="", magnetic="False", hull="0.0"):
    return {"formula": formula, "spacegroup_number": sg, "theoretical": theoretical, "is_metal": metal,
            "known_match": known, "known_tc_max_k": tc, "is_magnetic": magnetic,
            "energy_above_hull_ev_per_atom": hull, "mp_id": "mp-" + formula}


def test_prototype_ignores_which_elements_fill_the_sites():
    assert R.prototype("Nb3Sn", "223") == R.prototype("V3Si", "223")
    assert R.prototype("Nb3Sn", "223") != R.prototype("Nb3Sn", "221")
    assert R.prototype("NbN", "225") != R.prototype("Nb3Sn", "223")


def test_known_families_need_three_measured_superconductors():
    rows = [row("Nb3Sn", "223", known="exact", tc="18"), row("V3Si", "223", known="exact", tc="17"),
            row("Nb3Ge", "223", known="exact", tc="23"), row("NbN", "225", known="exact", tc="16"),
            row("TiN", "225", known="exact", tc="5"), row("Mo3Os", "223", theoretical="True", known="exact", tc="12")]
    families = R.superconductor_prototypes(rows)
    assert list(families.values()) == [3]                    # A3B in 223 only; the calculated row does not count


def test_pool_filters_in_the_plans_order():
    known = [row(f, "223", known="exact", tc="10") for f in ("Nb3Sn", "V3Si", "Nb3Ge")]
    new = [row("Ta3Sb", "223", theoretical="True"),                     # kept, on hull
           row("Zr3Pb", "223", theoretical="True", hull="0.02"),        # kept, above hull
           row("Cu3Sb", "223", theoretical="True"),                     # copper
           row("Fe3Ga", "223", theoretical="True", magnetic="True"),    # magnetic
           row("NbSn", "221", theoretical="True"),                      # not a known prototype
           row("Mo3Ir", "223", theoretical="True", hull="0.03"),        # above the table's cap
           row("Ti3Au", "223", theoretical="True", known="exact")]      # already in a database
    pool, steps = R.pool_steps(known + new, R.superconductor_prototypes(known))
    assert [r["formula"] for r in pool] == ["Ta3Sb", "Zr3Pb"]
    assert steps == {"never_made_untested_metals": 5, "in_known_superconductor_prototype": 4, "not_magnetic": 3,
                     "no_copper": 2, "on_hull": 1}


def test_crystal_system_boundaries():
    assert [R.crystal_system(n) for n in (1, 2, 3, 15, 16, 74, 75, 142, 143, 167, 168, 194, 195, 230)] == \
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    with pytest.raises(ValueError):
        R.crystal_system(231)


def test_system_fold_is_stable_and_spread():
    assert R.system_fold("Nb-Sn") == R.system_fold("Nb-Sn")
    assert len({R.system_fold(f"X{i}-Y") for i in range(200)}) == 5
