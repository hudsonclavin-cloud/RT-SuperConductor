"""Rung 4 shortlist 2: the filters in the plan's order, and the capped ranking."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rung4_shortlist2 as S  # noqa: E402


def row(elements, formation=-0.3, sites=4, on_hull=True):
    return {"on_hull": on_hull, "elements": set(elements), "formation": formation, "num_sites": sites}


def test_each_filter_removes_what_the_plan_says_in_order():
    rows = [row({"Nb", "Sn"}), row({"Nb", "Sn"}, on_hull=False), row({"Yb", "Pb"}), row({"Ac", "Pb"}),
            row({"Nb", "Tl"}), row({"La", "Pb"}), row({"Re", "Ir"}, formation=-0.02), row({"Nb", "Sn"}, sites=20)]
    keep, steps = S.element_steps(rows, {"Nb", "Sn", "Pb", "Tl", "Re", "Ir"})
    assert steps == {"pool": 8, "1_on_hull": 7, "2_not_radioactive_no_open_4f": 5, "3_no_toxic_element": 4,
                     "4_pseudopotentials_available": 3, "5_formation_energy": 2, "6_cell_size": 1}
    assert keep == [rows[0]]


def test_the_formation_energy_limit_is_inclusive_and_negative():
    keep, _ = S.element_steps([row({"Nb"}, formation=-0.10), row({"Nb"}, formation=-0.0999)], {"Nb"})
    assert len(keep) == 1 and S.FORMATION_AT_MOST == -0.10


def test_first_shortlist_s_failures_would_be_filtered():
    """AcYbTl2, NaYbPb2, CaYbPb2 and La3Pb each fall to a rule; SbTe2 passes the element rules."""
    available = S.available_pseudopotentials()
    for elements in ({"Ac", "Yb", "Tl"}, {"Na", "Yb", "Pb"}, {"Ca", "Yb", "Pb"}, {"La", "Pb"}):
        assert S.element_steps([row(elements)], available)[0] == []
    assert len(S.element_steps([row({"Sb", "Te"})], available)[0]) == 1


def test_capped_head_limits_prototypes_and_element_systems():
    rows = [{"element_system": "A-B", "prototype": "p1", "n": 1}, {"element_system": "A-B", "prototype": "p2", "n": 2},
            {"element_system": "A-B", "prototype": "p3", "n": 3}, {"element_system": "C-D", "prototype": "p1", "n": 4},
            {"element_system": "E-F", "prototype": "p1", "n": 5}, {"element_system": "G-H", "prototype": "p4", "n": 6}]
    kept = S.capped_head(rows, size=4)
    assert [r["n"] for r in kept] == [1, 2, 4, 6]          # third A-B and third p1 are skipped
