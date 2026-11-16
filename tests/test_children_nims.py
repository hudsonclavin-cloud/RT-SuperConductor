"""Generations 3 to 5 on NIMS data: record labels, time slices and mutations."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import children_nims as N  # noqa: E402


def test_record_label_reads_tc_then_measured_null():
    assert N.record_label([None, 4.2, None], None) == 1
    assert N.record_label([None, None], 1.5) == 0          # not superconducting down to 1.5 K
    assert N.record_label([0.0], None) == 0
    assert N.record_label([None, None], None) is None
    assert N.record_label([3.0], 1.5) == 1                 # a Tc wins over a null in the same record


def test_label_at_uses_only_what_was_published_by_then():
    records = [(2008, 0), (2013, 1)]
    assert N.label_at(records, 2010) == 0 and N.label_at(records, 2015) == 1 and N.label_at(records, 2005) is None


def test_cut_slice_keeps_only_new_element_systems_in_the_window():
    comps = {"old": {"Nb": 0.75, "Sn": 0.25}, "variant": {"Nb": 0.74, "Sn": 0.26}, "same_system": {"Nb": 0.5, "Sn": 0.5},
             "new": {"La": 0.75, "Ir": 0.25}, "late": {"Ca": 0.5, "Sb": 0.5}}
    first = {"old": 2000, "variant": 2012, "same_system": 2013, "new": 2014, "late": 2018}
    cut = N.cut_slice(comps, first, train_until=2010, first=2011, last=2015)
    assert cut["train"] == ["old"] and cut["window"] == 3 and cut["not_near"] == 2 and cut["test"] == ["new"]


def test_mutate_moves_one_setting_one_step_and_is_repeatable():
    for kind in N.KINDS:
        start = N.START[kind]
        for generation in N.GENERATIONS:
            for m in (1, 2):
                child = N.mutate(kind, start, generation, m)
                assert child == N.mutate(kind, start, generation, m)
                changed = [k for k in start if child[k] != start[k]]
                assert len(changed) == 1
                ladder = [tuple(s) if isinstance(s, (list, tuple)) else s for s in N.LADDERS[kind][changed[0]]]
                assert abs(ladder.index(child[changed[0]]) - ladder.index(start[changed[0]])) == 1


def test_every_start_sits_on_its_ladder():
    for kind in N.KINDS:
        for name, ladder in N.LADDERS[kind].items():
            assert N.START[kind][name] in ladder
    assert all(N.BASELINE[1][name] in N.BOOST_LADDER[name] for name in N.BOOST_LADDER)
