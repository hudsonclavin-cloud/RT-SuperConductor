"""The five-children experiment: record parsing, hit rules and the percentile score."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import children as K  # noqa: E402


def test_parse_tc_takes_plain_kelvin_values_only():
    assert K.parse_tc("2.8 K") == 2.8 and K.parse_tc(" 39K") == 39.0
    for text in ("", "0.71.4 K", "2-4 K", "< 2 K", "about 5 K", "12 mK"):
        assert K.parse_tc(text) is None


def test_slices_split_at_2016():
    assert [K.slice_of(y) for y in (1999, 2015, 2016, 2021)] == ["selection", "selection", "final", "final"]


def test_hit_status_drops_pressure_only_and_hydrogen_rich():
    ambient, pressed = {"pressure": False}, {"pressure": True}
    assert K.hit_status([pressed, ambient], {"La": 0.5, "Ir": 0.5}) == "hit"
    assert K.hit_status([pressed], {"La": 0.5, "Ir": 0.5}) == "excluded_pressure_only"
    assert K.hit_status([ambient], {"La": 0.1, "H": 0.9}) == "excluded_hydrogen_rich"


def test_percentiles_count_ties_as_half():
    scores = np.array([0.9, 0.5, 0.5, 0.1, 0.5])
    hits, negatives = np.array([0, 4]), np.array([1, 2, 3])
    assert K.percentiles(scores, hits, negatives).tolist() == [1.0, (1 + 0.5 * 2) / 3]
    summary = K.summarise(scores, hits, negatives)
    assert summary["hits"] == 2 and summary["hits_in_top_1_percent"] == 1


def test_paired_interval_is_zero_for_identical_scores_and_signed_otherwise():
    a = np.linspace(0.2, 0.9, 30)
    same = K.paired_interval(a, a)
    assert same["difference"] == 0 and same["low"] == 0 and same["high"] == 0
    better = K.paired_interval(a + 0.05, a)
    assert better["low"] > 0 and abs(better["difference"] - 0.05) < 1e-9


def test_rank_average_is_order_only():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 30.0, 20.0])
    assert K.rank_average([a, b]).tolist() == K.rank_average([a * 100, b - 5]).tolist()


def test_bag_rows_keeps_whole_groups():
    groups = np.array(["a", "a", "b", "c", "c"])
    bag = K.bag_rows(groups, seed=1)
    assert np.array_equal(bag, K.bag_rows(groups, seed=1))
    for name in "abc":
        counts = {int((bag == row).sum()) for row in np.flatnonzero(groups == name)}
        assert len(counts) == 1


def test_every_generation_1_kind_has_two_variants():
    assert set(K.VARIANTS) == set(K.GEN1) and all(len(v) == 2 for v in K.VARIANTS.values())


def test_feature_group_names_the_property():
    assert K.feature_group("magpie__MagpieData mean Electronegativity") == "Electronegativity"
    assert K.feature_group("shell__descriptor_shell_3_electrons") == "shell filling"
