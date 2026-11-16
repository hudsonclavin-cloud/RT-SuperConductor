"""Generations 6 and 7: the knowledge blocks, the breeding rule and the fold bootstrap."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import children_knowledge as G  # noqa: E402


def test_matthias_electron_count_is_the_group_number_less_ten_in_the_p_block():
    assert [G.valence_electrons(s) for s in ("Mg", "Nb", "Sn", "La", "Pb")] == [2.0, 5.0, 4.0, 3.0, 4.0]


def test_debye_estimate_matches_a_hand_calculation():
    # 3,480 m/s and 10.83 cm3/mol: n = 5.561e28 per m3, (3n/4pi)^(1/3) = 2.368e9 per m, times h v / k = 395.5 K
    assert G.debye_estimate(3480.0, 10.83) == pytest.approx(395.5, abs=0.5)


def test_element_features_summarise_the_formula():
    table = G.element_table()
    names = G.element_names()
    row = dict(zip(names, G.element_features({"Nb": 0.75, "Sn": 0.25}, table)))
    assert len(names) == len(row) == 44
    assert row["superconduction_temperature:max"] == 9.25
    assert row["superconduction_temperature:mean"] == pytest.approx(0.75 * 9.25 + 0.25 * table["Sn"]["superconduction_temperature"])
    assert row["valence_electrons:mean"] == pytest.approx(4.75)
    assert row["valence_electrons:distance_from_4.7"] == pytest.approx(0.05)
    assert table["Cu"]["superconduction_temperature"] == 0.0        # no listed Tc counts as zero


def test_a_missing_property_is_left_out_of_the_average_not_counted_as_zero():
    table = {"A": {**{p: 2.0 for p in (*G.ELEMENT_PROPERTIES, "debye_estimate")}, "inverse_root_mass": 0.1, "valence_electrons": 5.0},
             "B": {**{p: math.nan for p in (*G.ELEMENT_PROPERTIES, "debye_estimate")}, "inverse_root_mass": 0.2, "valence_electrons": 3.0}}
    row = dict(zip(G.element_names(), G.element_features({"A": 0.5, "B": 0.5}, table)))
    assert row["bulk_modulus:mean"] == 2.0 and row["bulk_modulus:range"] == 0.0
    only_b = dict(zip(G.element_names(), G.element_features({"B": 1.0}, table)))
    assert math.isnan(only_b["bulk_modulus:mean"])


def test_known_at_uses_only_records_up_to_the_training_year():
    records = [(1990, 0, 0.0), (1998, 1, 4.2), (2012, 1, 9.0)]
    assert G.known_at(records, 1985) is None
    assert G.known_at(records, 1995) == (0, 0.0)
    assert G.known_at(records, 2000) == (1, 4.2)
    assert G.known_at(records, 2020) == (1, 9.0)
    assert G.record_tc([None, 0.0, 3.5, 7.0]) == 3.5


def test_breeding_follows_the_plan_s_table():
    order = ["proxy", "elements", "alloy", "family", "tc-size"]
    three = G.breed(order, {"proxy", "elements", "alloy"})
    assert three["best"]["changes"] == ("proxy",)
    assert three["best-two"]["changes"] == ("proxy", "elements")
    assert three["best-three"]["changes"] == ("proxy", "elements", "alloy")
    assert three["best-two-bagged"] == {"changes": ("proxy", "elements"), "bags": G.BAGS}
    assert three["all"]["changes"] == G.CHANGES
    one = G.breed(order, {"elements"})
    assert one["best-three"]["changes"] == ("elements",)             # uses the ones that did beat the baseline
    none = G.breed(order, set())
    assert set(none) == {"all", "baseline-bagged"} and none["baseline-bagged"] == {"changes": (), "bags": G.BAGS}


def test_columns_add_only_the_blocks_a_child_was_given():
    blocks = {"X": np.zeros((3, 4)), "elements": np.ones((3, 2)), "proxy": np.ones((3, 1)), "alloy": np.ones((3, 5)),
              "family": np.ones((3, 4))}
    rows = np.arange(3)
    assert G.columns((), blocks, rows).shape == (3, 4)
    assert G.columns(("proxy", "tc-size"), blocks, rows).shape == (3, 5)
    assert G.columns(G.CHANGES, blocks, rows).shape == (3, 4 + 2 + 1 + 5 + 4)


def test_fold_bootstrap_gives_a_positive_interval_to_a_better_child_and_not_to_a_copy():
    rng = np.random.default_rng(0)
    folds = {}
    for name in ("A", "B"):
        y = rng.integers(0, 2, 400)
        noise = rng.normal(size=400)
        folds[name] = {"y": y, "groups": np.array([f"s{i // 4}" for i in range(400)]),
                       "family": np.array(["Other"] * 300 + ["Oxides"] * 100),
                       "scores": {"baseline": y * 0.5 + noise, "copy": y * 0.5 + noise, "better": y * 2.0 + noise}}
    out = G.fold_bootstrap(folds, ["baseline", "copy", "better"], resamples=200)
    assert out["better"]["primary_gain_99"][0] > 0 and out["better"]["all_gain_99"][0] > 0
    assert out["copy"]["primary_gain_99"] == [0.0, 0.0]


def test_endpoint_is_restricted_to_the_family_asked_for():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.9])
    family = np.array(["Other", "Other", "Oxides", "Oxides"])
    assert G.endpoint(y, scores, family, "Other") == 1.0
    assert G.endpoint(y, scores, family, None) == 0.5
    assert math.isnan(G.endpoint(y[:1], scores[:1], family[:1], "Other"))
