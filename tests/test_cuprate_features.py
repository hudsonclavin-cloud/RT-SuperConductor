import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pymatgen.core")

from cuprate_features import (  # noqa: E402
    FEATURE_NAMES, doping_features, disorder_features, load_overrides, override_for, plane_geometry,
    presland_dome,
)

OVERRIDES = ROOT / "data/training/cuprate-parent-overrides"
TABLE = ROOT / "data/training/cuprate-features.csv"


def _ybco():
    """YBa2Cu3O7 from textbook coordinates: two CuO2 planes and one Cu-O chain."""
    from pymatgen.core import Lattice, Structure

    species = ["Y", "Ba", "Ba", "Cu", "Cu", "Cu", "O", "O", "O", "O", "O", "O", "O"]
    coords = [[.5, .5, .5], [.5, .5, .184], [.5, .5, .816], [0, 0, 0], [0, 0, .355], [0, 0, .645],
              [0, .5, 0], [.5, 0, .378], [.5, 0, .622], [0, .5, .378], [0, .5, .622], [0, 0, .159], [0, 0, .841]]
    return Structure(Lattice.orthorhombic(3.82, 3.89, 11.68), species, coords)


def test_presland_dome_peaks_at_optimal_doping_and_is_clipped():
    assert presland_dome(0.16) == pytest.approx(1.0)
    assert presland_dome(0.05) == pytest.approx(1 - 82.6 * 0.11 ** 2)
    assert presland_dome(0.30) == 0.0
    assert presland_dome(-0.16) == pytest.approx(1.0)      # electron-doped side uses |p|


def test_doping_from_charge_balance():
    lsco = doping_features("La1.85Sr0.15CuO4")
    assert lsco["hole_doping_p"] == pytest.approx(0.15)
    assert lsco["electron_doped"] == 0.0 and lsco["valence_ambiguous"] == 0.0
    ncco = doping_features("Nd1.85Ce0.15CuO4")
    assert ncco["hole_doping_p"] == pytest.approx(-0.15)
    assert ncco["electron_doped"] == 1.0
    assert ncco["valence_ambiguous"] == 1.0                 # cerium's valence is not certain
    assert doping_features("YBa2Cu3O7")["cu_valence_formula"] == pytest.approx(7 / 3)


def test_unknown_valence_gives_no_number_rather_than_a_guess():
    out = doping_features("XeCuO2")
    assert math.isnan(out["hole_doping_p"]) and out["valence_ambiguous"] == 1.0


def test_disorder_counts_substitution():
    out = disorder_features("La1.85Sr0.15CuO4")
    assert out["fractional_species_count"] == 2.0
    assert out["largest_minor_fraction"] == pytest.approx(0.15)
    assert disorder_features("YBa2Cu3O7")["fractional_species_count"] == 0.0


def test_planes_are_told_apart_from_chains_in_ybco():
    geometry = plane_geometry(_ybco())
    assert geometry["planar_cu_fraction"] == pytest.approx(2 / 3)      # the chain copper is not a plane
    assert geometry["cuo2_layer_count"] == 2.0
    assert geometry["apical_o_count_mean"] == 1.0
    assert geometry["apical_o_distance_min_a"] == pytest.approx((0.355 - 0.159) * 11.68, abs=0.01)
    assert geometry["plane_buckling_rms_a"] == pytest.approx((0.378 - 0.355) * 11.68, abs=0.01)
    assert geometry["infinite_layer"] == 0.0


def test_the_two_corrected_parent_structures_have_the_geometry_the_rules_claim():
    from pymatgen.core import Structure

    t_phase = plane_geometry(Structure.from_file(str(OVERRIDES / "La2CuO4-T-phase-mp-aaaabdfb.cif")))
    assert t_phase["cuo2_layer_count"] == 1.0 and t_phase["apical_o_count_mean"] == 2.0
    assert 2.3 < t_phase["apical_o_distance_min_a"] < 2.6
    infinite = plane_geometry(Structure.from_file(str(OVERRIDES / "SrCuO2-infinite-layer-mp-aaaacdmw.cif")))
    assert infinite["infinite_layer"] == 1.0 and infinite["has_apical_oxygen"] == 0.0


def test_structure_without_copper_oxygen_planes_reports_none():
    from pymatgen.core import Lattice, Structure

    rocksalt = Structure(Lattice.cubic(4.2), ["Cu", "O"], [[0, 0, 0], [.5, .5, .5]])
    assert plane_geometry(rocksalt)["cuo2_layer_count"] == 0.0


def test_override_rules_use_the_formula_only():
    rules = load_overrides(OVERRIDES)
    assert override_for("La1.85Sr0.15CuO4", "mp-1077929", rules)["replacement_id"] == "mp-aaaabdfb"
    assert override_for("La1.9Ce0.1CuO4", "mp-1077929", rules) is None        # electron-doped: really T'
    assert override_for("Sr0.86Nd0.14CuO2", "mp-5787", rules)["replacement_id"] == "mp-aaaacdmw"
    assert override_for("SrCuO2", "mp-5787", rules) is None                    # phase unknown from the formula
    assert override_for("La1.85Sr0.15CuO4", "mp-4158", rules) is None
    assert all((OVERRIDES / rule["replacement_cif"]).is_file() for rule in rules)
    conditions = {key for rule in rules for key in rule if key.startswith("applies_")}
    assert conditions <= {"applies_if_formula_contains", "applies_unless_formula_contains"}   # never the Tc label


def test_committed_table_matches_the_feature_list():
    with TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 874
    assert all(name in rows[0] for name in FEATURE_NAMES)
    lsco = next(row for row in rows if row["formula"] == "Cu1La1.72Sr0.28O4")
    assert lsco["geometry_overridden"] == "1" and float(lsco["apical_o_count_mean"]) == 2.0
    ncco = [row for row in rows if row["parent_formula"] == "Nd2CuO4"]
    assert ncco and all(row["geometry_overridden"] == "0" and float(row["apical_o_count_mean"]) == 0.0 for row in ncco)


def test_family_groupings():
    from eval_cuprate_models import family_groups

    rows = [{"formula": "La1.85Sr0.15CuO4", "mp_parent_id": "a"}, {"formula": "La1.8Sr0.2CuO4", "mp_parent_id": "a"},
            {"formula": "La1.9Sr0.1CuO4", "mp_parent_id": "b"}, {"formula": "YBa2Cu3O7", "mp_parent_id": "c"}]
    parent = family_groups(rows, "parent")
    assert parent[0] == parent[1] and len(set(parent)) == 3
    strict = family_groups(rows, "strict")
    assert strict[0] == strict[1] == strict[2] and strict[3] != strict[0]    # same elements join a and b


def test_evaluation_holds_out_whole_families_and_reports_every_model():
    pytest.importorskip("sklearn")
    from eval_cuprate_models import MODELS, evaluate

    with TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parents = sorted({row["mp_parent_id"] for row in rows})[:40]
    subset = [row for row in rows if row["mp_parent_id"] in parents]
    rng = np.random.default_rng(0)
    fake_formula = (rng.normal(size=(len(subset), 4)), ["magpie__a", "magpie__b", "magpie__c", "magpie__d"])
    report = evaluate(subset, grouping="parent", repeats=1, splits=3, formula_features=fake_formula)
    assert report["families"] == 40 and len(report["fold_scores"]) == 3
    assert sum(fold["test_families"] for fold in report["fold_scores"]) == 40     # each family tested once
    assert set(report["results"]) == set(MODELS)
    assert report["results"]["cuprate_only"]["mae_k"]["mean"] > 0
    assert "two_part_learned" in report["paired_against_formula"]
