"""Composition parsing, encoding, grouping and splitting for the from-scratch network."""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rt_superconductor.scratchnet import data as D  # noqa: E402

ELEMENT_TABLE = ROOT / "data/training/element-features.csv"


def test_parse_formula_handles_decimals_and_nested_brackets():
    lsco = D.parse_formula("La1.85Sr0.15CuO4")
    assert lsco["La"] == pytest.approx(1.85 / 7) and lsco["O"] == pytest.approx(4 / 7)
    nested = D.parse_formula("Ba(Fe0.9Co0.1)2As2")
    assert nested["Fe"] == pytest.approx(1.8 / 5) and nested["Co"] == pytest.approx(0.2 / 5)
    assert D.parse_formula("Ca[Cu(OH)2]2")["H"] == pytest.approx(4 / 11)
    assert sum(D.parse_formula("YBa2Cu3O6.93").values()) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["", "La(Sr", "LaSr)", "la2CuO4", "La2-xSrxCuO4", "123"])
def test_parse_formula_rejects_what_it_cannot_read(bad):
    with pytest.raises(ValueError):
        D.parse_formula(bad)


def test_composition_key_ignores_how_the_formula_is_written():
    assert D.composition_key(D.parse_formula("MgB2")) == D.composition_key(D.parse_formula("Mg0.5B")) \
        == D.composition_key(D.parse_formula("B4Mg2")) == "B:0.667;Mg:0.333"
    assert D.parse_key("B:0.667;Mg:0.333") == {"B": 0.667, "Mg": 0.333}


def test_stable_hash_is_the_same_everywhere():
    assert D.stable_hash("scratchnet-split:B-Mg") == D.stable_hash("scratchnet-split:B-Mg")
    assert D.stable_hash("a") == int("ca978112ca1b", 16)       # sha256("a") starts ca978112ca1b...
    assert D.stable_hash("a") != D.stable_hash("b")


def test_committed_element_table_is_complete_and_standardized():
    with ELEMENT_TABLE.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 90 and len(raw[0]) == 29
    assert all(value not in ("", "nan") for row in raw for value in row.values())
    assert not {"He", "Ne", "Ar", "At", "Rn", "Fr"} & {row["symbol"] for row in raw}    # gaps, not imputed
    table = D.ElementTable.load(ELEMENT_TABLE)
    assert table.features.shape == (91, 28) and not table.features[0].any()
    np.testing.assert_allclose(table.features[1:].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(table.features[1:].std(axis=0), 1.0, atol=1e-4)
    assert table.index["H"] == 1 and table.symbols[table.index["Cu"] - 1] == "Cu"


def test_encode_orders_by_amount_and_pads():
    table = D.ElementTable.load(ELEMENT_TABLE)
    index, fraction, mask = D.encode([D.parse_formula("YBa2Cu3O7"), D.parse_formula("Nb3Sn")], table, max_elements=5)
    assert index.shape == (2, 5) and mask.sum(axis=1).tolist() == [4, 2]
    assert [table.symbols[i - 1] for i in index[0, :4]] == ["O", "Cu", "Ba", "Y"]
    np.testing.assert_allclose(fraction.sum(axis=1), 1.0, rtol=1e-6)
    assert not index[1, 2:].any() and not fraction[1, 2:].any()
    with pytest.raises(ValueError):
        D.encode([D.parse_formula("YBa2Cu3O7")], table, max_elements=3)
    with pytest.raises(KeyError):
        D.encode([{"He": 1.0}], table)


def test_write_table_is_byte_identical_on_rerun(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    D.write_table(tmp_path / "one.csv.gz", rows, ["a", "b"])
    D.write_table(tmp_path / "two.csv.gz", rows, ["a", "b"])
    assert (tmp_path / "one.csv.gz").read_bytes() == (tmp_path / "two.csv.gz").read_bytes()
    assert D.read_table(tmp_path / "one.csv.gz") == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


# ---------------------------------------------------------------------------
# Grouping and splitting (scripts/build_scratchnet_data.py)

build = pytest.importorskip("build_scratchnet_data")


def test_major_group_drops_dopants_below_five_percent():
    assert build.major_group(D.parse_formula("YBa2Cu3O7")) == "Ba-Cu-O-Y"
    assert build.major_group(D.parse_formula("YBa2Cu2.9Zn0.1O7")) == "Ba-Cu-O-Y"      # 0.8% zinc
    assert build.major_group(D.parse_formula("La1.6Sr0.4CuO4")) == "Cu-La-O-Sr"       # 5.7% strontium


def test_merge_groups_chains_near_twins_and_names_by_the_largest_member():
    major = ["A", "A", "A", "B", "C", "C", "D"]
    merged = build.merge_groups(major, [(2, 3), (3, 4)])        # A~B and B~C: one chain; D stands alone
    assert merged == ["A", "A", "A", "A", "A", "A", "D"]
    assert build.merge_groups(["B", "A"], [(0, 1)]) == ["A", "A"]   # a tie in size goes to the alphabetically first


def test_near_pairs_uses_the_largest_difference_on_any_element():
    compositions = [{"Cu": 0.5, "O": 0.5}, {"Cu": 0.485, "O": 0.5, "Zn": 0.015}, {"Cu": 0.47, "O": 0.5, "Zn": 0.03}]
    assert sorted(build.near_pairs(compositions, 0.02)) == [(0, 1), (1, 2)]


def test_assign_split_is_deterministic_and_keeps_huge_groups_for_training():
    assert build.assign_split("B-Mg", 10, 1000) == build.assign_split("B-Mg", 10, 1000)
    assert build.assign_split("Ba-Cu-O-Y", 60, 1000) == ("train_only", -1)
    splits = Counter(build.assign_split(f"group-{i}", 1, 10_000)[0] for i in range(4000))
    assert 0.12 < splits["lockbox"] / 4000 < 0.18
    folds = Counter(build.assign_split(f"group-{i}", 1, 10_000)[1] for i in range(4000))
    assert set(folds) == {-1, 0, 1, 2, 3, 4} and min(folds[f] for f in range(5)) > 500


def synthetic_index() -> dict:
    """Reports in the shape build_mp_candidates.known_index returns: {elements: [(fractions, Tc, source)]}."""
    def report(formula: str, tc: float, source: str):
        return (D.parse_formula(formula), tc, source)

    reports = [
        report("MgB2", 39.0, "stanev"), report("Mg0.5B", 39.0, "3dsc"), report("MgB2", 38.0, "mtg"),   # one compound
        report("YBa2Cu3O7", 92.0, "stanev"), report("YBa2Cu2.9Zn0.1O7", 80.0, "stanev"),               # near-twins
        report("NaCl", 0.0, "stanev"),                                                                 # negative
        report("FeSe", 8.0, "stanev"), report("FeSe", 0.0, "supermat:x"),                               # mixed
        report("LaNiO3", 0.0, "hosono:1"),                                                             # Hosono only
    ]
    index: dict = {}
    for fractions, tc, source in reports:
        index.setdefault(frozenset(fractions), []).append((fractions, tc, source))
    return index


def test_build_composition_rows_labels_and_groups():
    rows = {row["composition"]: row for row in build.build_composition_rows(synthetic_index(), family=lambda e: "x")}
    assert len(rows) == 6
    mgb2 = rows["B:0.667;Mg:0.333"]
    assert mgb2["label"] == 1 and mgb2["tc_k"] == 38.5 and mgb2["n_reports"] == 3     # 39 counted once, then 38
    assert mgb2["lineages"] == "supercon" and mgb2["tc_max_k"] == 39.0
    assert rows["Cl:0.500;Na:0.500"]["label"] == 0 and rows["Cl:0.500;Na:0.500"]["tc_k"] == ""
    fese = rows["Fe:0.500;Se:0.500"]
    assert fese["label"] == 1 and fese["mixed_reports"] == 1 and fese["lineages"] == "supercon;supermat"
    hosono = rows["La:0.200;Ni:0.200;O:0.600"]
    assert hosono["label"] == "" and hosono["hosono_only_zero"] == 1
    ybco = [row for key, row in rows.items() if key.startswith("Ba:")]
    assert len(ybco) == 2 and len({row["group"] for row in ybco}) == 1 and ybco[0]["group_rows"] == 2
    assert len({(row["split"], row["fold"]) for row in ybco}) == 1                    # twins share a side
    assert build.cross_group_near_pairs(list(rows.values())) == 0


def test_lineage_collapses_the_supercon_copies():
    assert [build.lineage(s) for s in ("stanev", "3dsc", "mtg", "hosono:4", "supermat:9", "audit:1")] == \
        ["supercon", "supercon", "supercon", "hosono", "supermat", "audit"]
