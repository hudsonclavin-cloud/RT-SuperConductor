"""The learning-cycle harness: table comparison, layout guard, made-metals order and its comparison."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import run_learning_cycle as L  # noqa: E402


def state(label="0", tc="", split="dev", fold="1"):
    return {"label": label, "tc_k": tc, "split": split, "fold": fold, "group": "g"}


def test_compare_tables_lists_changed_labels_and_layout_problems():
    before = {"A": state(), "B": state(), "C": state(split="lockbox", fold="-1"), "D": state()}
    after = {"A": state(label="1", tc="6.2"), "B": state(), "C": state(split="lockbox", fold="-1"),
             "D": state(fold="2"), "E": state()}
    out = L.compare_tables(before, after)
    assert [c["composition"] for c in out["changed"]] == ["A"]
    assert out["changed"][0]["label_before"] == "0" and out["changed"][0]["tc_k_after"] == "6.2"
    assert out["moved_between_splits"] == ["D"] and out["added"] == ["E"] and out["removed"] == []


def test_layout_guard_refuses_moved_added_or_removed_rows():
    with pytest.raises(SystemExit):
        L.check_unchanged_layout({"moved_between_splits": ["D"], "added": [], "removed": [], "changed": []})
    with pytest.raises(SystemExit):
        L.check_unchanged_layout({"moved_between_splits": [], "added": ["E"], "removed": [], "changed": []})
    L.check_unchanged_layout({"moved_between_splits": [], "added": [], "removed": [], "changed": [{"x": 1}]})


def test_composition_key_ignores_how_the_formula_is_written():
    assert L.composition_key("PtPb3Bi") == L.composition_key("BiPtPb3")
    assert L.composition_key("Zr6Ga2Ni") != L.composition_key("Zr6Ga2Co")


def made_row(formula, known="", metal="True", theoretical="False"):
    return {"mp_id": "mp-" + formula, "formula": formula, "is_metal": metal, "theoretical": theoretical,
            "known_match": known}


def test_made_metals_order_keeps_untested_made_metals_ordered_by_rounded_probability():
    rows = [made_row("Zr6Ga2Ni"), made_row("NbS", known="exact"), made_row("CuZr2"), made_row("BaPb3"),
            made_row("TiAlPt", theoretical="True"), made_row("Ba3Pb5"), made_row("MoS2", metal="False")]
    comps = [L.C.D_parse(r["formula"]) for r in rows]
    p = np.array([0.9551, 0.99, 0.99, 0.9642, 0.99, 0.9604, 0.99])
    tc = np.array([3.0, 9.0, 9.0, 3.3, 9.0, 4.6, 9.0])
    order = L.made_metals_order(rows, comps, p, tc)
    # NbS is known, CuZr2 has copper, TiAlPt is only calculated, MoS2 is not a metal.
    assert [r["formula"] for r in order] == ["BaPb3", "Ba3Pb5", "Zr6Ga2Ni"]
    assert [r["rank"] for r in order] == [1, 2, 3]


def test_made_metals_order_breaks_probability_ties_by_expected_tc():
    rows = [made_row("BaPb3"), made_row("Ba3Pb5")]
    comps = [L.C.D_parse(r["formula"]) for r in rows]
    order = L.made_metals_order(rows, comps, np.array([0.9601, 0.9604]), np.array([3.3, 4.6]))
    assert [r["formula"] for r in order] == ["Ba3Pb5", "BaPb3"]      # same rounded p; higher p x Tc first


def test_compare_orders_reports_overlap_moves_and_watched_compounds():
    previous = [{"formula": f, "rank": i} for i, f in enumerate(["BaPb3", "Zr6Ga2Ni", "BiPtPb3", "LuNiC2"], 1)]
    now = [{"formula": f, "rank": i} for i, f in enumerate(["Zr6Ga2Ni", "BaPb3", "LuNiC2", "SbTe"], 1)]
    out = L.compare_orders(previous, now, watch=("Zr6Ga2Ni", "PtPb3Bi"))
    assert out["compounds_in_both"] == 3
    assert out["head_overlap"]["top_10"] == 3
    assert out["watched"]["Zr6Ga2Ni"] == {"before": 2, "after": 1}
    assert out["watched"]["PtPb3Bi"] == {"before": 3, "after": None}
    assert out["entered"] == [L.composition_key("SbTe")]
    assert out["left"] == [L.composition_key("PtPb3Bi")]


def test_aligned_puts_saved_predictions_in_the_requested_order():
    saved = {"A": (0.1, 1.0), "B": (0.2, 2.0)}
    p, tc = L.aligned(saved, ["B", "C", "A"])
    assert p[0] == 0.2 and np.isnan(p[1]) and tc[2] == 1.0


def test_evidence_rows_name_existing_columns():
    """Every column the cycle documentation tells a reader to fill is in the evidence file's header."""
    with open(ROOT / "data/training/negative_evidence.csv", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    for column in ("formula", "publication_doi", "page_table_or_figure", "superconductivity_observed", "tc_k",
                   "reviewer", "date"):
        assert column in header


def tiny_dataset():
    n = 8
    split = np.array(["dev", "dev", "dev", "dev", "lockbox", "train_only", "dev", "dev"])
    family = np.array(["Other", "Cuprates", "Other", "Other", "Other", "Other", "Other", "Oxides"])
    label = np.array([0, 0, 1, 0, 0, 0, np.nan, 0], float)
    tc = np.array([np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    arrays = L.T.Arrays(np.zeros((n, 1), int), np.zeros((n, 1)), np.ones((n, 1), bool), label, tc,
                        np.array([f"g{i}" for i in range(n)]))
    rows = [{"composition": f"X:{i}"} for i in range(n)]
    return L.C.Dataset(rows, None, arrays, split, np.zeros(n, int), family, np.zeros((n, 2)), "sha")


def test_placebo_flips_only_development_non_cuprate_negatives_in_memory():
    ds = tiny_dataset()
    before = ds.arrays.label.copy()
    flipped = L.placebo(ds, 3, seed=0)
    eligible = {"X:0", "X:3", "X:7"}                # dev, label 0, not a cuprate
    assert set(flipped) == eligible
    rows = [int(key.split(":")[1]) for key in flipped]
    assert np.all(ds.arrays.label[rows] == 1) and np.all(ds.arrays.tc_k[rows] == 0.9)
    untouched = [i for i in range(8) if i not in rows]
    assert np.array_equal(ds.arrays.label[untouched], before[untouched], equal_nan=True)


def test_placebo_refuses_more_flips_than_eligible_rows():
    with pytest.raises(ValueError):
        L.placebo(tiny_dataset(), 4, seed=0)
