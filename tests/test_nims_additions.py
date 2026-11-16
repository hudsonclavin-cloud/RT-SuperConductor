"""Cycle 4: where added NIMS rows may go, and when a training round may fit them."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_nims_additions as A  # noqa: E402
import run_learning_cycle as L  # noqa: E402


def row(composition: str, group: str, split: str, fold: int) -> dict:
    import build_scratchnet_data as B
    import scratchnet_common as C

    return {"composition": composition, "major_group": B.major_group(C.fractions_of(composition)), "group": group,
            "split": split, "fold": fold}


TABLE = [row("Nb:0.750;Sn:0.250", "Nb-Sn", "dev", 2),
         row("Al:0.250;V:0.750", "Al-V", "lockbox", -1),
         row("La:0.500;Ni:0.500", "La-Ni", "dev", 4),
         row("Ga:0.500;La:0.500", "Ga-La", "dev", 1)]


def placed(new: list[str]) -> dict:
    return {entry["composition"]: entry for entry in A.place(TABLE, new)}


def test_a_new_row_takes_the_fold_of_the_group_it_joins():
    out = placed(["Nb:0.800;Sn:0.200"])["Nb:0.800;Sn:0.200"]
    assert out["placed"] == "joins a development group" and out["fold"] == 2 and out["group"] == "Nb-Sn"


def test_a_row_tied_to_a_locked_group_is_left_out():
    out = placed(["Al:0.300;V:0.700"])["Al:0.300;V:0.700"]
    assert out["fold"] is None and "locked" in out["placed"]


def test_a_minor_dopant_still_ties_a_row_to_the_locked_group():
    # 2% gallium is below the major-element threshold, so the major group is still Al-V
    out = placed(["Al:0.240;Ga:0.020;V:0.740"])["Al:0.240;Ga:0.020;V:0.740"]
    assert out["fold"] is None


def test_a_near_twin_ties_rows_across_major_groups():
    # 5% Ga makes the major group Ga-Nb-Sn, but the row is within 0.02 of a chain that reaches the Nb-Sn group
    new = ["Ga:0.050;Nb:0.730;Sn:0.220", "Ga:0.035;Nb:0.740;Sn:0.225", "Ga:0.018;Nb:0.745;Sn:0.237"]
    out = placed(new)
    assert {entry["fold"] for entry in out.values()} == {2}
    assert all(entry["placed"] == "joins a development group" for entry in out.values())


def test_a_row_tied_to_two_folds_is_left_out():
    # La-Ni (fold 4) and Ga-La (fold 1) are bridged by a chain of near twins through an added ternary
    new = ["Ga:0.250;La:0.500;Ni:0.250"]
    table = TABLE + [row("Ga:0.240;La:0.500;Ni:0.260", "La-Ni", "dev", 4),
                     row("Ga:0.260;La:0.500;Ni:0.240", "Ga-La", "dev", 1)]
    out = A.place(table, new)[0]
    assert out["fold"] is None and "more than one fold" in out["placed"]


def test_an_unrelated_row_forms_a_new_group_with_a_stable_fold():
    first = placed(["Mo:0.500;Re:0.500"])["Mo:0.500;Re:0.500"]
    again = placed(["Mo:0.500;Re:0.500", "Mo:0.510;Re:0.490"])
    assert first["placed"] == "new group" and first["fold"] in range(5)
    assert {entry["fold"] for entry in again.values()} == {first["fold"]}
    assert {entry["group"] for entry in again.values()} == {"Mo-Re"}


def extra_block() -> dict:
    return {"fold": np.array([0, 1, 1, 2]), "tied": [{"a"}, {"b"}, {"c", "d"}, {"e"}]}


def test_a_round_never_fits_the_held_out_fold_or_its_early_stopping_groups():
    extra = extra_block()
    assert L.added_rows_for(extra, held_fold=1, validation_groups=set()).tolist() == [0, 3]
    assert L.added_rows_for(extra, held_fold=0, validation_groups={"d"}).tolist() == [1, 3]
    assert L.added_rows_for(extra, held_fold=4, validation_groups={"a", "e"}).tolist() == [1, 2]


def test_duplicated_rows_keep_their_own_fold_and_group():
    import scratchnet_common as C

    ds = C.load_dataset() if (C.ROOT / "data/processed/scratchnet/compositions.csv.gz").is_file() else None
    if ds is None:
        import pytest
        pytest.skip("the composition table is not built here")
    X = np.zeros((len(ds.rows), 2))
    extra = L.duplicate_rows(ds, X, 50, seed=41)
    index = {r["composition"]: i for i, r in enumerate(ds.rows)}
    rows = np.array([index[key] for key in extra["keys"]])
    assert len(set(extra["keys"])) == 50 and (ds.split[rows] == "dev").all()
    assert (extra["fold"] == ds.fold[rows]).all()
    assert [next(iter(t)) for t in extra["tied"]] == ds.arrays.group[rows].tolist()
    assert extra["score"] is False and np.isfinite(extra["label"]).all()
    again = L.duplicate_rows(ds, X, 50, seed=41)
    assert again["keys"] == extra["keys"]


def test_the_score_on_added_rows_is_a_paired_difference():
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 60, dtype=float)
    better = y + rng.normal(scale=0.4, size=120)
    worse = y + rng.normal(scale=2.0, size=120)
    extra = {"label": y, "p_with": better, "p_without": worse, "cuprate": np.arange(120) < 20,
             "group": np.array([f"g{i // 4}" for i in range(120)])}
    out = L.score_added_rows(extra, resamples=200)
    assert out["all"]["rows"] == 120 and out["not cuprates"]["rows"] == 100
    assert out["all"]["difference"] > 0 and out["all"]["difference_interval_95"]["low"] > 0
    assert abs(out["all"]["difference"] - (out["all"]["auc_with_added_rows"] - out["all"]["auc_without"])) < 1e-12
