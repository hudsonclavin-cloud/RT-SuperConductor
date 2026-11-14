"""The diagnosis experiments: how training rows are thinned and what boosting is given as raw inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
diagnose = pytest.importorskip("scratchnet_diagnose")
C = pytest.importorskip("scratchnet_common")
build = pytest.importorskip("build_scratchnet_data")

from rt_superconductor.scratchnet import data as D  # noqa: E402

from test_scratchnet_rounds import dataset  # noqa: E402,F401  (the small made-up table)


def test_subsample_keeps_groups_whole_and_smaller_shares_inside_larger_ones(dataset):  # noqa: F811
    rows = np.flatnonzero(dataset.split != "lockbox")
    kept = {share: diagnose.subsample(dataset, rows, share) for share in (0.125, 0.25, 0.5, 1.0)}
    assert kept[1.0] is rows
    assert set(kept[0.125]) <= set(kept[0.25]) <= set(kept[0.5]) <= set(kept[1.0])
    assert len(kept[0.5]) < len(rows)
    for share in (0.25, 0.5):
        ordinary = [row for row in kept[share] if dataset.split[row] != "train_only"]
        for group in set(dataset.arrays.group[ordinary]):
            members = np.flatnonzero((dataset.arrays.group == group) & (dataset.split != "lockbox"))
            assert set(members) <= set(kept[share])                       # a group is in or out as a whole
    # The training-only group is thinned row by row, so some but not all of it is kept at one half.
    big = np.flatnonzero(dataset.split == "train_only")
    half = diagnose.subsample(dataset, big, 0.5)
    assert 0 < len(half) < len(big)
    assert diagnose.subsample(dataset, rows, 0.5).tolist() == kept[0.5].tolist()        # the same every time


def test_element_blocks_lay_out_amount_and_properties_most_abundant_first(dataset):  # noqa: F811
    blocks = diagnose.element_blocks(dataset)
    features = dataset.table.features.shape[1]
    assert blocks.shape == (len(dataset.rows), 8 * (1 + features))
    first = blocks[0].reshape(8, 1 + features)
    assert first[0, 0] >= first[1, 0] > 0                                 # amounts, largest first
    np.testing.assert_allclose(first[0, 1:], dataset.table.features[dataset.arrays.index[0, 0]])
    assert np.isnan(first[2:]).all()                                      # a two-element material: six empty slots
    np.testing.assert_allclose(np.nansum(blocks.reshape(len(dataset.rows), 8, -1)[:, :, 0], axis=1), 1.0, atol=2e-3)


def test_boosting_settings_come_from_the_round_two_selection(tmp_path):
    chosen = {"classifier": {"max_leaf_nodes": 31}, "regressor": {"max_leaf_nodes": 63}}
    (tmp_path / "scratchnet-round2.json").write_text(json.dumps({"selection": {"boosting": chosen}}))
    assert diagnose.boosting_settings(tmp_path) == {"classifier_settings": {"max_leaf_nodes": 31},
                                                    "regressor_settings": {"max_leaf_nodes": 63}}


def test_ordering_table_separates_the_likeliest_from_the_most_valuable():
    """Forty sure low-Tc rows and forty less sure high-Tc rows: the two orders pick different halves."""
    n = 80
    label = np.ones(n)
    label[40::4] = 0.0                                       # a quarter of the second half do not superconduct
    tc = np.where(np.arange(n) < 40, 2.0, 40.0)
    probability = np.where(np.arange(n) < 40, 0.95, 0.7) - 0.001 * np.arange(n) / n
    tc_if = np.where(np.arange(n) < 40, 2.0, 40.0)
    groups = np.arange(n) // 2
    table = diagnose.ordering_table(label, tc, groups, probability, tc_if, shares=(0.25,), resamples=200)
    entry = table["shares"]["0.25"]
    assert table["rows"] == 80 and entry["picks"] == 20 and entry["picks_in_common"] == 0
    assert entry["by_probability"] == {"hit_rate": 1.0, "mean_tc_k": 2.0, "median_tc_of_hits_k": 2.0}
    assert entry["by_expected_tc"]["hit_rate"] == 0.75 and entry["by_expected_tc"]["mean_tc_k"] == 30.0
    assert entry["hit_rate_difference"]["value"] == 0.25 and entry["hit_rate_difference"]["low"] > 0
    assert entry["mean_tc_difference_k"]["value"] == -28.0 and entry["mean_tc_difference_k"]["high"] < 0


def test_has_element_reads_the_right_column(dataset):
    rows = np.arange(len(dataset.rows))
    for symbol in ("Cu", "Nb", "O"):
        expected = np.array([symbol in row["system"].split("-") for row in dataset.rows])
        assert (C.has_element(dataset, rows, symbol) == expected).all() and expected.any()


def test_orderings_command_leaves_out_copper_and_the_two_families(dataset, tmp_path, monkeypatch):  # noqa: F811
    # The shared toy table files every system but Cu-O under "Other", so there the copper rule, the iron family
    # and the Oxides branch are never exercised. Give each system its real family (Fe-Se and Fe-As are Ferrites;
    # La-O, Al-O and Si-O are Oxides) and add Cu-Nb, a copper system outside the Cuprates.
    rows = [dict(row, family=C.family_of(set(row["system"].split("-")))) for row in dataset.rows]
    for step in range(12):
        x = 0.2 + 0.05 * step
        rows.append({"composition": D.composition_key({"Cu": x, "Nb": 1 - x}), "system": "Cu-Nb", "family": "Other",
                     "major_group": "Cu-Nb", "group": "Cu-Nb", "group_rows": 12, "n_elements": 2, "label": 1,
                     "tc_k": round(4.0 + 30.0 * x, 3), "split": "dev", "fold": 1})
    D.write_table(tmp_path / "compositions.csv.gz", rows, build.COMPOSITION_FIELDS)
    table = C.load_dataset(tmp_path / "compositions.csv.gz", report=None)
    family = {row["system"]: row["family"] for row in table.rows}
    assert family["Fe-Se"] == family["Fe-As"] == "Ferrites" and family["La-O"] == "Oxides"
    assert family["Cu-O"] == "Cuprates" and family["Cu-Nb"] == "Other"
    monkeypatch.setattr(C, "WORK", tmp_path)
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: table)
    dev = np.flatnonzero(table.split == "dev")
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "toy-oof.npz", rows=dev, distance=np.full(len(dev), 0.2),
             **{"p::boosting": rng.random(len(dev)), "tc::boosting": 1.0 + 9.0 * rng.random(len(dev))})
    report = diagnose.run_orderings("toy", tmp_path, resamples=50)
    # Out: the cuprate, both iron systems, and copper outside the cuprates. In: the other development systems,
    # less the three unlabelled Si-O rows.
    dropped = {"Cu-O", "Fe-Se", "Fe-As", "Cu-Nb"}
    kept = [i for i in dev if table.rows[i]["system"] not in dropped and table.rows[i]["label"] != ""]
    assert len(kept) == 10 * 12 - 3
    bands = report["bands"]
    assert bands["all distances"]["rows"] == len(kept)
    assert bands["0.1 or more from any training composition"]["rows"] == len(kept)
    assert bands["0.3 or more from any training composition"]["rows"] == 0
    assert json.loads((tmp_path / "scratchnet-orderings.json").read_text())["name"] == "toy"
