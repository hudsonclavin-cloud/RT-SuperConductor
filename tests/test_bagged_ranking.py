"""The bagged ranking: group resampling and the per-model rank order."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import run_bagged_ranking as B  # noqa: E402


def test_bag_rows_keeps_whole_groups_and_is_deterministic():
    groups = np.array(["a", "a", "b", "c", "c", "c", "d"])
    fit = np.array([0, 1, 2, 3, 4, 5, 6])
    bag = B.bag_rows(groups, fit, seed=3)
    assert np.array_equal(bag, B.bag_rows(groups, fit, seed=3))
    sizes = {"a": 2, "b": 1, "c": 3, "d": 1}
    for name, size in sizes.items():           # a group is in the bag as many whole copies as it was drawn
        members = np.flatnonzero(groups == name)
        counts = np.array([(bag == row).sum() for row in members])
        assert len(set(counts)) == 1, name
    assert len(set(np.unique(bag)) - set(fit.tolist())) == 0


def test_bag_rows_only_uses_the_fit_rows_given():
    groups = np.array(["a", "b", "c", "d"])
    bag = B.bag_rows(groups, np.array([1, 3]), seed=0)
    assert set(bag.tolist()) <= {1, 3}


def test_bag_seeds_differ_by_bag_and_fold():
    assert len({B.bag_seed(b, f) for b in range(8) for f in range(5)}) == 40


def test_member_ranks_round_probability_then_break_ties_by_expected_tc():
    p = np.array([0.9551, 0.9604, 0.9601, 0.5])
    tc = np.array([3.0, 4.6, 3.3, 9.0])
    assert B.member_ranks(p, tc).tolist() == [3, 1, 2, 4]
    p2, tc2 = np.array([0.9601, 0.9604]), np.array([3.3, 4.6])
    assert B.member_ranks(p2, tc2).tolist() == [2, 1]      # same rounded p: higher p x Tc first
