import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eph_gpu as E  # noqa: E402


def test_split_depends_only_on_the_element_set():
    assert E.split_of("Nb:0.667;Zr:0.333") == E.split_of("Zr:0.5;Nb:0.5")
    shares = [E.split_of(f"{a}:0.5;{b}:0.5") for a, b in itertools.combinations(["H", "Li", "B", "C", "N", "Mg", "Al", "Si",
                                                                                 "Ca", "Sc", "Ti", "V", "Nb", "Mo"], 2)]
    assert set(shares) == {"fit", "valid", "test"}


def test_auc_matches_pair_counting_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60).astype(float)
    s = rng.integers(0, 6, 60).astype(float)
    pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a, ya in zip(s, y) for b, yb in zip(s, y) if ya == 1 and yb == 0]
    assert abs(E.auc(y, s) - np.mean(pairs)) < 1e-12
    assert np.allclose(E.ranks(np.array([3.0, 1.0, 3.0, 2.0])), [2.5, 0.0, 2.5, 1.0])


def test_settings_are_drawn_without_repeats():
    grid = E.configs(0)
    assert len(grid) == int(np.prod([len(v) for v in E.GRID.values()]))
    assert len({tuple(sorted(c.items())) for c in grid}) == len(grid)
    assert grid == E.configs(0) and grid != E.configs(1)


def test_a_sleep_is_charged_as_one_usual_epoch():
    said = []
    clock = E.Clock(spent=100.0, budget=3600.0, say=said.append)
    for _ in range(6):
        clock.charge(10.0, epoch=True)
    assert clock.charge(7 * 3600.0, epoch=True) == 10.0          # a 7 h gap mid-member
    assert abs(clock.spent - 170.0) < 1e-9 and said
    assert clock.charge(50.0, epoch=True) == 50.0                # slower epochs are charged in full
