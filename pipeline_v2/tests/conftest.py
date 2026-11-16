import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

# Without the optional ML dependencies, skip this folder instead of failing collection.
if any(importlib.util.find_spec(name) is None for name in ("numpy", "pandas", "sklearn", "joblib")):
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture
def toy():
    """300 rows, 60 element systems; the label depends on feature 0 and on element 5."""
    import numpy as np
    import pandas as pd

    from rtsc2.data import MAX_ELEMENTS, Dataset

    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 6))
    X[rng.random((n, 6)) < 0.05] = np.nan
    index = np.zeros((n, MAX_ELEMENTS), dtype=np.int64)
    fraction = np.zeros((n, MAX_ELEMENTS), dtype=np.float32)
    index[:, 0], index[:, 1] = rng.integers(1, 10, n), rng.integers(10, 20, n)
    fraction[:, 0], fraction[:, 1] = 0.5, 0.5
    signal = np.nan_to_num(X[:, 0]) + 1.5 * (index[:, 0] == 5) + rng.normal(scale=0.5, size=n)
    label = (signal > 0.3).astype(float)
    tc = np.where(label == 1, np.exp(rng.normal(1.0, 1.0, n)), np.nan)
    groups = np.asarray([f"g{i % 60}" for i in range(n)])
    frame = pd.DataFrame({"mp_id": [f"mp-{i}" for i in range(n)]})
    return Dataset(frame["mp_id"].to_numpy(), np.asarray(["X"] * n), X, tuple(f"f{i}" for i in range(6)),
                   index, fraction, label, tc, groups, frame)
