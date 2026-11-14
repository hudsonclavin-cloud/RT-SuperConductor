"""The stability check trains on the right rows and scores both variants on the same stable rows."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
S = pytest.importorskip("lambda_stability_check")


def toy_rows(n=60):
    """Stable rows follow lambda = 0.3 + feature; unstable rows carry wild lambdas that should only hurt."""
    rng = np.random.default_rng(1)
    elements = ["Nb", "Ta", "V", "Mo", "W", "Ti", "Zr", "Hf", "Re", "Os", "Ru", "Rh", "Pd", "Pt", "Ir", "Al", "Ga",
                "In", "Sn", "Pb", "Si", "Ge", "B", "C", "N"]
    pairs = [(a, b) for k, a in enumerate(elements) for b in elements[k + 1:]]
    rows = []
    for i in range(n):
        x = float(rng.random())
        stable = i % 3 != 0
        lam = 0.3 + x if stable else 5.0 + 10.0 * float(rng.random())
        row = {"formula": "".join(pairs[i]), "lambda": lam, "omega_log_k": 150.0 + 50.0 * x,
               "allen_dynes_tc_mu0p1_k": S.P.allen_dynes_tc_k(lam, 150.0 + 50.0 * x),
               "source_stability": "stable" if stable else "unstable", "magpie__x": x, "shell__y": 1.0,
               "lewis_dot_proxy_avg_outer_electrons": 2.0, "lewis_dot_proxy_total_electrons_per_reduced_formula": 4.0}
        row.update({name: 0.0 for name in S.P.BOND_FEATURE_NAMES})
        rows.append(row)
    return rows


def test_dropping_wild_unstable_rows_helps_on_stable_rows():
    report = S.compare(toy_rows(), n_splits=3, trees=30, resamples=50)
    assert report["rows"] == 60 and report["stable_rows"] == 40
    assert report["largest_lambda_unstable"] > 5 > report["largest_lambda_stable"]
    assert report["stable rows only"]["lambda_mae"] < report["all rows"]["lambda_mae"]
    gap = report["stable_only_minus_all"]["lambda_mae"]
    assert gap["low"] <= gap["median"] <= gap["high"] < 0


def test_feature_names_match_the_production_inputs():
    names = S.feature_names(toy_rows(3)[0])
    assert names[:2] == ["magpie__x", "shell__y"] and names[-len(S.P.BOND_FEATURE_NAMES):] == S.P.BOND_FEATURE_NAMES
