import numpy as np
import pandas as pd

from rtsc2 import data


def test_training_labels_follow_evidence_tiers():
    frame = pd.DataFrame({
        "known_status": ["superconductor", "tested_non_superconductor", "tested_non_superconductor",
                         "untested_by_formula", "tested_non_superconductor"],
        "negative_training_use": [np.nan, "use", "exclude", np.nan, np.nan]})
    assert data.training_labels(frame).tolist()[:2] == [1.0, 0.0]
    # an excluded negative and an untested material carry no label; a pre-tier negative is usable
    assert np.isnan(data.training_labels(frame)[2]) and np.isnan(data.training_labels(frame)[3])
    assert data.training_labels(frame)[4] == 0.0


def test_labels_match_legacy_policy_on_the_real_table():
    if not data.CANDIDATES.is_file():
        return
    frame = pd.read_csv(data.CANDIDATES, dtype=str, keep_default_na=False).replace("", np.nan)
    label = data.training_labels(frame)
    assert int(np.nansum(label)) == int((frame["known_status"] == "superconductor").sum())
    excluded = (frame["negative_training_use"] == "exclude").to_numpy()
    assert np.isnan(label[excluded]).all()


def test_element_tokens_are_fractions_sorted_by_share():
    import pytest

    pytest.importorskip("pymatgen.core")
    index, fraction = data.element_tokens(["Nb3Sn", "Pb"])
    assert index[0, :2].tolist() == [41, 50] and np.allclose(fraction[0, :2], [0.75, 0.25])
    assert index[1, 0] == 82 and fraction[1].sum() == 1.0 and index[1, 1] == 0
