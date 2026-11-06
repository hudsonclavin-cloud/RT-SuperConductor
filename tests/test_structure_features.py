import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from structure_features import (  # noqa: E402
    CENTROSYMMETRIC, POINT_GROUPS, compute, debye_temperature, load_feature_table, structure_row,
)

TABLE = ROOT / "data/candidates/mp-structure-features.csv.gz"


def test_debye_temperature_matches_known_metals():
    # Anderson's method with handbook moduli; measured values 428 K and 343 K.
    assert debye_temperature(76, 26, 2.70, 16.6) == pytest.approx(428, rel=0.06)
    assert debye_temperature(140, 48, 8.96, 11.8) == pytest.approx(343, rel=0.03)
    assert debye_temperature(float("nan"), 26, 2.7, 16.6) is None


def test_point_group_tables_are_consistent():
    assert len(POINT_GROUPS) == 32 and CENTROSYMMETRIC <= set(POINT_GROUPS) and len(CENTROSYMMETRIC) == 11


def _doc(mp_id, species, coords, a):
    from pymatgen.core import Lattice, Structure

    return {"mp_id": mp_id, "structure": Structure(Lattice.cubic(a), species, coords).as_dict()}


def test_structure_row_for_rock_salt_and_bcc():
    nacl = structure_row(_doc("nacl", ["Na"] * 4 + ["Cl"] * 4,
                              [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                               [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]], 5.64))
    assert nacl["sym_pg_m-3m"] == 1.0 and nacl["sym_centering_F"] == 1.0 and nacl["sym_centrosymmetric"] == 1.0
    assert nacl["geo_coordination_mean"] == pytest.approx(6.0)
    assert nacl["geo_unlike_contact_share"] == pytest.approx(1.0)
    assert nacl["conn_dimensionality"] == 3
    nb = structure_row(_doc("nb", ["Nb", "Nb"], [[0, 0, 0], [.5, .5, .5]], 3.30))
    # bcc: 8 nearest plus 6 next-nearest neighbours lie within the 1.2x shell (second shell is 15.5% farther)
    assert nb["sym_centering_I"] == 1.0 and nb["geo_coordination_mean"] == pytest.approx(14.0)
    assert nb["geo_metal_metal_contact_share"] == pytest.approx(1.0)


def test_compute_writes_aligned_table_with_stiffness(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = [_doc("mp-1", ["Nb", "Nb"], [[0, 0, 0], [.5, .5, .5]], 3.30),
            _doc("mp-2", ["Cu"] * 4, [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], 3.61)]
    with gzip.open(raw / "structures.jsonl.gz", "wt") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    (raw / "elasticity.csv").write_text("mp_id,bulk_modulus_vrh_gpa,shear_modulus_vrh_gpa\nmp-2,140,48\n")
    report = compute(raw, tmp_path / "out.csv.gz", workers=1)
    assert report["materials"] == 2 and report["with_elastic_data"] == 1
    X, names = load_feature_table(tmp_path / "out.csv.gz", ["mp-2", "mp-1", "missing"])
    debye = names.index("stiff_debye_temperature_k")
    assert X[0, debye] == pytest.approx(343, rel=0.05)
    assert X[1, debye] != X[1, debye]  # NaN: no elastic data
    assert all(v != v for v in X[2])    # unknown material -> all NaN


@pytest.mark.skipif(not TABLE.is_file(), reason="structure table not present")
def test_committed_structure_table_covers_the_snapshot():
    with gzip.open(TABLE, "rt") as handle:
        header = handle.readline().strip().split(",")
        rows = sum(1 for _ in handle)
    assert rows == 30358 and header[0] == "mp_id" and "stiff_debye_temperature_k" in header
