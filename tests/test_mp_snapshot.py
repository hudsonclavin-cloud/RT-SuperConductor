import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_mp_snapshot import COLUMNS, fetch_all, flatten, page_url, write_snapshot  # noqa: E402
from build_mp_candidates import build, family, fractions, match_known  # noqa: E402

SNAPSHOT = ROOT / "data/candidates/mp-stable-summary.csv.gz"
MANIFEST = ROOT / "data/candidates/mp-stable-summary.manifest.json"


def _doc(i: int) -> dict:
    return {"material_id": f"mp-{i}", "formula_pretty": "Nb3Sn", "elements": ["Nb", "Sn"], "nelements": 2,
            "nsites": 8, "energy_above_hull": 0.0, "formation_energy_per_atom": -0.3, "is_stable": True,
            "theoretical": False, "symmetry": {"crystal_system": "Cubic", "number": 223, "symbol": "Pm-3n"},
            "volume": 148.0, "density": 8.9, "is_metal": True, "band_gap": 0.0, "is_magnetic": False,
            "ordering": "NM", "total_magnetization": 0.0, "total_magnetization_normalized_formula_units": 0.0,
            "num_magnetic_sites": 0, "database_IDs": {"icsd": ["icsd-1"]}}


def test_query_matches_the_pool_scope():
    url = page_url(0)
    assert "energy_above_hull_max=0.025" in url and "nsites_max=20" in url and "deprecated=false" in url


def test_paging_collects_every_page_and_rejects_duplicates():
    pages = {0: [_doc(0), _doc(1)], 2: [_doc(2)]}

    def fetch(url):
        skip = int(url.split("_skip=")[1])
        return {"meta": {"total_doc": 3, "api_version": "test"}, "data": pages.get(skip, [])}

    rows, meta = fetch_all(fetch, progress=False)
    assert [r["mp_id"] for r in rows] == ["mp-0", "mp-1", "mp-2"]
    assert set(rows[0]) == set(COLUMNS)

    def dup(url):
        return {"meta": {"total_doc": 2}, "data": [_doc(0)]}

    with pytest.raises(RuntimeError):
        fetch_all(dup, progress=False)


def test_snapshot_is_byte_reproducible(tmp_path):
    rows = [flatten(_doc(0))]
    write_snapshot(rows, {"api_version": "t", "time_stamp": "x"}, tmp_path / "a.csv.gz", tmp_path / "a.json")
    write_snapshot(rows, {"api_version": "t", "time_stamp": "x"}, tmp_path / "b.csv.gz", tmp_path / "b.json")
    assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "b.csv.gz").read_bytes()


@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="snapshot not present")
def test_committed_snapshot_matches_its_manifest():
    import hashlib

    manifest = json.loads(MANIFEST.read_text())
    raw = gzip.open(SNAPSHOT).read()
    assert hashlib.sha256(raw).hexdigest() == manifest["csv_sha256_uncompressed"]
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert len(rows) == manifest["materials"] == 30358
    assert len({r["mp_id"] for r in rows}) == len(rows)


def test_families_by_element_rule():
    assert family({"Y", "Ba", "Cu", "O"}) == "Cuprates"
    assert family({"La", "Fe", "As", "O"}) == "Ferrites"
    assert family({"Sr", "Ti", "O"}) == "Oxides"
    assert family({"Nb", "Sn"}) == "Other"


def test_known_matching_by_atomic_fraction():
    index = {}
    for formula, tc in (("B2Mg0.98", 38.4), ("Nb3Sn", 17.8), ("V1B2", 0.0)):
        frac = fractions(formula)
        index.setdefault(frozenset(frac), []).append((frac, tc, "test"))
    assert match_known("MgB2", index)[0] == "near"          # 2% Mg deficiency
    assert match_known("Nb6Sn2", index)[0] == "exact"       # same ratio, different scale
    assert match_known("VB2", index) == ("exact", [(0.0, "test")])
    assert match_known("Nb2Sn", index) == ("", [])          # different compound


def test_build_labels_known_and_untested(tmp_path):
    snap = tmp_path / "snap.csv.gz"
    rows = [flatten(_doc(0)), {**flatten(_doc(1)), "mp_id": "mp-x", "formula": "Nb2Sn", "elements": "Nb Sn"}]
    write_snapshot(rows, {}, snap, tmp_path / "m.json")
    stanev = tmp_path / "stanev.csv"
    stanev.write_text("name,Tc\nNb3Sn,17.8\n")
    report = build(snap, tmp_path / "out.csv.gz", threedsc=tmp_path / "absent.csv", stanev=stanev)
    out = {r["mp_id"]: r for r in csv.DictReader(gzip.open(tmp_path / "out.csv.gz", "rt"))}
    assert out["mp-0"]["known_status"] == "superconductor" and out["mp-0"]["known_tc_max_k"] == "17.8"
    assert out["mp-x"]["known_status"] == "untested_by_formula"
    assert report["counts"]["family_Other"] == 2


def test_reference_tables_are_found_where_the_downloader_writes_them(tmp_path):
    """CI has the downloader's layout, not a clone of the 3DSC repository."""
    from build_mp_candidates import _first_existing

    downloaded = tmp_path / "3dsc/MP/3DSC_MP.csv"
    cloned = tmp_path / "3dsc/repo/superconductors_3D/data/final/MP/3DSC_MP.csv"
    assert _first_existing(downloaded, cloned) == downloaded   # neither exists: name the downloader's path
    cloned.parent.mkdir(parents=True)
    cloned.write_text("x\n")
    assert _first_existing(downloaded, cloned) == cloned
    downloaded.parent.mkdir(parents=True)
    downloaded.write_text("x\n")
    assert _first_existing(downloaded, cloned) == downloaded
