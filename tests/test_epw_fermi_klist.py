"""scripts/epw_fermi_klist.py: the batched interpolation matches the band-check model, the selection keeps exactly
the points with a band near E_F, and the list file has EPW's format with the full grid's weight."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_wannier_bands as W  # noqa: E402
import epw_fermi_klist as F  # noqa: E402


def write_two_band_model(folder: Path) -> None:
    """Two orbitals on a cubic lattice: hopping -1 eV to the six neighbours, on-site 0 and 3 eV, small mixing."""
    vectors = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    lines = ["test", "2", str(len(vectors)), " ".join(["1"] * len(vectors))]
    for r in vectors:
        for n in (1, 2):
            for m in (1, 2):
                if r == (0, 0, 0):
                    value = {(1, 1): 0.0, (2, 2): 3.0}.get((m, n), 0.2)
                else:
                    value = -1.0 if m == n else 0.05
                lines.append(f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{m:5d}{n:5d}{value:12.6f}{0.0:12.6f}")
    (folder / "t_hr.dat").write_text("\n".join(lines) + "\n")


def test_batched_bands_match_the_band_check_model(tmp_path):
    write_two_band_model(tmp_path)
    model = W.WannierModel(W.read_hr(tmp_path / "t_hr.dat"))
    k = np.random.default_rng(0).random((50, 3))
    assert np.allclose(F.bands_on(model, k, chunk=7), model.bands(k), atol=1e-10)


def test_selection_and_file(tmp_path):
    write_two_band_model(tmp_path)
    summary = F.build(tmp_path, "t", 8, fermi=0.5, window=0.6, sigma=0.05, out=tmp_path / "k.kf")
    model = W.WannierModel(W.read_hr(tmp_path / "t_hr.dat"))
    bands = model.bands(W.uniform_grid(8))
    expected = np.any(np.abs(bands - 0.5) <= 0.6, axis=1)
    assert summary["kept"] == int(expected.sum()) and 0 < summary["kept"] < 512
    lines = (tmp_path / "k.kf").read_text().splitlines()
    assert lines[0] == f"{summary['kept']} crystal" and len(lines) == summary["kept"] + 1
    assert all(float(line.split()[3]) == 1 / 512 for line in lines[1:])
    assert summary["closest_excluded_state_ev"] > 0.6
    assert summary["dos_relative_difference"] < 1e-12
