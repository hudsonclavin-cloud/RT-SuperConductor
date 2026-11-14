import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_wannier_bands as cwb  # noqa: E402

HA = cwb.HARTREE_EV
ONSITE, HOP = 0.0, -1.0           # eV; simple cubic, one orbital, nearest neighbours


def tight_binding(kfrac):
    k = 2 * np.pi * np.atleast_2d(kfrac)
    return ONSITE + 2 * HOP * np.cos(k).sum(axis=1)


def write_hr(path, blocks, num_wann, degeneracy=None):
    vectors = list(blocks)
    degeneracy = degeneracy or [1] * len(vectors)
    lines = [" written for a test", f"{num_wann:12d}", f"{len(vectors):12d}"]
    for start in range(0, len(degeneracy), 15):
        lines.append("".join(f"{d:5d}" for d in degeneracy[start:start + 15]))
    for r in vectors:
        for n in range(num_wann):
            for m in range(num_wann):
                value = blocks[r][m, n]
                lines.append(f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{m + 1:5d}{n + 1:5d}{value.real:12.6f}{value.imag:12.6f}")
    path.write_text("\n".join(lines) + "\n")


def cubic_blocks():
    blocks = {(0, 0, 0): np.array([[ONSITE]], complex)}
    for axis in range(3):
        for sign in (1, -1):
            r = [0, 0, 0]
            r[axis] = sign
            blocks[tuple(r)] = np.array([[HOP]], complex)
    return blocks


def write_qe_xml(path, kfrac, energies_ev, weights=None, fermi_ev=None, alat=5.0):
    """A minimal QE data-file-schema.xml: simple cubic cell, so cartesian k (2pi/alat) = fractional k."""
    weights = np.ones(len(kfrac)) if weights is None else weights
    cell = "".join(f"<a{i + 1}>{' '.join(str(alat * float(i == j)) for j in range(3))}</a{i + 1}>" for i in range(3))
    states = "".join(
        f"<ks_energies><k_point weight=\"{w}\">{' '.join(map(str, k))}</k_point>"
        f"<eigenvalues size=\"{len(e)}\">{' '.join(str(x / HA) for x in e)}</eigenvalues></ks_energies>"
        for k, e, w in zip(kfrac, energies_ev, weights))
    fermi = f"<fermi_energy>{fermi_ev / HA}</fermi_energy>" if fermi_ev is not None else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<root><output><atomic_structure alat=\"{alat}\"><cell>{cell}</cell></atomic_structure>"
                    f"<band_structure>{fermi}{states}</band_structure></output></root>")


EPW_IN = """--
&inputepw
  prefix      = 'tb'
  nbndsub     = 1
  dis_froz_min= -10.0
  dis_froz_max= 10.0
  fsthick     = 0.4 ! eV
  degaussw    = 0.1 ! eV
  nk1         = 4
  nk2         = 4
  nk3         = 4
 /
"""


def test_namelist_values_are_read_with_comments_and_fortran_exponents():
    text = "  degaussw    = 0.1 ! eV\n  conv_thr = 1.0d-12\n  prefix = 'ta'\n"
    assert cwb.read_float(text, "degaussw") == 0.1
    assert cwb.read_float(text, "conv_thr") == pytest.approx(1e-12)
    assert cwb.read_namelist_value(text, "prefix") == "ta"
    assert cwb.read_float(text, "missing", 7.0) == 7.0


def test_interpolation_reproduces_a_known_tight_binding_model(tmp_path):
    write_hr(tmp_path / "tb_hr.dat", cubic_blocks(), 1)
    model = cwb.WannierModel(cwb.read_hr(tmp_path / "tb_hr.dat"))
    k = np.random.default_rng(0).random((50, 3))
    assert np.allclose(model.bands(k)[:, 0], tight_binding(k), atol=1e-9)


def test_degeneracy_divides_each_term(tmp_path):
    write_hr(tmp_path / "tb_hr.dat", cubic_blocks(), 1, degeneracy=[1] + [2] * 6)
    model = cwb.WannierModel(cwb.read_hr(tmp_path / "tb_hr.dat"))
    k = np.array([[0.1, 0.2, 0.3]])
    assert model.bands(k)[0, 0] == pytest.approx(ONSITE + HOP * np.cos(2 * np.pi * k).sum())


def test_wsvec_shifts_split_a_term_between_equivalent_vectors(tmp_path):
    blocks = {(0, 0, 0): np.zeros((1, 1), complex), (1, 0, 0): np.array([[HOP]], complex)}
    write_hr(tmp_path / "tb_hr.dat", blocks, 1)
    (tmp_path / "tb_wsvec.dat").write_text("## written with use_ws_distance=.true.\n"
                                           "    1    0    0    1    1\n    2\n    0    0    0\n    1    0    0\n")
    shifted = cwb.WannierModel(cwb.read_hr(tmp_path / "tb_hr.dat"), cwb.read_wsvec(tmp_path / "tb_wsvec.dat"))
    plain = cwb.WannierModel(cwb.read_hr(tmp_path / "tb_hr.dat"))
    x = 0.17
    k = np.array([[x, 0.0, 0.0]])
    # Half the hopping at R = 1 and half at R = 2, then made Hermitian.
    assert shifted.bands(k)[0, 0] == pytest.approx(HOP / 2 * (np.cos(2 * np.pi * x) + np.cos(4 * np.pi * x)))
    assert plain.bands(k)[0, 0] == pytest.approx(HOP * np.cos(2 * np.pi * x))


def test_an_hr_line_r_m_n_is_matrix_element_m_n(tmp_path):
    a = 0.3 + 0.4j
    blocks = {(0, 0, 0): np.diag([1.0, -1.0]).astype(complex),
              (1, 0, 0): np.array([[0, a], [0, 0]], complex),
              (-1, 0, 0): np.array([[0, 0], [np.conj(a), 0]], complex)}
    write_hr(tmp_path / "tb_hr.dat", blocks, 2)
    model = cwb.WannierModel(cwb.read_hr(tmp_path / "tb_hr.dat"))
    k = np.array([0.23, 0.0, 0.0])
    phase = np.exp(2j * np.pi * 0.23)
    expected = np.array([[1.0, a * phase], [np.conj(a) / phase, -1.0]])
    assert np.allclose(model.hamiltonian(k), expected)


def test_wrong_vector_count_is_refused(tmp_path):
    write_hr(tmp_path / "tb_hr.dat", cubic_blocks(), 1)
    text = (tmp_path / "tb_hr.dat").read_text().splitlines()
    text[2] = "           8"
    (tmp_path / "tb_hr.dat").write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="lattice vectors"):
        cwb.read_hr(tmp_path / "tb_hr.dat")


def test_nearest_state_errors_and_dos():
    direct = np.array([[0.0, 0.5, 3.0], [0.2, 0.9, 5.0]])
    interp = np.array([[0.01, 0.45], [0.2, 0.8]])
    errors = cwb.nearest_state_errors(direct, interp, centre=0.3, half_width=0.4)
    assert sorted(np.round(errors, 3)) == [0.0, 0.01, 0.05]           # 0.9, 3.0 and 5.0 are outside the window
    dos = cwb.gaussian_dos(np.array([[0.0]]), np.array([1.0]), at=0.0, sigma=0.1)
    assert dos == pytest.approx(1 / (0.1 * math.sqrt(math.pi)))


def _fake_run(tmp_path, nbnd=18):
    run = tmp_path / "run" / "epw"
    (tmp_path / "run" / "pp").mkdir(parents=True)
    (tmp_path / "run" / "pp" / "X.upf").write_text("pseudo\n")
    save = run / "tb.save"
    save.mkdir(parents=True)
    (save / "charge-density.hdf5").write_bytes(b"rho")
    (save / "data-file-schema.xml").write_text("<root/>")
    (save / "wfc1.hdf5").write_bytes(b"large wavefunction")
    scf = (" &control\n    calculation = 'scf'\n    prefix = 'tb'\n    pseudo_dir = '../pp/'\n    outdir = './'\n"
           "    tprnfor = .true.\n /\n &system\n    ibrav = 1\n    celldm(1) = 5.0\n    nat = 1\n    ntyp = 1\n"
           "    ecutwfc = 40\n /\n &electrons\n /\nATOMIC_SPECIES\n X 1.0 X.upf\nATOMIC_POSITIONS crystal\n"
           " X 0 0 0\nK_POINTS automatic\n 8 8 8 0 0 0\n")
    (run / "scf.in").write_text(scf)
    (run / "nscf.in").write_text(scf.replace("'scf'", "'nscf'").replace("ecutwfc = 40", f"ecutwfc = 40\n    nbnd = {nbnd}"))
    (run / "epw.in").write_text(EPW_IN)
    return run


def test_prepare_writes_a_small_self_contained_check(tmp_path):
    run = _fake_run(tmp_path)
    meta = cwb.prepare(run, tmp_path / "check", random_points=20, grid_points=5, dense_grid=6)
    check = tmp_path / "check"
    bands = (check / "bands.in").read_text()
    assert "calculation = 'bands'" in bands and "pseudo_dir  = './pp/'" in bands
    assert "nbnd        = 18" in bands and "tprnfor" not in bands
    count = int(bands.split("K_POINTS crystal\n")[1].splitlines()[0])
    assert count == 25 == len(meta["on_grid"]) and sum(meta["on_grid"]) == 5
    grid_rows = bands.split("K_POINTS crystal\n")[1].splitlines()[1:][20:]
    for row in grid_rows:                                          # grid points sit on the 4x4x4 coarse grid
        assert all(abs(float(x) * 4 - round(float(x) * 4)) < 1e-9 for x in row.split()[:3])
    dense = (check / "dense" / "nscf.in").read_text()
    assert "calculation = 'nscf'" in dense and "K_POINTS automatic\n 6 6 6 0 0 0" in dense
    for folder in (check, check / "dense"):
        assert (folder / "pp" / "X.upf").is_file()
        assert (folder / "tb.save" / "charge-density.hdf5").is_file()
        assert not (folder / "tb.save" / "wfc1.hdf5").exists()          # wavefunctions are not copied
    run_sh = (check / "run.sh").read_text()
    assert (check / "run.sh").stat().st_mode & 0o111
    assert 'launcher="${LAUNCHER:-srun}"' in run_sh and "cp tb.save/data-file-schema.xml bands.xml" in run_sh


def test_prepare_refuses_a_run_without_its_band_count_or_density(tmp_path):
    run = _fake_run(tmp_path, nbnd=0)
    with pytest.raises(ValueError, match="nbnd"):
        cwb.prepare(run, tmp_path / "check")
    run = _fake_run(tmp_path / "second")
    (run / "tb.save" / "charge-density.hdf5").unlink()
    with pytest.raises(ValueError, match="charge density"):
        cwb.prepare(run, tmp_path / "check2")


def _analysis_case(tmp_path, offset_ev=0.0):
    run = _fake_run(tmp_path)
    write_hr(run / "tb_hr.dat", cubic_blocks(), 1)
    check = tmp_path / "check"
    rng = np.random.default_rng(1)
    test = rng.random((200, 3))
    grid = np.array([[0.25, 0.5, 0.0], [0.75, 0.25, 0.5]])
    points = np.vstack([test, grid])
    direct = tight_binding(points)[:, None].copy()
    direct[:200] += offset_ev
    write_qe_xml(check / "bands.xml", points, direct)
    n = 12
    dense = cwb.uniform_grid(n)
    write_qe_xml(check / "dense" / "tb.save" / "data-file-schema.xml", dense, tight_binding(dense)[:, None],
                 fermi_ev=0.3)
    (check / "check.json").write_text(json.dumps({"on_grid": [False] * 200 + [True] * 2, "dense_grid": n}))
    return run, check


def test_analyze_passes_an_exact_interpolation(tmp_path):
    run, check = _analysis_case(tmp_path)
    report = cwb.analyze(run, check)
    assert report["verdict"] == "pass", report
    assert report["errors_within_fsthick"]["mean_mev"] == pytest.approx(0, abs=1e-3)
    assert report["coarse_grid_exactness"]["max_mev"] == pytest.approx(0, abs=1e-3)
    assert report["dos_at_fermi_level_per_spin"]["0.1"]["ratio"] == pytest.approx(1.0)
    assert report["thresholds"]["max_p95_mev"] == 100.0                # EPW's degaussw by default
    assert report["test_kpoints"] == 200


def test_analyze_fails_when_bands_are_off_between_grid_points(tmp_path):
    run, check = _analysis_case(tmp_path, offset_ev=0.08)
    report = cwb.analyze(run, check)
    assert report["verdict"] == "fail"
    assert report["checks"]["mean_error_within_fsthick"] is False
    assert report["errors_within_fsthick"]["mean_mev"] == pytest.approx(80, abs=1)
    assert report["coarse_grid_exactness"]["max_mev"] == pytest.approx(0, abs=1e-3)   # grid points still exact


def test_analyze_flags_files_from_a_different_calculation(tmp_path):
    run, check = _analysis_case(tmp_path)
    blocks = cubic_blocks()
    blocks[(0, 0, 0)] = np.array([[0.5]], complex)                     # a different onsite energy
    write_hr(run / "tb_hr.dat", blocks, 1)
    report = cwb.analyze(run, check)
    assert report["verdict"] == "fail"
    assert any("do not belong" in problem or "does not belong" in problem for problem in report["problems"])


def test_cli_exit_codes(tmp_path):
    run, check = _analysis_case(tmp_path)
    assert cwb.main(["analyze", "--epw-dir", str(run), "--check-dir", str(check)]) == 0
    assert json.loads((check / "report.json").read_text())["verdict"] == "pass"
    assert cwb.main(["analyze", "--epw-dir", str(run), "--check-dir", str(check), "--max-mean-mev", "-1"]) == 1
    assert cwb.main(["analyze", "--epw-dir", str(run), "--check-dir", str(tmp_path / "missing")]) == 2
