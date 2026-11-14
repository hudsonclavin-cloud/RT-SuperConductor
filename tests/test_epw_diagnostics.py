"""Checks on a finished EPW run: frequency moments and symmetry of the coupling."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import epw_diagnostics as E  # noqa: E402

BCC = 0.5 * np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]])   # Quantum ESPRESSO's ibrav = 3


def write_a2f(path: Path, omega: np.ndarray, a2f: np.ndarray) -> None:
    lines = [" w[meV] a2f and integrated 2*a2f/w for   10 smearing values"]
    lines += [f"{w:12.7f} {f:12.7f} {f:12.7f} {f * 0.9:12.7f}" for w, f in zip(omega, a2f)]
    lines += [" Integrated el-ph coupling", "  #    1.0   1.0   1.0"]
    path.write_text("\n".join(lines) + "\n")


def test_moments_of_a_single_sharp_phonon_peak(tmp_path):
    omega = np.linspace(0.01, 30.0, 6000)
    centre, lam = 12.0, 0.8                                    # one phonon at 12 meV carrying lambda = 0.8
    a2f = lam * centre / 2.0 * np.exp(-0.5 * ((omega - centre) / 0.05) ** 2) / (0.05 * np.sqrt(2 * np.pi))
    write_a2f(tmp_path / "x.a2f", omega, a2f)
    out = E.moments(*E.read_a2f(tmp_path / "x.a2f"), mass_amu=180.94788)
    assert out["lambda"] == pytest.approx(lam, rel=2e-3)
    for key in ("omega_log_k", "omega_1_k", "omega_2_k"):
        assert out[key] == pytest.approx(centre / E.MEV_PER_KELVIN, rel=2e-3)        # all averages sit at the peak
    assert out["half_of_lambda_below_mev"] == pytest.approx(centre, abs=0.05)
    assert out["eta_ev_per_a2"] == pytest.approx(E.eta_from(lam, centre / E.MEV_PER_KELVIN, 180.94788), rel=5e-3)
    # The second smearing column is 0.9 of the first, so its lambda is too.
    assert E.moments(*E.read_a2f(tmp_path / "x.a2f", column=3))["lambda"] == pytest.approx(0.9 * lam, rel=2e-3)
    with pytest.raises(ValueError):
        E.read_a2f(ROOT / "README.md")


def test_eta_scales_with_lambda_and_with_frequency_squared():
    base = E.eta_from(1.0, 150.0, 180.94788)
    assert E.eta_from(2.0, 150.0, 180.94788) == pytest.approx(2 * base)
    assert E.eta_from(1.0, 300.0, 180.94788) == pytest.approx(4 * base)
    assert 7.0 < base < 8.5                      # tantalum-like numbers give a few eV per square angstrom


spglib = pytest.importorskip("spglib")


@pytest.fixture(scope="module")
def bcc_operations():
    return E.reciprocal_operations(BCC, np.zeros((1, 3)), ["Ta"])


def test_stars_of_a_bcc_grid_have_the_textbook_sizes(bcc_operations):
    grid = (12, 12, 12)
    assert len(bcc_operations) == 48
    sizes = [len(E.star_of(np.array(q), bcc_operations, grid)) for q in
             ([0, 0, 0], [0, 0, 1 / 12], [1 / 12, 1 / 12, 1 / 12], [0, 1 / 12, 2 / 12], [3 / 12, 4 / 12, 1 / 12])]
    assert sizes == [1, 12, 6, 24, 48]
    # Every grid point belongs to exactly one star, and a bcc 12-grid has 72 of them.
    seen, stars = set(), 0
    for i in range(12):
        for j in range(12):
            for k in range(12):
                if (i, j, k) not in seen:
                    seen.update(E.star_of(np.array([i, j, k]) / 12, bcc_operations, grid))
                    stars += 1
    assert stars == 72 and len(seen) == 1728
    with pytest.raises(ValueError):
        E.grid_key(np.array([0.05, 0.0, 0.0]), grid)                 # not a point of this grid


def epw_text(points: list[tuple[tuple, list[float], list[float]]]) -> str:
    lines = ["     Using uniform q-mesh:    12   12   12"]
    for number, (q, omegas, lambdas) in enumerate(points, start=1):
        lines.append(f"     ismear =     1 iq = {number:7d} coord.: {q[0]:9.5f} {q[1]:9.5f} {q[2]:9.5f} wt: {1 / len(points):9.5f} Temp:      0.075K")
        for mode, (omega, lam) in enumerate(zip(omegas, lambdas), start=1):
            lines.append(f"     lambda___( {mode:6d} )= {lam:14.6f}   gamma___= {0.01:14.6f} meV   omega= {omega:11.4f} meV")
        lines.append(f"     lambda___( tot )= {sum(lambdas):14.6f}")
    return "\n".join(lines) + "\n"


def test_symmetry_report_passes_a_symmetric_output_and_fails_a_broken_one(tmp_path, bcc_operations):
    grid = (12, 12, 12)
    star = [np.array(k) / 12 for k in E.star_of(np.array([0, 0, 1 / 12]), bcc_operations, grid)]
    other = [np.array(k) / 12 for k in E.star_of(np.array([1 / 12, 1 / 12, 1 / 12]), bcc_operations, grid)]
    good = [(tuple(q), [2.0, 2.5, 5.8], [1.4, 1.2, 0.4]) for q in star] + \
           [(tuple(q), [2.8, 2.8, 8.7], [0.8, 0.8, 0.3]) for q in other]
    (tmp_path / "good.out").write_text(epw_text(good))
    points = E.parse_epw_output(tmp_path / "good.out")
    assert len(points) == 18 and points[0]["lambda"] == pytest.approx(3.0)
    assert E.fine_grid((tmp_path / "good.out").read_text()) == grid
    report = E.symmetry_report(points, bcc_operations, grid)
    assert report["verdict"] == "pass" and report["stars_with_two_or_more_members"] == 2
    assert report["median_spread_over_mean"] == 0.0 and report["negative_coupling_points"] == 0

    broken = list(good)
    broken[0] = (broken[0][0], [2.0, 2.5, 5.8], [2.8, 1.2, 0.4])          # one member of the first star disagrees
    broken[3] = (broken[3][0], [2.0, 2.5, 5.8], [-0.5, 1.2, 0.4])         # and one has a negative coupling
    broken[13] = (broken[13][0], [2.8, 2.9, 8.7], [0.8, 0.8, 0.3])        # and one frequency is off in the second
    (tmp_path / "bad.out").write_text(epw_text(broken))
    report = E.symmetry_report(E.parse_epw_output(tmp_path / "bad.out"), bcc_operations, grid)
    assert report["verdict"] == "fail" and report["negative_coupling_points"] == 0     # the total stays positive there
    assert report["largest_frequency_spread_mev"] == pytest.approx(0.1)
    assert report["worst_stars"][0]["lambda_max"] == pytest.approx(4.4)
    assert any("frequencies differ" in problem for problem in report["problems"])
    assert any("not symmetric" in problem for problem in report["problems"])
    negative = [(q, om, [-2.0, 0.5, 0.4]) if i == 0 else (q, om, lam) for i, (q, om, lam) in enumerate(good)]
    (tmp_path / "negative.out").write_text(epw_text(negative))
    report = E.symmetry_report(E.parse_epw_output(tmp_path / "negative.out"), bcc_operations, grid)
    assert report["negative_coupling_points"] == 1 and any("delta_approx" in p for p in report["problems"])
    # An output with one point per star (a symmetry-reduced run) cannot be checked.
    lonely = E.symmetry_report(points[:1] + points[-1:], bcc_operations, grid)
    assert lonely["verdict"] == "fail" and "no star has two members" in lonely["problems"][0]


def test_star_file_and_structure_reader_and_command_line(tmp_path, bcc_operations, capsys):
    xml = tmp_path / "ta.xml"
    cell = BCC * 6.2736
    xml.write_text(
        '<qes><output><atomic_structure alat="6.2736" nat="1"><atomic_positions>'
        '<atom name="Ta" index="1">0.0 0.0 0.0</atom></atomic_positions><cell>'
        + "".join(f"<a{i + 1}>{' '.join(f'{x:.8f}' for x in row)}</a{i + 1}>" for i, row in enumerate(cell))
        + "</cell></atomic_structure></output></qes>")
    lattice, positions, species = E.read_structure(xml)
    np.testing.assert_allclose(lattice, cell)
    assert species == ["Ta"] and np.allclose(positions, 0.0)
    assert len(E.reciprocal_operations(lattice, positions, species)) == 48
    out = tmp_path / "star.qf"
    assert E.main(["stars", "--xml", str(xml), "--grid", "12", "12", "12", "--point", "0", "0", "0.0833333333",
                   "--point", "0.0833333333", "0.0833333333", "0.0833333333", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "18 crystal" and len(lines) == 19
    assert sum(float(line.split()[3]) for line in lines[1:]) == pytest.approx(1.0)
    star = [tuple(float(x) for x in line.split()[:3]) for line in lines[1:13]]
    (tmp_path / "epw.out").write_text(epw_text([(q, [2.0, 2.5, 5.8], [1.4, 1.2, 0.4]) for q in star]))
    capsys.readouterr()
    assert E.main(["symmetry", "--epw-out", str(tmp_path / "epw.out"), "--xml", str(xml),
                   "--report", str(tmp_path / "report.json")]) == 0
    assert '"verdict": "pass"' in capsys.readouterr().out and (tmp_path / "report.json").exists()
