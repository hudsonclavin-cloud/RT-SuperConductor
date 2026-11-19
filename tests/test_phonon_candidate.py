"""The Hazel candidate packages: input generation, relaxed-cell parsing and the chain report."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phonon_chain_report as R  # noqa: E402
import prepare_phonon_candidate as P  # noqa: E402

RELAX = """
Begin final coordinates
     new unit-cell volume =    392.0 a.u.^3
CELL_PARAMETERS (angstrom)
   3.880000000   0.000000000   0.000000000
   0.000000000   3.880000000   0.000000000
   0.000000000   0.000000000   3.880000000

ATOMIC_POSITIONS (crystal)
Ir            0.0000000000        0.0000000000        0.0000000000
Re            0.0000000000        0.5000000000        0.5000000000
Re            0.5000000000        0.0000000000        0.5000000000
Re            0.5000000000        0.5000000000        0.0000000000
End final coordinates
"""


def test_from_relax_reads_the_final_cell_and_positions():
    cell, sites = P.from_relax(RELAX)
    assert cell[0] == [3.88, 0.0, 0.0] and len(sites) == 4
    assert sites[1] == ("Re", [0.0, 0.5, 0.5])
    with pytest.raises(SystemExit):
        P.from_relax("the run stopped early")


def test_from_relax_converts_alat_units():
    text = RELAX.replace("(angstrom)", "(alat= 7.33213700)").replace("3.880000000", "1.000000000")
    cell, _ = P.from_relax(text)
    assert abs(cell[0][0] - 7.332137 * P.BOHR) < 1e-9


def test_meshes_are_consistent_for_every_candidate():
    for spec in P.CANDIDATES.values():
        for k, q in ((spec["k"], spec["q"]), (spec["smoke_k"], spec["smoke_q"])):
            assert all(a % b == 0 for a, b in zip(k, q))          # q must divide k for the interpolation


def test_inputs_carry_the_settings_the_method_needs():
    pytest.importorskip("pymatgen")
    cell, sites = P.from_relax(RELAX)
    scf = P.pw_input("re3ir", "scf", cell, sites, (12, 12, 12), la2f=True)
    assert "la2F        = .true." in scf and " 12 12 12 0 0 0" in scf and "nat         = 4" in scf
    assert "Re_ONCV_PBE_sr.upf" in scf and "Ir_ONCV_PBE_sr.upf" in scf and "pseudo_dir  = '../pp/'" in scf
    ph = P.ph_input("re3ir", (4, 4, 4))
    assert "electron_phonon = 'interpolated'" in ph and "recover  = .true." in ph and "nq1 = 4" in ph
    assert "la2F   = .true." in P.q2r_input("re3ir") and "la2F   = .true." in P.matdyn_input("re3ir", (24, 24, 24))


def test_allen_dynes_matches_a_hand_calculation():
    assert abs(R.allen_dynes(1.0, 100.0, 0.10) - 6.96) < 0.02
    assert R.allen_dynes(0.1, 100.0, 0.13) == 0.0                 # too weak to superconduct


def test_report_reads_frequencies_and_flags_imaginary_modes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "package.json").write_text('{"name": "x", "formula": "X"}', encoding="utf-8")
    (tmp_path / "CHAIN_MODE.txt").write_text("mode=full\n", encoding="utf-8")
    (run / "x.dyn0").write_text("", encoding="utf-8")
    (run / "x.dyn1").write_text("     freq (    1) =      -1.50 [THz] =     -50.03 [cm-1]\n"
                                "     freq (    2) =       3.00 [THz] =     100.07 [cm-1]\n", encoding="utf-8")
    out = R.report(run)
    assert out["lowest_frequency_cm1"] == -50.03 and out["imaginary_mode"] is True
    assert out["verdict"].startswith("incomplete") or out["verdict"].startswith("fails")


def test_symmetrise_restores_the_symmetry_rounding_lost():
    """Zr6Ga2Ni's Materials Project coordinates, as stored: 2 operations before, the 12 of P-62m after."""
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cell = [[3.90878567, -6.77034521, -0.00000008], [3.90883824, 6.77037573, 0.00000004], [-0.00000002, 0.00000003, 3.3873679]]
    sites = [("Zr", [0.25264019, 0.99999461, 0.50000002]), ("Zr", [0.99999866, 0.25264437, 0.50000003]),
             ("Zr", [0.74735709, 0.74735842, 0.50000002]), ("Zr", [0.59502954, 0.99998575, 0.99999997]),
             ("Zr", [0.9999927, 0.5950267, 0.99999997]), ("Zr", [0.40495731, 0.40495211, 0.0]),
             ("Ga", [0.33331795, 0.66666693, 0.49999999]), ("Ga", [0.66668244, 0.33333046, 0.5]),
             ("Ni", [0.00002413, 0.00004066, 0.00000001])]

    def operations(c, s):
        structure = Structure(Lattice(c), [e for e, _ in s], [x for _, x in s])
        return len(SpacegroupAnalyzer(structure, symprec=1e-5).get_symmetry_operations())

    assert operations(cell, sites) == 2
    new_cell, new_sites = P.symmetrise(cell, sites)
    assert operations(new_cell, new_sites) == 12
    assert sorted(e for e, _ in new_sites) == sorted(e for e, _ in sites)


LAMBDA_FILE = """
  Electron-phonon coupling constant, lambda

 Broadening   0.0150 lambda       2.8010 dos(Ef) 24.8488 omega_ln [K]     79.7752
 Broadening   0.0200 lambda       2.6031 dos(Ef) 23.5174 omega_ln [K]     80.6585
 Broadening   0.0250 lambda       2.4204 dos(Ef) 22.8094 omega_ln [K]     81.6900
"""


def test_lambda_table_reads_matdyn_s_real_format():
    """Lines copied from the lambda file matdyn.x 7.6 wrote in the Re3Ir rehearsal."""
    rows = R.lambda_table(LAMBDA_FILE)
    assert [row["broadening_ry"] for row in rows] == [0.015, 0.02, 0.025]
    assert rows[1] == {"broadening_ry": 0.02, "lambda": 2.6031, "dos_ef": 23.5174, "omega_log_k": 80.6585}


def test_report_finds_the_package_when_given_a_relative_path(tmp_path, monkeypatch):
    """The job runs the report as `phonon_chain_report.py .` from inside run/; it must still read ../package.json."""
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "package.json").write_text('{"name": "x", "formula": "X"}', encoding="utf-8")
    (tmp_path / "CHAIN_MODE.txt").write_text("mode=smoke\n", encoding="utf-8")
    (run / "x.dyn0").write_text("", encoding="utf-8")
    (run / "x.dyn1").write_text("     freq (    1) =       3.00 [THz] =     100.07 [cm-1]\n", encoding="utf-8")
    (run / "lambda").write_text(LAMBDA_FILE, encoding="utf-8")
    monkeypatch.chdir(run)
    out = R.report(Path("."))
    assert out["formula"] == "X" and out["mode"] == "smoke"
    assert out["verdict"].startswith("smoke run") and out["headline"]["lambda"] == 2.6031


def test_hexagonal_cells_are_written_in_quantum_espresso_s_setting():
    """SbTe2's relaxed cell as the Materials Project orients it; ph.x refused to label its modes in that setting."""
    cell = P.qe_orientation([[2.218996908, -3.843414532, 0.0], [2.218996908, 3.843414532, 0.0], [0.0, 0.0, 5.687223496]])
    a = (2.218996908 ** 2 + 3.843414532 ** 2) ** 0.5
    assert cell[0] == [pytest.approx(a), 0.0, 0.0]
    assert cell[1] == [pytest.approx(-a / 2), pytest.approx(a * 3 ** 0.5 / 2), 0.0]
    assert cell[2] == [0.0, 0.0, pytest.approx(5.687223496)]
    cubic = [[3.88, 0.0, 0.0], [0.0, 3.88, 0.0], [0.0, 0.0, 3.88]]
    assert P.qe_orientation(cubic) == cubic


GAMMA_DYN = """
     Diagonalizing the dynamical matrix

     q = (    0.000000000   0.000000000   0.000000000 )

 **************************************************************************
     freq (    1) =      -0.320745 [THz] =     -10.698889 [cm-1]
 (  0.574290  0.000000 -0.000025  0.000000  0.000000  0.000000 )
 (  0.578874  0.000000 -0.000025  0.000000  0.000000  0.000000 )
 (  0.578874  0.000000 -0.000025  0.000000  0.000000  0.000000 )
     freq (    2) =       2.217168 [THz] =      73.956764 [cm-1]
 (  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000 )
 (  0.707107  0.000000  0.000000  0.000000  0.000000  0.000000 )
 ( -0.707107  0.000000  0.000000  0.000000  0.000000  0.000000 )
 **************************************************************************
"""


def test_zone_centre_translations_are_not_counted_as_imaginary_modes():
    """Lines from SbTe2's zone-centre file: the -10.7 cm-1 mode is the whole crystal sliding, zero by symmetry."""
    found = R.modes(GAMMA_DYN)
    assert [item["translation"] for item in found] == [True, False]
    assert R.frequencies(GAMMA_DYN) == [73.956764]


def test_the_same_pattern_away_from_the_zone_centre_still_counts():
    """A soft mode is only excused at q = 0. Anywhere else, atoms moving together is a real (acoustic) mode."""
    elsewhere = GAMMA_DYN.replace("0.000000000   0.000000000   0.000000000 )", "0.000000000   0.000000000   0.500000000 )")
    assert R.frequencies(elsewhere) == [-10.698889, 73.956764]


def test_a_soft_optical_mode_at_the_zone_centre_still_counts():
    soft = GAMMA_DYN.replace("2.217168 [THz] =      73.956764", "-1.169600 [THz] =     -39.013358")
    assert R.frequencies(soft) == [-39.013358]


def test_a_full_run_of_a_candidate_that_failed_its_pre_check_is_refused(tmp_path):
    import subprocess

    script = ROOT / "scripts/hpc/submit_phonon_chain.sh"
    package = ROOT / "data/benchmarks/candidates/re3ir"
    assert (package / "PRECHECK_FAILED.txt").is_file()

    def run(**extra):
        env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "DRY_RUN": "1", **extra}
        return subprocess.run([str(script), str(package), str(tmp_path / "run"), "compute", "normal", "4"],
                              env=env, capture_output=True, text=True)

    refused = run(CHAIN_MODE="full")
    assert refused.returncode == 6 and "failed its stability pre-check" in refused.stderr
    assert run(CHAIN_MODE="full", ALLOW_FAILED="1").returncode == 0
    assert run(CHAIN_MODE="smoke").returncode == 0


def test_memory_is_requested_only_when_asked(tmp_path):
    import subprocess

    script = ROOT / "scripts/hpc/submit_phonon_chain.sh"
    package = ROOT / "data/benchmarks/candidates/sbte2"

    def run(**extra):
        env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "DRY_RUN": "1", **extra}
        return subprocess.run([str(script), str(package), str(tmp_path / "run"), "compute_partners", "short", "16"],
                              env=env, capture_output=True, text=True)

    plain, asked = run(), run(MEM="16G")
    assert plain.returncode == 0 and "--mem" not in plain.stderr
    assert asked.returncode == 0 and asked.stderr.count("--mem=16G") == 6
    assert "--partition=compute_partners --qos=short" in asked.stderr


PHONON_FIXTURES = ROOT / "tests/fixtures/phonon"


@pytest.mark.parametrize("name, translations, lowest_real, counted", [
    ("namgpb2", [18.054071, 18.054071, 18.054071], 60.435167, 9),
    ("limgpb2", [14.66871, 14.66871, 14.66871], 72.305195, 9),
    ("csir2p2", [-9.37198, 11.880014, 11.880014], 50.831082, 12),
])
def test_translations_are_found_when_atoms_differ_in_mass(name, translations, lowest_real, counted):
    """Real zone-centre files from the shortlist 2 pre-check. Na and Pb move by different amounts in a translation
    that ph.x has left impure, so the equal-amounts test missed them; the mass-weighted overlap does not."""
    text = (PHONON_FIXTURES / f"{name}.gamma.dyn").read_text(encoding="utf-8")
    found = R.modes(text)
    assert [item["cm1"] for item in found if item["translation"]] == translations
    values = R.frequencies(text)
    assert len(values) == counted and min(values) == lowest_real


def test_masses_are_read_from_the_header_in_atom_order():
    text = (PHONON_FIXTURES / "namgpb2.gamma.dyn").read_text(encoding="utf-8")
    masses = R.atom_masses(text)
    assert len(masses) == 4 and masses[2] == masses[3]                 # Na, Mg, Pb, Pb
    assert masses[2] / masses[0] == pytest.approx(207.2 / 22.99, rel=1e-3)
    assert R.atom_masses(GAMMA_DYN) is None                            # no header: the older test is used


def test_a_soft_optical_mode_still_counts_with_unequal_masses():
    text = (PHONON_FIXTURES / "namgpb2.gamma.dyn").read_text(encoding="utf-8")
    soft = text.replace("1.811801 [THz] =      60.435167", "-1.199170 [THz] =     -40.000000")
    assert soft != text and min(R.frequencies(soft)) == -40.0


def test_translations_are_excused_only_at_the_zone_centre_with_unequal_masses():
    text = (PHONON_FIXTURES / "csir2p2.gamma.dyn").read_text(encoding="utf-8")
    moved = text.replace("q = (    0.000000000   0.000000000   0.000000000 )",
                         "q = (    0.000000000   0.000000000   0.500000000 )")
    assert moved != text and min(R.frequencies(moved)) == -9.37198


def test_the_overlap_is_one_for_a_pure_translation_and_zero_for_no_net_momentum():
    masses = [1.0, 9.0]
    assert R.translation_overlap([(0.3, 0.0, 0.0), (0.3, 0.0, 0.0)], masses) == pytest.approx(1.0)
    assert R.translation_overlap([(0.9, 0.0, 0.0), (-0.1, 0.0, 0.0)], masses) == pytest.approx(0.0)
    # the light atom moving alone is mostly not a translation
    assert R.translation_overlap([(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)], masses) < 0.4


def qe_output(program: str, version: str = "7.6", processes: int = 16, done: bool = True) -> str:
    lines = [f"     Program {program} v.{version} starts on 17Nov2026 at 17: 6:45 ",
             f"     Parallel version (MPI & OpenMP), running on      {processes} processor cores"]
    if processes > 1:
        lines.append(f"     Number of MPI processes:                {processes}")
    lines.append("   JOB DONE." if done else "     iter #   5 total cpu time :  5077.4 secs")
    return "\n".join(lines) + "\n"


def chain_outputs(run, **changes):
    names = {"scf_fit.out": "PWSCF", "scf.out": "PWSCF", "ph.out": "PHONON", "q2r.out": "Q2R", "matdyn.out": "MATDYN"}
    for name, program in names.items():
        options = {"processes": 16 if program in ("PWSCF", "PHONON") else 1, **changes.get(name, {})}
        if options.pop("missing", False):
            continue
        (run / name).write_text(qe_output(program, **options), encoding="utf-8")


def test_the_report_records_which_quantum_espresso_ran(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    chain_outputs(run)
    out = R.programs(run)
    assert out["is_quantum_espresso"] and out["versions"] == ["7.6"] and out["matches_validated_version"]
    assert out["all_finished"]
    assert out["outputs"]["ph.out"] == {"present": True, "program": "PHONON", "version": "7.6", "mpi_processes": 16,
                                        "finished": True}
    assert out["outputs"]["q2r.out"]["mpi_processes"] == 1


def test_the_report_flags_another_version_an_unfinished_step_and_a_serial_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    chain_outputs(run, **{"ph.out": {"version": "7.3.1", "processes": 1, "done": False}, "matdyn.out": {"missing": True}})
    out = R.programs(run)
    assert out["versions"] == ["7.3.1", "7.6"] and not out["matches_validated_version"]
    assert not out["all_finished"] and out["outputs"]["matdyn.out"] == {"present": False}
    assert out["outputs"]["ph.out"]["mpi_processes"] == 1 and out["outputs"]["ph.out"]["finished"] is False


def test_a_file_that_is_not_quantum_espresso_output_is_caught(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    chain_outputs(run)
    (run / "scf.out").write_text("slurmstepd: error: execve(): pw.x: No such file or directory\n", encoding="utf-8")
    out = R.programs(run)
    assert not out["is_quantum_espresso"] and out["outputs"]["scf.out"]["program"] is None


def test_every_candidate_names_exactly_one_structure_source():
    for name, spec in P.CANDIDATES.items():
        sources = [key for key in ("mp_id", "alexandria_id") if key in spec]
        assert len(sources) == 1, name
        if "alexandria_id" in spec:
            assert spec["alexandria_id"] in P.ALEXANDRIA, name
        assert P.source_fields(spec) == {sources[0]: spec[sources[0]]}


def test_the_tetragonal_namgpb2_starts_from_alexandria_s_layered_structure():
    spec = P.CANDIDATES["namgpb2tet"]
    cell, sites = P.initial_structure(spec)
    assert sorted(element for element, _ in sites) == ["Mg", "Na", "Pb", "Pb"]
    assert abs(cell[0][0] - cell[1][1]) < 1e-9 and cell[2][2] > 2 * cell[0][0]   # tetragonal, c about 2a
    # Pb on the cell edge at heights z and 1 - z; Na and Mg at the face centre, stacked along c.
    lead = sorted(xyz[2] for element, xyz in sites if element == "Pb")
    assert abs(lead[0] + lead[1] - 1.0) < 1e-9
    assert {element: xyz for element, xyz in sites if element != "Pb"} == {"Na": [0.5, 0.5, 0.5], "Mg": [0.5, 0.5, 0.0]}


def test_the_tetragonal_namgpb2_keeps_its_symmetry_when_symmetrised():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cell, sites = P.symmetrise(*P.initial_structure(P.CANDIDATES["namgpb2tet"]))
    structure = Structure(Lattice(cell), [s for s, _ in sites], [xyz for _, xyz in sites])
    assert SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol() == "P4/mmm"
