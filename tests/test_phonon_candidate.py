"""The Hazel candidate packages: input generation, relaxed-cell parsing and the chain report."""
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
