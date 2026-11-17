"""The second blind table: reading compositions from file names and nulls from susceptibility curves."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import blind_test2 as B  # noqa: E402


@pytest.mark.parametrize("name, formula", [
    ("ML_GobletOfFire_20210628_1_EP_0_IH1004A-Al2Fe3Si4-AC-64p3mg", "Al2Fe3Si4"),
    ("ML_GobletOfFire_20210728_4_EP_0_IH1013A-Zr3Fe4Sn3p75Inp25-AC-33p7mg", "Zr3Fe4Sn3.75In.25"),
    ("ML_Champion_20220322_1_IH_0_IH1042A_Mg5Sn(BO5)2_16p1mg_051122_SCTest", "Mg5Sn(BO5)2"),
    ("ML_Dukie_20220218_2_BW_0_BW1001B_ZrNiIn2_25-1_InFlux_Crystal4_1p39mg_HC_051122", "ZrNiIn2"),
    ("ML_GobletOfFire_20201117_EP2100C_Ta3Si_1_17pt8mg", "Ta3Si"),
])
def test_the_composition_is_read_from_the_file_name(name, formula):
    assert B.formula_in(name) == (formula, "")


@pytest.mark.parametrize("name, word", [
    ("EP2115A", "no composition"),                                             # a laboratory code only
    ("ML_INIGO_20220331_3_TW_0_TW1029C-Ba2CaMoO6reduced-AC-4.9mg", "reduced"),
    ("ML_Kima_20210322_3_EP_0_EP2131C_MgCuBi+Bi_19p5mg_PPMS", "mixture"),
    ("ML_INIGO_20220518_1_TW_0_TW1055A-Mix_-Sr2LiTeO6-Sr2Lip8Te1p2O6-AC-32.2mg", "mixture"),
    ("ML_Cornell_20210609_4_EP_EP2154D_NdNiO2onSrTiO3", "film"),
])
def test_samples_the_plan_leaves_out_are_refused_with_a_reason(name, word):
    formula, reason = B.formula_in(name)
    assert formula is None and word in reason


def test_a_sample_code_is_never_taken_for_a_formula():
    # BW1001B would parse as boron and tungsten
    assert B.formula_in("ML_Dukie_20220218_1_BW_0_BW1001B_SCTest")[0] is None
    assert B.formula_in("ML_x_HC_051122")[0] is None


def test_mass_comes_from_the_header_then_the_file_name():
    assert B.mass_mg(["INFO,47.7,SAMPLE_MASS"], "x_10p0mg") == 47.7
    assert B.mass_mg(["INFO,mass,52.1"], "x") == 52.1
    assert B.mass_mg(["INFO,,SAMPLE_MASS"], "IH1004A-Al2Fe3Si4-AC-64p3mg") == 64.3
    assert B.mass_mg([], "Ta3Si_1_17pt8mg") == 17.8
    assert B.mass_mg([], "nothing here") is None


def scan(values, temperatures=None, field=10.0, mass=50.0, column="AC X' (emu/Oe)"):
    temperatures = temperatures or [2.0 + 0.5 * i for i in range(len(values))]
    lines = ["[Header]", f"INFO,{mass},SAMPLE_MASS", "[Data]", f"Comment,Temperature (K),Magnetic Field (Oe),{column}"]
    lines += [f",{t},{field},{v}" for t, v in zip(temperatures, values)]
    return "\n".join(lines)


def full_shielding(mass_mg=50.0):
    """The susceptibility (emu/Oe) of a fully shielding sample of this mass at the rule's density."""
    return -(mass_mg / 1000.0) / (4.0 * math.pi * B.DENSITY_G_CM3)


def test_a_superconductor_gives_a_shielding_fraction_near_one():
    chi = full_shielding()
    out = B.read_scan(scan([chi, chi, chi, chi * 0.5, 0.0, 0.0, 0.0, 0.0]), "x")
    assert out["usable"] and out["shielding_fraction"] == pytest.approx(1.0)
    assert out["lowest_k"] == 2.0 and B.judge([out]) == ("signal seen", "a diamagnetic signal of 2% of full shielding or more")


def test_a_paramagnet_and_a_ferromagnet_are_nulls():
    para = B.read_scan(scan([3e-7, 2e-7, 1.5e-7, 1e-7, 1e-7, 1e-7, 1e-7]), "x")
    ferro = B.read_scan(scan([5e-3, 6e-3, 9e-3, 4e-3, 1e-3, 1e-4, 1e-5]), "x")       # large, positive, falling
    assert B.judge([para])[0] == "null" and B.judge([ferro])[0] == "null"
    assert ferro["shielding_fraction"] < 0


def test_one_bad_point_does_not_make_a_signal():
    chi = full_shielding()
    out = B.read_scan(scan([1e-7, 1e-7, chi, 1e-7, 1e-7, 1e-7, 1e-7]), "x")           # a single spike
    assert out["shielding_fraction"] < B.SIGNAL_FRACTION


def test_a_small_impurity_signal_is_left_out_not_called_a_null():
    chi = 0.05 * full_shielding()
    out = B.read_scan(scan([chi, chi, chi, 0.0, 0.0, 0.0, 0.0]), "x")
    assert out["shielding_fraction"] == pytest.approx(0.05) and B.judge([out])[0] == "signal seen"


def test_a_scan_must_reach_two_kelvin_and_stay_in_a_small_fixed_field():
    high = B.read_scan(scan([1e-7] * 6, temperatures=[5, 6, 7, 8, 9, 10]), "x")
    assert high["usable"] and B.judge([high]) == ("left out", "a scan does not reach 2.1 K")
    assert B.read_scan(scan([1e-7] * 6, field=1000.0), "x")["reason"] == "field above 100 Oe"
    sweep = "\n".join(["[Data]", "Temperature (K),Magnetic Field (Oe),AC X' (emu/Oe)"] +
                      [f"{2 + i},{100 * i},1e-7" for i in range(6)])
    assert B.read_scan("[Header]\nINFO,50,SAMPLE_MASS\n" + sweep, "x")["reason"] == "the field changes during the scan"
    assert B.read_scan(scan([1e-7] * 6, temperatures=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5]), "x")["reason"] == "not a temperature scan"


def test_the_older_instrument_s_columns_are_read_too():
    chi = full_shielding()
    text = "\n".join(["[Header]", "INFO,mass,50", "[Data]",
                      "Comment,Temperature (K),Magnetic Field (Oe),Amplitude (Oe),M' (emu)"] +
                     [f",{2 + 0.5 * i},10,3,{3 * v}" for i, v in enumerate([chi, chi, chi, 0, 0, 0, 0])])
    out = B.read_scan(text, "x")
    assert out["kind"] == "AC" and out["shielding_fraction"] == pytest.approx(1.0)


def test_any_scan_with_a_signal_decides_the_composition():
    chi = full_shielding()
    clean = B.read_scan(scan([1e-7] * 7), "x")
    signal = B.read_scan(scan([chi, chi, chi, 0, 0, 0, 0]), "x")
    assert B.judge([clean, signal])[0] == "signal seen"
    assert B.judge([{"usable": False, "reason": "no sample mass"}]) == ("left out", "no usable scan")


def test_an_abstract_is_rebuilt_from_its_word_positions():
    assert B.abstract_text({"superconducts": [2], "It": [0], "below": [3], "bulk": [1], "4.2": [4], "K": [5]}) == \
        "It bulk superconducts below 4.2 K"
    assert B.abstract_text(None) == ""


def test_the_paper_filter_wants_a_temperature_and_refuses_films_pressure_and_theory():
    good = "Polycrystalline LaPt3P shows bulk superconductivity below 1.1 K."
    assert B.keep_paper("Superconductivity in LaPt3P", good)
    assert not B.keep_paper("Superconductivity in LaPt3P", "A new superconductor is reported.")          # no temperature
    assert not B.keep_paper("Superconductivity in LaPt3P thin films", good)
    assert not B.keep_paper("Superconductivity in LaPt3P", good + " Under pressure Tc rises to 3 K.")
    assert not B.keep_paper("Superconductivity in LaPt3P", "First-principles calculations give Tc of 2 K.")
