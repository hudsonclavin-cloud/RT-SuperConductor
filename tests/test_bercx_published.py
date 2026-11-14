"""scripts/bercx_published.py: parsing both layouts of the supplementary LaTeX, the evidence tiers, the Fisher test,
the made-metal rule, and that the committed summary agrees with the committed join table."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bercx_published as B  # noqa: E402

LAYOUT_ONE = r"""
\subsection{Mn}
        \textbf{Space group} & $I4/mmm$ (139)      & \textbf{Allen-Dynes $T_c$} [K] & 14.8540       \\
        \textbf{Source database} & mpds-S558472~\cite{Endoh_1971}     & \multicolumn{1}{r}{$\boldsymbol{\omega_{\rm log}}$[meV]} & 21.8839     \\
        \textbf{MC3D entry} & \href{https://x/mc3d-22931/pbesol-v1}{mc3d-22931} &  \multicolumn{1}{r}{$\boldsymbol{\lambda}$}               & 0.9880  \\
        \textbf{Category} & problematic &  & \\
        \textbf{Note}&\multicolumn{3}{p{11cm}}{A note.} \\
        a=2.4766 & $\alpha$=119.9993 & \textbf{Isotropic $T_c$} [K] & 18.5555   \\
        b=2.4766 & $\beta$=119.9993   &   \textbf{Anisotropic $T_c$} [K] & 21.0548\\
                &                  &  \textbf{Experimental $T_c$} [K] & 9.25~\cite{Cappelletti_1967} \\
"""

LAYOUT_TWO = r"""
\subsection{Nb$_3$Ge}
        \textbf{Space group} & $Pm\bar{3}n$ (223)      & \textbf{Allen-Dynes $T_c$ [K]} & 18.0596       \\
        \textbf{Source database} & mpds-S555906     & \multicolumn{1}{r}{$\boldsymbol{\omega_{\rm log}}$[meV]} & 8.6716     \\
        \textbf{MC3D entry} & \href{https://x/mc3d-65870/pbesol-v1}{mc3d-65870} &  \multicolumn{1}{r}{$\boldsymbol{\lambda}$}               & 3.3381  \\
        \textbf{Category} & \multicolumn{3}{l}{known-exp-structure} \\
        \textbf{Cell parameters [\AA, $^\circ$]}   & &  \textbf{Isotropic $T_c$  [K]} & $<$~1                     \\
"""


def test_parse_both_layouts():
    one = B.parse_material(LAYOUT_ONE)
    assert (one["formula"], one["spacegroup"], one["mc3d_id"], one["category"]) == ("Mn", "139", "mc3d-22931", "problematic")
    assert (one["lambda"], one["omega_log_mev"], one["tc_allen_dynes_k"]) == (0.988, 21.8839, 14.854)
    assert (one["tc_iso_k"], one["tc_aniso_k"], one["tc_experimental"]) == (18.5555, 21.0548, "9.25")
    two = B.parse_material(LAYOUT_TWO)
    assert (two["formula"], two["spacegroup"], two["category"], two["tc_allen_dynes_k"]) == ("Nb3Ge", "223", "known-exp-structure", 18.0596)
    assert two["tc_iso_k"] is None and two["tc_iso_text"] == "<1"
    assert B.parse_material(r"\section{Methods} no material here") is None


def test_tiers():
    key = ("Nb", "229")
    si = {key: {"tc_iso_k": 19.7, "category": "known-exp-structure"},
          ("Mn", "139"): {"tc_iso_k": 18.6, "category": "problematic"},
          ("TiS", "194"): {"tc_iso_k": None, "category": "known-comp"}}
    b, c, d = {key, ("A", "1"), ("C", "2"), ("D", "3")}, {key, ("C", "2"), ("D", "3")}, {key, ("D", "3")}
    assert B.tier(key, si, b, c, d) == "T4"
    assert B.tier(("Mn", "139"), si, b, c, d) == "T3"
    assert B.tier(("TiS", "194"), si, b, c, d) == "T4-low"
    assert [B.tier(k, si, b, c, d) for k in (("D", "3"), ("C", "2"), ("A", "1"), ("Z", "9"))] == ["T3", "T2", "T1", "T0"]
    assert B.tier(("Nb", "221"), si, b, c, d) == "T0"   # same formula, other structure


def test_composition_key_is_order_and_multiple_free():
    pytest.importorskip("pymatgen")
    assert B.composition_key("Ba(AgAs)2") == B.composition_key("BaAg2As2") == B.composition_key("As4Ag4Ba2")


def test_fisher_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for table in ([[3, 186], [11, 1398]], [[0, 10], [5, 5]], [[7, 3], [2, 9]]):
        assert B.fisher_two_sided(*table[0], *table[1]) == pytest.approx(stats.fisher_exact(table)[1], rel=1e-9)


def test_made_metal_rule_matches_the_ranking_script():
    import run_scratchnet as R
    base = {"is_metal": "True", "theoretical": "False", "family": "Other", "element_system": "Nb-Sn"}
    variants = [base, {**base, "is_metal": "False"}, {**base, "theoretical": "True"}, {**base, "family": "Cuprates"},
                {**base, "family": "Oxides"}, {**base, "element_system": "Cu-Nb"}, {**base, "family": "Iron-based"}]
    assert [B.made_metal(v) for v in variants] == [R.made_metal(v) for v in variants]


def test_committed_summary_matches_the_join_table():
    rows = list(csv.DictReader(open(ROOT / "docs" / "bercx-made-metals-join.csv", newline="")))
    summary = json.loads((ROOT / "docs" / "bercx-made-metals-summary.json").read_text())
    top = summary["top_n"]
    for group, pick in (("top", lambda r: int(r["made_metals_rank"]) <= top), ("rest", lambda r: int(r["made_metals_rank"]) > top)):
        chosen = [r for r in rows if pick(r)]
        assert len(chosen) == summary["structure_matched"][group]
        for t in B.TIERS[:-1]:
            assert sum(r["tier"] == t for r in chosen) == summary["tier_counts"][group][t], (group, t)
    assert sum(summary["tier_counts"]["top"].values()) == top
    assert sum(summary["tier_counts"]["top"].values()) + sum(summary["tier_counts"]["rest"].values()) == summary["made_metals"]
    (a, b), (c, d) = summary["fisher_t4_top_vs_rest"]["table"]
    assert summary["fisher_t4_top_vs_rest"]["p_two_sided"] == round(B.fisher_two_sided(a, b, c, d), 5)
    assert {r["tier"] for r in rows} <= set(B.TIERS[:-1])


def test_experimental_tc_keeps_text_values():
    page = LAYOUT_ONE.replace(r"9.25~\cite{Cappelletti_1967}", r"no~\cite{x} (film)")
    assert B.parse_material(page)["tc_experimental"] == "no (film)"


def test_tier_edge_cases():
    b = c = d = set()
    def si(*records):
        return {("X", "1"): list(records)}
    assert B.tier(("X", "1"), si({"tc_iso_k": 5.0, "category": "unknown"}), b, c, d) == "T4"          # boundary is inclusive
    assert B.tier(("X", "1"), si({"tc_iso_k": 4.99, "category": "unknown"}), b, c, d) == "T4-low"
    assert B.tier(("X", "1"), si({"tc_iso_k": None, "category": "problematic"}), b, c, d) == "T3"     # no verdict at any Tc
    agree = si({"tc_iso_k": 9.0, "category": "unknown"}, {"tc_iso_k": 8.0, "category": "known-exp-formula"})
    clash = si({"tc_iso_k": 9.0, "category": "unknown"}, {"tc_iso_k": 9.0, "category": "problematic"})
    assert B.tier(("X", "1"), agree, b, c, d) == "T4"
    assert B.tier(("X", "1"), clash, b, c, d) == "T3"


def test_si_records_are_also_filed_under_table_d_space_group():
    pytest.importorskip("pymatgen")
    rows = [{"formula": "V", "spacegroup": "139", "source_id": "icsd-60614"}]
    index = B.si_index(rows, {"icsd-60614": ("V", "229")})
    assert set(index) == {("V", "139"), ("V", "229")}


def test_join_ranks_agree_with_the_committed_made_metals_list():
    committed = list(csv.DictReader(open(ROOT / "docs" / "scratchnet-candidates-boosting-top-made-metals.csv", newline="")))
    joined = [r for r in csv.DictReader(open(ROOT / "docs" / "bercx-made-metals-join.csv", newline=""))
              if int(r["made_metals_rank"]) <= len(committed)]
    assert joined
    for r in joined:
        assert committed[int(r["made_metals_rank"]) - 1]["mp_id"] == r["mp_id"], r["formula"]
