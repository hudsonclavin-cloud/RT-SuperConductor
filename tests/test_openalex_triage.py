"""Literature triage helpers: formula detection, one query per element system, and lead detection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import openalex_triage as O  # noqa: E402


def test_small_formula_finds_whole_number_compounds_only():
    assert O.small_formula({"Y": 1 / 6, "Ru": 0.5, "B": 1 / 3}) is not None
    assert O.small_formula({"Nb": 0.5, "S": 0.5}) == "NbS"
    assert O.small_formula({"In": 0.95, "Tl": 0.05}) is None          # 1:19 needs 20 atoms


def test_query_uses_element_names_and_is_shared_by_a_system():
    kind, query = O.query_for({"In": 0.95, "Tl": 0.05})
    assert kind == "alloy" and query == "superconductivity indium thallium alloy"
    assert O.query_for({"In": 0.93, "Tl": 0.07}) == (kind, query)
    assert O.query_for({"Y": 1 / 6, "Ru": 0.5, "B": 1 / 3}) == ("compound", "superconductivity boron ruthenium yttrium")


def test_lead_needs_superconductivity_and_every_element():
    assert O.is_lead({"title": "Superconductivity in the Alloy System Indium-Thallium"}, {"In", "Tl"})
    assert O.is_lead({"title": "Bulk superconductivity in the kagome metal YRu3B2"}, {"Y", "Ru", "B"})
    assert not O.is_lead({"title": "Thermal conductivity of indium-thallium alloys"}, {"In", "Tl"})
    assert not O.is_lead({"title": "Nb3Sn superconductor wires"}, {"N", "Sn"})     # N is not in Nb3Sn
    assert not O.is_lead({"title": "In this superconductor"}, {"In"})               # a bare English word


def test_abstract_text_rebuilds_the_word_order():
    assert O.abstract_text({"superconducts": [1], "It": [0], "below": [2], "4": [3], "K": [4]}) == "It superconducts below 4 K"
    assert O.abstract_text(None) == ""
