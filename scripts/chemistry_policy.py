"""Conservative, composition-aware substitution and screening rules.

Shannon radii and oxidation states are filters, not proofs that a generated
structure is chemically stable. Layer-role inference uses explicit prototype
chemistry when available and conservative stoichiometric signatures otherwise.
"""
from __future__ import annotations

import re
from functools import lru_cache

from pymatgen.core import Composition, Element, Species

MAX_RADIUS_FRACTIONAL_MISMATCH = 0.15
INSULATING_GAP_EV = 0.1
_RADIOACTIVE_SYMBOLS = {"Tc", "Pm", "Po", "Th", "U"}
_ROMAN = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}


def active_layer_elements(parent_formula: str) -> set[str]:
    """Infer chemically diagnostic layer elements for the requested prototypes.

    The transition metal and its bonded pnictogen/chalcogen/oxide partner are
    held fixed. Without a CIF this is a cautious formula-level proxy.
    """
    elements = set(Composition(parent_formula).get_el_amt_dict())
    if "Fe" in elements and "As" in elements:
        return {"Fe", "As"}
    if "Fe" in elements and elements.intersection({"S", "Se", "Te"}):
        return {"Fe", *elements.intersection({"S", "Se", "Te"})}
    if "Cu" in elements and "O" in elements:
        return {"Cu", "O"}
    if "Ni" in elements and "As" in elements:
        return {"Ni", "As"}
    return set()


def _family_definers(family: str | None) -> set[str]:
    name = (family or "").casefold()
    protected: set[str] = set()
    if "ferrite" in name or "iron-based" in name or "iron based" in name:
        protected.add("Fe")
    if "cuprate" in name:
        protected.add("Cu")
    return protected


def _oxidation_states(element: Element) -> set[int]:
    return {int(value) for value in element.common_oxidation_states if int(value) != 0}


def _parent_oxidation_state(parent: Composition, symbol: str) -> int | None:
    try:
        guesses = parent.oxi_state_guesses(max_sites=-1)
    except (ValueError, TypeError):
        return None
    for guess in guesses:
        value = guess.get(symbol)
        if value is not None and float(value).is_integer():
            return int(value)
    return None


def _shared_role_states(old: Element, new: Element, parent: Composition) -> set[int]:
    shared = _oxidation_states(old) & _oxidation_states(new)
    old_parent_state = _parent_oxidation_state(parent, old.symbol)
    if old_parent_state is not None and old_parent_state in shared:
        shared = {old_parent_state}
    # A mixed-sign shared state does not identify the crystallographic site.
    signs = {1 if state > 0 else -1 for state in shared}
    return shared if len(signs) == 1 else set()


def _radius_pair(old: Element, new: Element, oxidation_state: int, coordination_number: int | None) -> tuple[float, float] | None:
    left, right = Species(old.symbol, oxidation_state), Species(new.symbol, oxidation_state)
    coordinations = [coordination_number] if coordination_number is not None else list(_ROMAN)
    for coordination in coordinations:
        roman = _ROMAN.get(int(coordination))
        if roman is None:
            continue
        try:
            r_old = float(left.get_shannon_radius(roman))
            r_new = float(right.get_shannon_radius(roman))
        except (KeyError, ValueError, TypeError):
            continue
        if r_old > 0 and r_new > 0:
            return r_old, r_new
    return None


def _is_ionic(parent: Composition) -> bool:
    elements = parent.elements
    # Do not mistake negative formal states occasionally assigned to metals in
    # intermetallics (for example Sn in Nb3Sn) for an ionic anion sublattice.
    has_cation = any(el.is_metal and any(state > 0 for state in _oxidation_states(el)) for el in elements)
    has_anion = any(not el.is_metal and any(state < 0 for state in _oxidation_states(el)) for el in elements)
    return has_cation and has_anion


@lru_cache(maxsize=2048)
def uniform_site_coordination(structure_path: str, element: str) -> int | None:
    """Return one unambiguous CrystalNN coordination for an element in a CIF."""
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.analysis.local_env import CrystalNN
        structure = CifParser(structure_path, occupancy_tolerance=1.1).parse_structures(primitive=False)[0]
        indices = [i for i, site in enumerate(structure) if element in site.species.as_dict()]
        if not indices:
            return None
        calculator = CrystalNN()
        values = []
        for index in indices:
            count = calculator.get_cn(structure, index)
            if count is None or abs(float(count) - round(float(count))) > 0.05:
                return None
            values.append(int(round(float(count))))
        return values[0] if values and len(set(values)) == 1 else None
    except Exception:
        return None


def substitution_rejection(
    old: str,
    new: str,
    parent_formula: str,
    candidate_formula: str,
    parent_gap: float | None = None,
    *,
    family: str | None = None,
    coordination_number: int | None = None,
) -> str | None:
    """Return the first named rejection rule, or ``None`` if basic screens pass."""
    if old == "H" or new == "H":
        return "hydrogen"
    parent = Composition(parent_formula)
    old_el, new_el = Element(old), Element(new)
    if old in _family_definers(family):
        return "family_defining_element"
    layer = active_layer_elements(parent_formula)
    if old in layer:
        return "active_layer_element"
    if parent_gap is not None and float(parent_gap) > INSULATING_GAP_EV:
        return "insulating_parent"
    if not (old_el.is_metal and new_el.is_metal) and not (old_el.is_metalloid and new_el.is_metalloid) and not (not old_el.is_metal and not old_el.is_metalloid and not new_el.is_metal and not new_el.is_metalloid):
        return "site_role_mismatch"

    shared = _shared_role_states(old_el, new_el, parent)
    if not shared:
        return "no_common_oxidation_state"
    # Do not move an oxide's O site to a different chalcogen/pnictogen row.
    if old == "O" and new_el.group == 16 and new != "O" and "O" in parent_formula and any(el.is_metal for el in parent.elements):
        return "oxide_anion_row_swap"

    if _is_ionic(parent):
        try:
            overrides = {el.symbol: sorted(_oxidation_states(el)) for el in Composition(candidate_formula).elements}
            if not Composition(candidate_formula).oxi_state_guesses(oxi_states_override=overrides, max_sites=-1):
                return "ionic_charge_imbalance"
        except (ValueError, TypeError):
            return "ionic_charge_imbalance"

    radii = [_radius_pair(old_el, new_el, state, coordination_number) for state in sorted(shared)]
    radii = [pair for pair in radii if pair is not None]
    if not radii:
        return "ionic_radii_unavailable"
    if not any(abs(left - right) / left <= MAX_RADIUS_FRACTIONAL_MISMATCH for left, right in radii):
        return "ionic_radius_mismatch"
    return None


def radioactive_elements(formula: str) -> list[str]:
    """List project-flagged radioactive elements (Tc, Pm, Po, Th, U, Np onward)."""
    present = {el.symbol for el in Composition(formula).elements}
    heavy_actinides = {el.symbol for el in Element if el.Z >= 93 and el.Z <= 103}
    return sorted(present & (_RADIOACTIVE_SYMBOLS | heavy_actinides), key=lambda symbol: Element(symbol).Z)


def is_binary_transition_metal_oxide_or_halide(formula: str) -> bool:
    """Conservatively flag binary d-block oxides and halides as insulator-like."""
    composition = Composition(formula)
    elements = composition.elements
    if len(elements) != 2:
        return False
    anions = {"O", "F", "Cl", "Br", "I"}
    transition_metals = {el.symbol for el in elements if el.is_transition_metal}
    return len(transition_metals) == 1 and bool({el.symbol for el in elements} & anions)


def insulator_rejection(formula: str, band_gap_ev: float | None = None, gap_method: str | None = None) -> str | None:
    """Return an auditable composition or MP method-specific insulator flag."""
    if is_binary_transition_metal_oxide_or_halide(formula):
        return "binary_transition_metal_oxide_or_halide"
    method = (gap_method or "").casefold().replace(" ", "")
    if band_gap_ev is not None and float(band_gap_ev) > INSULATING_GAP_EV:
        if "r2scan" in method:
            return "mp_r2scan_gap_gt_0p1_ev"
        if "gga+u" in method or "gga_u" in method or "ggau" in method:
            return "mp_gga_u_gap_gt_0p1_ev"
    return None
