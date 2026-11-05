"""Pool-specific gates layered on the repository's shared chemistry policy."""
from __future__ import annotations

import math

from pymatgen.core import Composition, Element

from chemistry_policy import insulator_rejection, radioactive_elements as main_radioactive_elements


_ADDITIONAL_RADIOACTIVE_SYMBOLS = {"Ac", "Ra", "Pa"}
_HEAVY_ELEMENTS = {element.symbol for element in Element if element.Z >= 93}


def radioactive_elements(formula: str) -> list[str]:
    """Use main's radioactive flags and extend them to all requested symbols."""
    present = {element.symbol for element in Composition(formula).elements}
    excluded = set(main_radioactive_elements(formula))
    excluded.update(present & (_ADDITIONAL_RADIOACTIVE_SYMBOLS | _HEAVY_ELEMENTS))
    return sorted(excluded, key=lambda symbol: Element(symbol).Z)


def polyanion_salt_rejection(formula: str) -> str | None:
    """Reject common carbonate, sulfate, hydroxide, nitrate, and phosphate salts."""
    composition = Composition(formula)
    amounts = {symbol: float(amount) for symbol, amount in composition.get_el_amt_dict().items()}
    if not any(element.is_metal for element in composition.elements):
        return None
    oxygen = amounts.get("O", 0.0)
    if amounts.get("C", 0.0) > 0 and oxygen >= 3 * amounts["C"]:
        return "carbonate_salt"
    if amounts.get("S", 0.0) > 0 and oxygen >= 4 * amounts["S"]:
        return "sulfate_salt"
    if amounts.get("O", 0.0) > 0 and amounts.get("H", 0.0) >= amounts["O"]:
        return "hydroxide_salt"
    if amounts.get("N", 0.0) > 0 and oxygen >= 3 * amounts["N"]:
        return "nitrate_salt"
    if amounts.get("P", 0.0) > 0 and oxygen >= 3 * amounts["P"]:
        return "phosphate_salt"
    return None


def alkali_alkaline_earth_transition_metal_fluoride_rejection(formula: str) -> str | None:
    """Reject fluorides combining an alkali/alkaline-earth and transition metal."""
    elements = Composition(formula).elements
    has_s_block_metal = any(element.group in (1, 2) for element in elements if element.symbol != "H")
    has_transition_metal = any(element.is_transition_metal for element in elements)
    if any(element.symbol == "F" for element in elements) and has_s_block_metal and has_transition_metal:
        return "alkali_alkaline_earth_transition_metal_fluoride"
    return None


def undoped_bacuO2_parent_rejection(formula: str) -> str | None:
    """Apply the explicit narrow BaCuO2 parent exclusion requested for v2."""
    amounts = {
        element.symbol: float(amount)
        for element, amount in Composition(formula).element_composition.items()
    }
    if set(amounts) == {"Ba", "Cu", "O"}:
        scale = amounts["Ba"]
        if scale > 0 and math.isclose(amounts["Cu"] / scale, 1.0, abs_tol=1e-8) and math.isclose(
            amounts["O"] / scale, 2.0, abs_tol=1e-8
        ):
            return "undoped_bacuO2_parent_composition"
    return None


def untested_pool_composition_rejection(formula: str) -> str | None:
    """Apply shared main policy and additional conservative pool-only screens."""
    excluded = radioactive_elements(formula)
    if excluded:
        return "radioactive_element:" + ";".join(excluded)
    for check in (
        polyanion_salt_rejection,
        alkali_alkaline_earth_transition_metal_fluoride_rejection,
        undoped_bacuO2_parent_rejection,
    ):
        reason = check(formula)
        if reason:
            return reason
    return insulator_rejection(formula)


def dos_ef_rejection(dos_ef_states_ev_per_atom: float | None) -> str | None:
    """Require a finite MP DOS(Ef) strictly above 0.1 states/eV/atom."""
    if dos_ef_states_ev_per_atom is None:
        return "static_or_uniform_dos_missing"
    try:
        value = float(dos_ef_states_ev_per_atom)
    except (TypeError, ValueError):
        return "static_or_uniform_dos_invalid"
    if not math.isfinite(value) or value <= 0.1:
        return "dos_ef_le_0.1_states_per_ev_per_atom"
    return None


def untested_pool_candidate_rejection(
    formula: str, dos_ef_states_ev_per_atom: float | None
) -> str | None:
    """Require shared chemistry clearance and qualifying static/uniform DOS."""
    reason = untested_pool_composition_rejection(formula)
    if reason:
        return reason
    return dos_ef_rejection(dos_ef_states_ev_per_atom)
