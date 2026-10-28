"""Formula-level chemistry descriptors with explicit solid-state limits.

These descriptors describe neutral atoms combined according to a formula unit.
They are useful model inputs, but they are not a quantum-mechanical charge
analysis. Lewis dots, oxidation states, and sigma/pi bonds in a crystal require
structure and electronic-density information and remain explicitly unknown.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


PERIODIC_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm "
    "Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(PERIODIC_SYMBOLS, start=1)}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")

# Madelung order gives a reproducible neutral-atom proxy for shell/orbital
# occupancy. It preserves the information needed by the descriptor while
# avoiding claims about oxidation state or crystal charge transfer.
ORBITAL_ORDER = (
    ("1s", 1, 2),
    ("2s", 2, 2),
    ("2p", 2, 6),
    ("3s", 3, 2),
    ("3p", 3, 6),
    ("4s", 4, 2),
    ("3d", 3, 10),
    ("4p", 4, 6),
    ("5s", 5, 2),
    ("4d", 4, 10),
    ("5p", 5, 6),
    ("6s", 6, 2),
    ("4f", 4, 14),
    ("5d", 5, 10),
    ("6p", 6, 6),
    ("7s", 7, 2),
    ("5f", 5, 14),
    ("6d", 6, 10),
    ("7p", 7, 6),
)
# Neutral-atom outer-shell counts used only for the Lewis-dot proxy. Transition
# metal Lewis structures are not literal crystal bonds; the overrides avoid
# presenting the simple Madelung fill as a chemically exact dot count.
LEWIS_OUTER_ELECTRONS = {
    "Cr": 1,
    "Cu": 1,
    "Nb": 1,
    "Mo": 1,
    "Ru": 1,
    "Rh": 1,
    "Pd": 0,
    "Ag": 1,
    "Pt": 1,
    "Au": 1,
}


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple normalized elemental formula into atom counts."""

    text = "".join(formula.split())
    amounts: dict[str, float] = defaultdict(float)
    position = 0
    for match in FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            raise ValueError(f"unsupported formula syntax: {formula!r}")
        symbol = match.group(1)
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"unknown element {symbol!r} in {formula!r}")
        amounts[symbol] += float(match.group(2) or 1.0)
        position = match.end()
    if position != len(text) or not amounts:
        raise ValueError(f"unsupported formula syntax: {formula!r}")
    return dict(amounts)


def neutral_atom_occupancy(atomic_number: int) -> tuple[dict[str, int], tuple[int, ...]]:
    """Return orbital and principal-shell electron counts for a neutral atom."""

    if not 1 <= atomic_number <= len(PERIODIC_SYMBOLS):
        raise ValueError(f"atomic number out of range: {atomic_number}")
    remaining = atomic_number
    orbitals: dict[str, int] = {}
    shells = [0] * 7
    for label, shell, capacity in ORBITAL_ORDER:
        count = min(remaining, capacity)
        orbitals[label] = count
        shells[shell - 1] += count
        remaining -= count
        if remaining == 0:
            break
    return orbitals, tuple(shells)


def _weighted(values: Iterable[tuple[float, float]]) -> float:
    values = list(values)
    total = sum(weight for _, weight in values)
    return sum(value * weight for value, weight in values) / total


def descriptor_names() -> list[str]:
    names = [
        "descriptor_total_atoms",
        "descriptor_element_count",
        "descriptor_avg_atomic_number",
        "descriptor_avg_electrons_per_atom",
        "descriptor_avg_outer_shell_electrons",
        "descriptor_avg_s_electrons",
        "descriptor_avg_p_electrons",
        "descriptor_avg_d_electrons",
        "descriptor_avg_f_electrons",
    ]
    names.extend(f"descriptor_shell_{shell}_fill_pct" for shell in range(1, 8))
    names.extend(f"descriptor_shell_{shell}_electrons" for shell in range(1, 8))
    return names


def formula_descriptor_vector(formula: str) -> tuple[list[float], dict[str, object]]:
    """Return numeric formula descriptors and an auditable metadata record."""

    amounts = parse_formula(formula)
    total_atoms = sum(amounts.values())
    element_data = {}
    for symbol in amounts:
        z = ATOMIC_NUMBERS[symbol]
        orbitals, shells = neutral_atom_occupancy(z)
        element_data[symbol] = {"atomic_number": z, "orbitals": orbitals, "shells": shells}
    avg_z = _weighted((element_data[s]["atomic_number"], amounts[s]) for s in amounts)
    avg_electrons = _weighted((float(element_data[s]["atomic_number"]), amounts[s]) for s in amounts)
    avg_outer = _weighted(
        (float(next(value for value in reversed(element_data[s]["shells"]) if value)), amounts[s])
        for s in amounts
    )
    avg_s = _weighted((sum(v for k, v in element_data[s]["orbitals"].items() if k.endswith("s")), amounts[s]) for s in amounts)
    avg_p = _weighted((sum(v for k, v in element_data[s]["orbitals"].items() if k.endswith("p")), amounts[s]) for s in amounts)
    avg_d = _weighted((sum(v for k, v in element_data[s]["orbitals"].items() if k.endswith("d")), amounts[s]) for s in amounts)
    avg_f = _weighted((sum(v for k, v in element_data[s]["orbitals"].items() if k.endswith("f")), amounts[s]) for s in amounts)
    shell_electrons = [
        _weighted((float(element_data[s]["shells"][index]), amounts[s]) for s in amounts)
        for index in range(7)
    ]
    shell_capacities = [2 * shell * shell for shell in range(1, 8)]
    shell_fill = [100.0 * electrons / capacity for electrons, capacity in zip(shell_electrons, shell_capacities)]
    vector = [
        total_atoms,
        float(len(amounts)),
        avg_z,
        avg_electrons,
        avg_outer,
        avg_s,
        avg_p,
        avg_d,
        avg_f,
        *shell_fill,
        *shell_electrons,
    ]
    configurations = []
    lewis = []
    for symbol in sorted(amounts):
        info = element_data[symbol]
        configuration = " ".join(f"{label}{count}" for label, count in info["orbitals"].items() if count)
        outer_shell = max(i for i, value in enumerate(info["shells"], start=1) if value)
        outer_electrons = LEWIS_OUTER_ELECTRONS.get(symbol, info["shells"][outer_shell - 1])
        configurations.append(f"{symbol}:{configuration}")
        lewis.append(f"{symbol}:{min(outer_electrons, 8)}")
    metadata = {
        "descriptor_status": "neutral_atom_formula_proxy",
        "elemental_electron_configurations": ";".join(configurations),
        "lewis_dot_proxy_outer_electrons": ";".join(lewis),
        "lewis_structure_status": "proxy_only;crystal_bonding_not_inferred",
        "sigma_bond_count": None,
        "pi_bond_count": None,
        "bond_descriptor_status": "unknown_requires_crystal_structure_or_density",
        "oxidation_state_status": "unknown_requires_chemical_or_structure_analysis",
        "shell_percentage_definition": "neutral_atom shell electrons divided by 2n^2 capacity; stoichiometric average",
    }
    return vector, metadata


def formula_descriptor_record(formula: str) -> dict[str, object]:
    vector, metadata = formula_descriptor_vector(formula)
    return dict(zip(descriptor_names(), vector), **metadata)
