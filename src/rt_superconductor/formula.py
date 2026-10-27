"""Conservative chemical-formula canonicalization for duplicate review."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


# Used only to reject obvious non-element tokens. Formula syntax remains
# intentionally conservative; unsupported structures go to human review.
ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db "
    "Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    .split()
)
TOKEN = re.compile(r"(?P<element>[A-Z][a-z]?)(?P<count>(?:\d+(?:\.\d*)?|\.\d+)?)")


@dataclass(frozen=True, slots=True)
class FormulaResult:
    original: str
    canonical: str | None
    status: str
    reason: str | None = None


def _format_count(value: Decimal) -> str:
    if value == 1:
        return ""
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def canonicalize_formula(formula: str) -> FormulaResult:
    """Return a stable key for simple, explicit elemental formulas.

    Parentheses, hydrates, charge annotations, variables, and malformed tokens
    are rejected instead of guessed. This protects the review queue from false
    equivalences; unsupported records remain available with their original text.
    """

    original = formula
    text = "".join(formula.split())
    if not text:
        return FormulaResult(original, None, "unparseable", "empty formula")
    if any(character in text for character in "()[]{}·+−-/_"):
        return FormulaResult(original, None, "unsupported", "grouping, hydrate, charge, or separator syntax")

    amounts: dict[str, Decimal] = {}
    position = 0
    for match in TOKEN.finditer(text):
        if match.start() != position:
            return FormulaResult(original, None, "unparseable", "unexpected formula token")
        element = match.group("element")
        if element not in ELEMENTS:
            return FormulaResult(original, None, "unparseable", f"unknown element token: {element}")
        count_text = match.group("count")
        try:
            count = Decimal(count_text) if count_text else Decimal(1)
        except InvalidOperation:
            return FormulaResult(original, None, "unparseable", f"invalid count for {element}")
        if count < 0:
            return FormulaResult(original, None, "unparseable", f"negative count for {element}")
        if count > 0:
            amounts[element] = amounts.get(element, Decimal(0)) + count
        position = match.end()
    if position != len(text) or not amounts:
        return FormulaResult(original, None, "unparseable", "formula did not contain elemental tokens")

    ordered = sorted(amounts)
    if "C" in amounts:
        ordered = ["C"] + (["H"] if "H" in amounts else []) + [
            element for element in ordered if element not in {"C", "H"}
        ]
    canonical = "".join(element + _format_count(amounts[element]) for element in ordered)
    return FormulaResult(original, canonical, "parsed")
