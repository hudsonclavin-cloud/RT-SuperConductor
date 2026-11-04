"""Conservative, auditable Materials Project composition lookups.

Materials Project's formula search is a retrieval operation, not a proof that
the returned composition is the requested composition.  This module keeps the
retrieval and the validation decision together so callers cannot accidentally
use an unvalidated fuzzy match.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from time import sleep
from typing import Any, Iterable, Mapping, Sequence

from .formula import ELEMENTS, TOKEN


_AMBIGUOUS_MARKERS = re.compile(r"(?:[x-z]|delta|δ|Δ|mixed|phase|sample|film|powder)", re.IGNORECASE)
_ALL_CAPS_LABEL = re.compile(r"^[A-Z]{3,}$")
_UNSAFE_CHARS = set("()[]{}+−–-/_·,;:@=<>?")


@dataclass(frozen=True, slots=True)
class FormulaParse:
    """Result of parsing a fixed-composition formula."""

    original: str
    normalized: str | None
    signature: tuple[tuple[str, int], ...] | None
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.signature is not None:
            result["signature"] = [list(item) for item in self.signature]
        return result


def _format_count(value: int) -> str:
    return "" if value == 1 else str(value)


def _signature(amounts: Mapping[str, Decimal]) -> tuple[tuple[str, int], ...]:
    """Return a scale-invariant integer composition signature.

    Decimal amounts are converted to exact rational numbers before reducing.
    Thus ``Ba0.5K0.5BiO3`` and ``BaKO6`` are *not* considered equal merely by
    element set, while equivalent fractional representations are equivalent.
    """

    fractions = {element: Fraction(value) for element, value in amounts.items()}
    common_denominator = reduce(lcm, (value.denominator for value in fractions.values()), 1)
    integer_amounts = {
        element: value.numerator * (common_denominator // value.denominator)
        for element, value in fractions.items()
    }
    divisor = reduce(gcd, integer_amounts.values())
    return tuple(sorted((element, amount // divisor) for element, amount in integer_amounts.items()))


def _canonical_from_signature(signature: tuple[tuple[str, int], ...]) -> str:
    elements = [element for element, _ in signature]
    if "C" in elements:
        ordered = ["C"] + (["H"] if "H" in elements else []) + [
            element for element in elements if element not in {"C", "H"}
        ]
        amounts = dict(signature)
        return "".join(element + _format_count(amounts[element]) for element in ordered)
    return "".join(element + _format_count(amount) for element, amount in signature)


def normalize_fixed_formula(formula: str | None) -> FormulaParse:
    """Parse only an explicit, fixed elemental composition.

    The parser intentionally rejects variables, mixed phases, grouping and
    sample labels.  Rejection is preferable to guessing because the result is
    used to validate an external database match.  A successful result is
    normalized to a reduced integer composition and includes a comparison
    signature independent of element ordering and whitespace.
    """

    original = "" if formula is None else str(formula)
    text = "".join(original.split())
    if not text:
        return FormulaParse(original, None, None, "ambiguous_input", "empty formula")
    if _ALL_CAPS_LABEL.fullmatch(text):
        return FormulaParse(original, None, None, "ambiguous_input", "all-uppercase sample label or abbreviation")
    if any(marker in text.casefold() for marker in ("δ", "delta")):
        return FormulaParse(original, None, None, "ambiguous_input", "variable stoichiometry marker")
    if _AMBIGUOUS_MARKERS.search(text):
        # Lower-case x/y/z are variables; the uppercase Y is a valid element.
        if re.search(r"[xyz]", text) or re.search(r"(?:mixed|phase|sample|film|powder)", text, re.IGNORECASE):
            return FormulaParse(original, None, None, "ambiguous_input", "variable stoichiometry or sample label")
    if any(character in text for character in _UNSAFE_CHARS):
        return FormulaParse(original, None, None, "ambiguous_input", "grouping, charge, mixed-phase, or separator syntax")

    amounts: dict[str, Decimal] = {}
    position = 0
    for match in TOKEN.finditer(text):
        if match.start() != position:
            return FormulaParse(original, None, None, "ambiguous_input", "unexpected or uncertain formula token")
        element = match.group("element")
        if element not in ELEMENTS:
            return FormulaParse(original, None, None, "ambiguous_input", f"unknown element token: {element}")
        count_text = match.group("count")
        try:
            count = Decimal(count_text) if count_text else Decimal(1)
        except InvalidOperation:
            return FormulaParse(original, None, None, "ambiguous_input", f"invalid count for {element}")
        if not count.is_finite() or count <= 0:
            return FormulaParse(original, None, None, "ambiguous_input", f"non-positive or non-finite count for {element}")
        amounts[element] = amounts.get(element, Decimal(0)) + count
        position = match.end()
    if position != len(text) or not amounts:
        return FormulaParse(original, None, None, "ambiguous_input", "formula did not contain only elemental tokens")

    signature = _signature(amounts)
    return FormulaParse(original, _canonical_from_signature(signature), signature, "parsed")


def _value(document: Any, name: str, default: Any = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(name, default)
    return getattr(document, name, default)


def _spacegroup_number(document: Any) -> int | None:
    direct = _value(document, "spacegroup_number")
    symmetry = _value(document, "symmetry")
    nested = _value(symmetry, "number") if symmetry is not None else None
    value = direct if direct is not None else nested
    return int(value) if value is not None else None


def _formula_from_document(document: Any) -> str | None:
    for field in ("formula_pretty", "composition_reduced", "formula", "composition"):
        value = _value(document, field)
        if value is None:
            continue
        if isinstance(value, Mapping):
            # A Composition-like mapping is converted to a formula only when
            # its amounts are explicit numbers; no oxidation-state inference.
            pieces: list[str] = []
            for element, amount in value.items():
                pieces.append(f"{element}{amount if amount != 1 else ''}")
            value = "".join(pieces)
        return str(value)
    return None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _redact_error(error: BaseException, secrets: Sequence[str] = ()) -> str:
    detail = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[redacted]")
    # Avoid retaining a credential-like query parameter in an artifact even if
    # a client library includes it in its exception text.
    return re.sub(r"(?i)(api[_-]?key|token)=([^&\s]+)", r"\1=[redacted]", detail)


def _candidate_record(document: Any, query: FormulaParse) -> dict[str, Any]:
    source_formula = _formula_from_document(document)
    parsed = normalize_fixed_formula(source_formula)
    material_id = _value(document, "material_id", _value(document, "mp_id", ""))
    record: dict[str, Any] = {
        "material_id": "" if material_id is None else str(material_id),
        "formula": source_formula or "",
        "normalized_formula": parsed.normalized,
        "energy_above_hull": _optional_float(_value(document, "energy_above_hull")),
        "spacegroup_number": _spacegroup_number(document),
        "band_gap_ev": _optional_float(_value(document, "band_gap")),
        # Do not infer a functional from a generic MP gap. The method is only
        # populated when the API document explicitly carries it.
        "band_gap_method": _value(document, "band_gap_method", _value(document, "run_type", _value(document, "calc_type"))),
        "is_metal": _value(document, "is_metal"),
        "formation_energy_per_atom": _optional_float(_value(document, "formation_energy_per_atom")),
        "total_magnetization_muB": _optional_float(_value(document, "total_magnetization")),
        "density_g_cm3": _optional_float(_value(document, "density")),
        "volume_angstrom3": _optional_float(_value(document, "volume")),
        "validation_status": "rejected_unresolved_returned_formula",
        "validation_detail": parsed.reason or "returned composition is unavailable or unresolved",
    }
    if parsed.status == "parsed" and parsed.signature == query.signature:
        record["validation_status"] = "validated"
        record["validation_detail"] = "returned reduced composition exactly matches the query signature"
    elif parsed.status == "parsed":
        record["validation_status"] = "rejected_mismatch"
        record["validation_detail"] = (
            f"returned signature {parsed.signature} differs from query signature {query.signature}"
        )
    return record


def audit_materials_project_queries(
    rows: Iterable[Mapping[str, Any]],
    *,
    summary_client: Any | None,
    query_limit: int | None = None,
    error_detail: str | None = None,
    error_secrets: Sequence[str] = (),
    request_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return one auditable record per normalized query group.

    ``summary_client`` is an injectable object exposing ``search``.  Passing
    ``None`` records fixed formulas as API errors without making network calls,
    which is the safe degraded behavior when no credential is available.
    """

    groups: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        source_id = str(row.get("record_id", ""))
        source_formula = str(row.get("formula", ""))
        parsed = normalize_fixed_formula(source_formula)
        if parsed.status != "parsed":
            skipped += 1
            queries.append({
                "source_record_ids": [source_id],
                "original_formulas": [source_formula],
                "normalized_formula": None,
                "query_status": "ambiguous_input",
                "validation_status": "not_applicable",
                "returned_candidates": [],
                "error_detail": parsed.reason,
            })
            continue
        group = groups.setdefault(parsed.normalized or "", {
            "parsed": parsed,
            "source_record_ids": [],
            "original_formulas": [],
        })
        group["source_record_ids"].append(source_id)
        group["original_formulas"].append(source_formula)

    unique_groups = list(groups.values())
    if query_limit is not None:
        query_groups = unique_groups[: max(0, query_limit)]
        for group in unique_groups[max(0, query_limit):]:
            queries.append({
                "source_record_ids": group["source_record_ids"],
                "original_formulas": group["original_formulas"],
                "normalized_formula": group["parsed"].normalized,
                "query_status": "not_queried",
                "validation_status": "not_applicable",
                "returned_candidates": [],
                "error_detail": "screen query limit reached before this formula",
            })
    else:
        query_groups = unique_groups

    queried_formulas: list[str] = []
    for group in query_groups:
        parsed: FormulaParse = group["parsed"]
        normalized = parsed.normalized or ""
        queried_formulas.append(normalized)
        base = {
            "source_record_ids": group["source_record_ids"],
            "original_formulas": group["original_formulas"],
            "normalized_formula": normalized,
            "query_status": "query_error",
            "validation_status": "not_applicable",
            "returned_candidates": [],
            "error_detail": None,
        }
        if summary_client is None:
            base["error_detail"] = error_detail or "Materials Project summary client is unavailable"
            queries.append(base)
            continue
        try:
            documents = summary_client.search(
                formula=normalized,
                fields=[
                    "material_id", "formula_pretty", "composition_reduced", "energy_above_hull",
                    "symmetry", "band_gap", "is_metal", "formation_energy_per_atom",
                    "total_magnetization", "density", "volume", "origins",
                ],
            )
            candidates = [_candidate_record(document, parsed) for document in list(documents or [])]
            base["returned_candidates"] = candidates
            valid = [candidate for candidate in candidates if candidate["validation_status"] == "validated"]
            if valid:
                base["query_status"] = "matched"
                base["validation_status"] = "validated"
            elif candidates:
                base["query_status"] = "no_match"
                base["validation_status"] = "rejected_mismatch"
                base["error_detail"] = "all returned structures failed composition validation"
            else:
                base["query_status"] = "no_match"
            if request_delay_seconds:
                sleep(request_delay_seconds)
        except Exception as error:  # one bad formula must not discard the batch
            base["error_detail"] = _redact_error(error, error_secrets)
        queries.append(base)

    all_candidates = [candidate for query in queries for candidate in query["returned_candidates"]]
    validated = [candidate for candidate in all_candidates if candidate["validation_status"] == "validated"]
    rejected = [candidate for candidate in all_candidates if candidate["validation_status"] != "validated"]
    api_errors = [query for query in queries if query["query_status"] == "query_error"]
    return {
        "status": "ready" if summary_client is not None else "degraded_no_mp_api_key",
        "query_records": queries,
        "queried_formulas": queried_formulas,
        "queried_formula_count": len(queried_formulas),
        "unique_fixed_formula_count": len(unique_groups),
        "skipped_ambiguous_formula_count": skipped,
        "safe_match_count": len(validated),
        "matched_query_count": sum(query["query_status"] == "matched" for query in queries),
        "no_match_count": sum(query["query_status"] == "no_match" for query in queries),
        "rejected_mismatch_count": len(rejected),
        "api_error_count": len(api_errors),
        "not_queried_count": sum(query["query_status"] == "not_queried" for query in queries),
        "ambiguous_query_count": sum(query["query_status"] == "ambiguous_input" for query in queries),
    }


def validated_candidates(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Index only composition-validated candidates by normalized formula."""

    matches: dict[str, list[dict[str, Any]]] = {}
    for query in report.get("query_records", []):
        formula = query.get("normalized_formula")
        if not formula:
            continue
        safe = [candidate for candidate in query.get("returned_candidates", []) if candidate.get("validation_status") == "validated"]
        if safe:
            matches[formula] = safe
    return matches
