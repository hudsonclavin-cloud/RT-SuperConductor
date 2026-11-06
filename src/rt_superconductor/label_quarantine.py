"""Auditable exclusion rules for suspicious high-temperature labels.

Quarantine leaves the source observation untouched.  It only prevents the
row from fitting models while a human reviewer decides whether the label is
valid, mislabeled, or needs a corrected transcription.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


QUEUE_COLUMNS = (
    "record_id", "source_db", "mp_id", "formula", "tc_kelvin", "reason",
    "audit_status", "review_decision", "reviewer", "review_date", "notes",
    "excluded_from_training",
)


def suspected_label_reason(record: Any) -> str | None:
    """Return a review reason for high-risk labels, without changing source data."""
    source = str(getattr(record, "source_db", "") or "").strip().casefold()
    mp_id = str(getattr(record, "mp_id", "") or "").strip().casefold()
    if source != "3dsc":
        return None
    if int(getattr(record, "is_synthetic", 0) or 0) != 0 or str(
        getattr(record, "provenance", "") or ""
    ).strip().casefold() == "synthetic_demo":
        return None
    if mp_id == "mp-510625":
        return "suspected_label_error_shared_mp_structure_mp-510625"
    try:
        high_tc = float(getattr(record, "tc_kelvin", 0.0)) > 77.0
    except (TypeError, ValueError):
        high_tc = False
    formula = str(getattr(record, "formula", "") or "")
    symbols = set(re.findall(r"[A-Z][a-z]?", formula))
    if high_tc and "Cu" not in symbols:
        return "high_tc_above_77K_formula_without_Cu_needs_label_audit"
    return None


def load_quarantine_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    """Load reviewer decisions keyed by record id; absent files mean all rules apply."""
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row.get("record_id") or "").strip(): row
            for row in csv.DictReader(handle)
            if (row.get("record_id") or "").strip()
        }


def quarantine_reason(record: Any, decisions: dict[str, dict[str, str]] | None = None) -> str | None:
    """Return the active exclusion reason; a documented reviewer retain releases it."""
    reason = suspected_label_reason(record)
    if reason is None:
        return None
    decision = (decisions or {}).get(str(getattr(record, "record_id", "")), {})
    status = str(decision.get("audit_status", "pending")).strip().casefold()
    disposition = str(decision.get("review_decision", "")).strip().casefold()
    if status in {"reviewed", "complete"} and disposition in {"retain", "validated"}:
        return None
    return reason


def write_label_audit_queue(records: list[Any], output: Path) -> dict[str, int]:
    """Refresh the queue while preserving any reviewer-entered decision fields."""
    existing = load_quarantine_decisions(output)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        seen.add(record.record_id)
        reason = suspected_label_reason(record)
        prior = existing.get(record.record_id, {})
        if reason is None and not prior:
            continue
        status = prior.get("audit_status", "pending") or "pending"
        decision = prior.get("review_decision", "")
        if reason is None:
            status = "not_applicable"
            decision = decision or "synthetic_demo_not_empirical_label"
            reason = "not_applicable_synthetic_demo_not_source_measurement"
            notes = prior.get("notes", "") or (
                "Retained for queue history; this is a synthetic demo fixture, not a 3DSC source measurement."
            )
        else:
            notes = prior.get("notes", "")
        released = status.strip().casefold() in {"reviewed", "complete"} and decision.strip().casefold() in {"retain", "validated"}
        rows.append({
            "record_id": record.record_id,
            "source_db": record.source_db,
            "mp_id": record.mp_id,
            "formula": record.formula,
            "tc_kelvin": f"{record.tc_kelvin:g}",
            "reason": reason,
            "audit_status": status,
            "review_decision": decision,
            "reviewer": prior.get("reviewer", ""),
            "review_date": prior.get("review_date", ""),
            "notes": notes,
            "excluded_from_training": "false" if released or status == "not_applicable" else "true",
        })
    for record_id, prior in existing.items():
        if record_id in seen:
            continue
        # Keep review provenance even when a source export is temporarily not
        # part of the current harmonized input. Missing is not a review decision.
        rows.append({column: prior.get(column, "") for column in QUEUE_COLUMNS})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "queue_rows": len(rows),
        "active_exclusions": sum(row["excluded_from_training"] == "true" for row in rows),
        "released_after_review": sum(
            row["audit_status"].strip().casefold() in {"reviewed", "complete"}
            and row["review_decision"].strip().casefold() in {"retain", "validated"}
            for row in rows
        ),
        "not_applicable": sum(row["audit_status"] == "not_applicable" for row in rows),
        "preserved_prior_rows_not_in_input": sum(row["record_id"] not in seen for row in rows),
    }
