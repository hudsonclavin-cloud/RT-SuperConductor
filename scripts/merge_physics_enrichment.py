#!/usr/bin/env python3
"""Merge reference properties without replacing geometry or changing row identity."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    identifiers = [row.get("record_id", "") for row in rows]
    if not fields or "record_id" not in fields or any(not value for value in identifiers):
        raise ValueError("Every feature row needs a record_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate record_id in " + str(path))
    return fields, rows

def merge(base, enrichment, output, reference_ids=None):
    base_fields, rows = read_table(base)
    source_fields, source_rows = read_table(enrichment)
    source = {row["record_id"]: row for row in source_rows}
    if set(source) != {row["record_id"] for row in rows}:
        raise ValueError("Enrichment cohort differs from the geometry cohort")
    properties = [name for name in source_fields if name.startswith(("mp_", "orbital_", "element_orbital_")) or name == "reference_mp_id"]
    for row in rows:
        new = source[row["record_id"]]
        if reference_ids is not None:
            expected = reference_ids.get(row["record_id"], "") or ""
            actual = new.get("reference_mp_id", "") or ""
            if actual != expected:
                raise ValueError("Reference MP ID mismatch for " + row["record_id"])
        # Current API MVC lookups have produced unrelated MP compounds. Raw
        # audit artifacts must not turn those unresolved legacy IDs into data.
        numeric_properties = ("mp_bulk_modulus_gpa", "mp_dos_ef_states_ev_cell", "orbital_s_fraction", "orbital_p_fraction", "orbital_d_fraction", "orbital_f_fraction")
        if str(new.get("reference_mp_id", "")).startswith("mvc-") and any(new.get(name) not in (None, "") for name in numeric_properties):
            raise ValueError("Unresolved MVC reference has unvalidated numeric properties: " + row["record_id"])
        for name in properties:
            row[name] = new.get(name, "")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(base_fields + properties)))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "rows": len(rows), "merged_columns": properties,
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "enrichment_sha256": hashlib.sha256(enrichment.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "reference_ids_checked": reference_ids is not None,
        "method": "Exact record_id cohort; import only reference-property columns; preserve geometry",
        "interpretation": "MP properties belong to reference calculations, not measured candidate samples",
    }
    output.with_suffix(".merge.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--enrichment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    args = parser.parse_args()
    from rt_superconductor.harmonized import load_harmonized
    references = {record.record_id: record.mp_id for record in load_harmonized(args.input)}
    print(json.dumps(merge(args.base, args.enrichment, args.output, references), indent=2))
