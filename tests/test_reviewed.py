import csv
import tempfile
import unittest
from pathlib import Path

from rt_superconductor.reviewed import REQUIRED_COLUMNS, load_reviewed_records, validate_row


def _row(**overrides):
    row = {column: "reviewed" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "record_id": "sc2-1",
            "formula": "YBa2Cu3O7",
            "tc_kelvin": "92",
            "pressure_gpa": "0",
            "pressure_condition": "ambient",
            "measurement_method": "resistivity and magnetization",
            "sample_form": "polycrystalline pellet",
            "synthesis_conditions": "solid-state reaction; details in source",
            "source_doi": "10.1234/example",
            "source_record_id": "supercon2:abc",
            "reviewer": "researcher",
            "review_date": "2026-10-28",
            "evidence_status": "confirmed",
            "notes": "verified against the paper",
        }
    )
    row.update(overrides)
    return row


class ReviewedTrainingTests(unittest.TestCase):
    def test_valid_ambient_record(self):
        record = validate_row(_row(), 2)
        self.assertEqual(record.tc_kelvin, 92.0)
        self.assertEqual(record.pressure_gpa, 0.0)

    def test_rejects_unconfirmed_record(self):
        with self.assertRaisesRegex(ValueError, "confirmed"):
            validate_row(_row(evidence_status="uncertain"), 2)

    def test_known_pressure_requires_value(self):
        with self.assertRaisesRegex(ValueError, "known pressure"):
            validate_row(_row(pressure_gpa="", pressure_condition="known"), 2)

    def test_directory_load_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow(_row())
                writer.writerow(_row(record_id="sc2-1"))
            with self.assertRaisesRegex(ValueError, "duplicate record_id"):
                load_reviewed_records(Path(tmp))

    def test_empty_template_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.csv"
            path.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
            self.assertEqual(load_reviewed_records(Path(tmp)), [])
