import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import load_harmonized, validate_row, write_harmonized


class SynthesisPressureTests(unittest.TestCase):
    def test_optional_synthesis_pressure_round_trips_separately(self):
        records = load_harmonized(ROOT / "data/training/harmonized_supercon.csv")
        updated = [replace(records[0], applied_pressure_gpa=0.000101325, synthesis_pressure_gpa=6.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "harmonized.csv"
            write_harmonized(path, updated)
            loaded = load_harmonized(path)[0]
        self.assertEqual(loaded.applied_pressure_gpa, 0.000101325)
        self.assertEqual(loaded.synthesis_pressure_gpa, 6.0)

    def test_blank_synthesis_pressure_is_unknown(self):
        with (ROOT / "data/training/harmonized_supercon.csv").open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        row["synthesis_pressure_gpa"] = ""
        self.assertIsNone(validate_row(row, 2).synthesis_pressure_gpa)

    def test_negative_synthesis_pressure_is_rejected(self):
        with (ROOT / "data/training/harmonized_supercon.csv").open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        row["synthesis_pressure_gpa"] = "-0.5"
        with self.assertRaisesRegex(ValueError, "synthesis_pressure_gpa cannot be negative"):
            validate_row(row, 2)

    def test_nonnumeric_synthesis_pressure_is_rejected(self):
        with (ROOT / "data/training/harmonized_supercon.csv").open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        row["synthesis_pressure_gpa"] = "high pressure"
        with self.assertRaisesRegex(ValueError, "synthesis_pressure_gpa must be numeric"):
            validate_row(row, 2)


if __name__ == "__main__":
    unittest.main()
