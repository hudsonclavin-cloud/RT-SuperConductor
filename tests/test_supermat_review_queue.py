import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_supermat_review_queue import build_queue


class SuperMatReviewQueueTests(unittest.TestCase):
    def test_pending_rows_keep_unknown_pressure_and_are_never_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("id", "material", "tcValue", "pressure", "me_method", "filename"))
                writer.writeheader()
                writer.writerow({"id": "1", "material": "YBa2Cu3O7", "tcValue": "92 K", "pressure": "ambient", "me_method": "resistivity", "filename": "paper.xml"})
                writer.writerow({"id": "2", "material": "H3S", "tcValue": "203 K", "pressure": "150 GPa", "me_method": "resistivity", "filename": "hydride.xml"})
                writer.writerow({"id": "3", "material": "unknown", "tcValue": "85 K", "pressure": "10-20 GPa", "me_method": "", "filename": "range.xml"})
            output, manifest = root / "queue.csv", root / "manifest.json"
            report = build_queue(source, output, manifest)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(report["candidate_rows"], 3)
            self.assertFalse(report["training_eligible"])
            self.assertEqual(rows[0]["pressure_bucket"], "within_provisional_engineering_ceiling")
            self.assertEqual(rows[1]["pressure_bucket"], "high_pressure_control")
            self.assertEqual(rows[2]["pressure_gpa"], "")
            self.assertEqual(rows[2]["pressure_bucket"], "pressure_unknown_or_ambiguous")
            self.assertTrue(all(row["review_status"] == "pending_paper_review" for row in rows))


if __name__ == "__main__":
    unittest.main()
