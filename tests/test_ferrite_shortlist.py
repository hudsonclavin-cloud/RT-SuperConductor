import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_ferrite_shortlist import validate_shortlist


class FerriteShortlistValidationTests(unittest.TestCase):
    def test_missing_literature_exports_is_blocked_with_zero_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "ferrites.csv"
            candidates.write_text(
                "rank,candidate_id,formula,predicted_tc_kelvin,known_in_database\n"
                "1,c1,Fe3O4,30,unknown\n",
                encoding="utf-8",
            )
            report = validate_shortlist(
                candidates_path=candidates,
                output_path=root / "queue.csv",
                report_path=root / "report.json",
            )

            self.assertEqual(report["status"], "blocked_missing_supercon2_and_supermat_exports")
            self.assertEqual(report["known_superconductor_count"], 0)
            self.assertTrue((root / "queue.csv").is_file())

    def test_matches_are_pending_and_ordering_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "ferrites.csv"
            candidates.write_text(
                "rank,candidate_id,formula,predicted_tc_kelvin,known_in_database\n"
                "1,c1,Fe3O4,30,known\n"
                "2,c2,LaFeO3,20,known\n",
                encoding="utf-8",
            )
            supercon2 = root / "supercon2.csv"
            fields = (
                "id,rawMaterial,materialId,name,formula,materialClass,fabrication,criticalTemperature,"
                "criticalTemperatureMeasurementMethod,appliedPressure,section,subsection,title,doi,journal,year"
            )
            with supercon2.open("w", newline="", encoding="utf-8") as handle:
                handle.write(fields + "\n")
                writer = csv.DictWriter(handle, fieldnames=fields.split(","))
                writer.writerow({"id": "one", "rawMaterial": "Fe3O4", "materialId": "id1", "name": "Ferrite", "formula": "Fe3O4", "materialClass": "Ferrites", "criticalTemperature": "30 K", "doi": "10.1000/fe3o4", "year": "2020"})
                writer.writerow({"id": "two", "rawMaterial": "LaFeO3", "materialId": "id2", "name": "Ferrite", "formula": "LaFeO3", "materialClass": "Ferrites", "criticalTemperature": "20 K", "doi": "10.1000/lafeo3", "year": "2020"})
            output = root / "queue.csv"
            report = validate_shortlist(
                candidates_path=candidates,
                output_path=output,
                report_path=root / "report.json",
                supercon2_path=supercon2,
            )

            self.assertEqual(report["matched_candidate_count"], 2)
            self.assertEqual(report["known_superconductor_count"], 2)
            self.assertAlmostEqual(report["ordering_spearman_rho"], 1.0)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["review_status"] == "pending_paper_review" for row in rows))
            self.assertEqual(rows[0]["doi"], "10.1000/fe3o4")


if __name__ == "__main__":
    unittest.main()
