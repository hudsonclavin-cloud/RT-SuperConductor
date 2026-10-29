import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_review_campaign import run_campaign


class ReviewCampaignTests(unittest.TestCase):
    def test_campaign_never_relabels_extracted_tc_as_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "queue.csv"
            fields = (
                "review_id", "source_record_id", "formula", "extracted_tc_kelvin",
                "extracted_pressure_gpa", "source_doi", "source_title", "review_status",
                "review_risk",
            )
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "review_id": "supercon2-review-001", "source_record_id": "rec1",
                    "formula": "MnP", "extracted_tc_kelvin": "290", "extracted_pressure_gpa": "0.0001",
                    "source_doi": "10.1/example", "source_title": "Magnetic ground states",
                    "review_status": "pending_paper_review", "review_risk": "unusually_high_extracted_tc",
                })
            output = root / "report.json"
            report = run_campaign(source, output, batch_size=1)
            serialized = output.read_text(encoding="utf-8")
            parsed = json.loads(serialized)
            self.assertEqual(parsed["learning_cycles_completed"], 0)
            self.assertEqual(parsed["model_inference_status"], "blocked_no_validated_candidate_model")
            self.assertIn('"extracted_tc_kelvin": "290"', serialized)
            self.assertNotIn("predicted_tc_k", serialized)
            self.assertNotIn("uncertainty_k", serialized)
            self.assertEqual(parsed["stages"][2]["tasks"][0]["task_type"], "paper_review")
            self.assertEqual(report["campaign_type"], "literature_review_triage")


if __name__ == "__main__":
    unittest.main()
