import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rt_superconductor.label_quarantine import (  # noqa: E402
    quarantine_reason,
    suspected_label_reason,
    write_label_audit_queue,
)
from rt_superconductor.harmonized import HarmonizedRecord  # noqa: E402
from train_pressure_aware import select_training_records  # noqa: E402


def _record(record_id, *, source_db="3dsc", mp_id="", formula="CrCuO2", tc=132.0,
            review_status="curated_database", synthetic=0):
    return HarmonizedRecord(
        record_id=record_id, source_db=source_db, formula=formula, tc_kelvin=tc,
        applied_pressure_gpa=None, measurement_method="unknown", sample_form="unknown",
        doi="source", review_status=review_status, provenance="curated", is_synthetic=synthetic,
        chemical_family="Cuprates", mp_id=mp_id,
    )


class LabelQuarantineTests(unittest.TestCase):
    def test_mp510625_rows_are_suspected_and_other_high_tc_no_cu_is_flagged(self):
        rows = [
            _record("a", mp_id="mp-510625", formula="CrCuO2", tc=132),
            _record("b", mp_id="mp-510625", formula="Cr0.95CuFe0.05O2", tc=117.5),
            _record("c", mp_id="mp-510625", formula="Cr0.9CuFe0.1O2", tc=98),
            _record("d", mp_id="mp-other", formula="FeSe", tc=80),
            _record("e", source_db="uci", mp_id="", formula="H3S", tc=203),
        ]
        self.assertTrue(all(suspected_label_reason(row) == "suspected_label_error_shared_mp_structure_mp-510625" for row in rows[:3]))
        self.assertEqual(suspected_label_reason(rows[3]), "high_tc_above_77K_formula_without_Cu_needs_label_audit")
        self.assertIsNone(suspected_label_reason(rows[4]))

    def test_queue_records_exclusions_without_mutating_input_and_preserves_review_decision(self):
        row = _record("a", mp_id="mp-510625")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.csv"
            self.assertEqual(write_label_audit_queue([row], path)["active_exclusions"], 1)
            self.assertEqual(row.review_status, "curated_database")
            with path.open(newline="", encoding="utf-8") as handle:
                stored = next(csv.DictReader(handle))
            stored["audit_status"] = "reviewed"
            stored["review_decision"] = "retain"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(stored))
                writer.writeheader()
                writer.writerow(stored)
            counts = write_label_audit_queue([row], path)
            self.assertEqual(counts["released_after_review"], 1)
            self.assertIsNone(quarantine_reason(row, {"a": stored}))

    def test_training_selector_excludes_quarantined_row_only(self):
        quarantined = _record("suspect", mp_id="mp-510625")
        clean = _record("clean", mp_id="mp-123", formula="YBa2Cu3O7", tc=92)
        selected = select_training_records([quarantined, clean], label_audit_queue_path=None)
        self.assertEqual([row.record_id for row in selected], ["clean"])


if __name__ == "__main__":
    unittest.main()
