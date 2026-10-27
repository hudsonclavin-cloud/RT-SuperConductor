import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.evidence import EvidenceKind, EvidenceRecord


class EvidenceRecordTests(unittest.TestCase):
    def test_unknown_transition_and_pressure_remain_unknown(self) -> None:
        record = EvidenceRecord(
            record_id="source:1",
            composition="Example2X",
            source_reference="doi:example",
            evidence_kind=EvidenceKind.EXPERIMENTAL,
        )

        self.assertIsNone(record.reported_tc_k)
        self.assertIsNone(record.measurement_pressure_gpa)
        self.assertFalse(record.has_ambient_pressure_evidence)

    def test_known_atmospheric_pressure_is_recognized(self) -> None:
        record = EvidenceRecord(
            record_id="source:2",
            composition="Example2Y",
            source_reference="doi:example-2",
            evidence_kind=EvidenceKind.EXPERIMENTAL,
            measurement_pressure_gpa=0.000101325,
        )

        self.assertTrue(record.has_ambient_pressure_evidence)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRecord(
                record_id="source:3",
                composition="Example2Z",
                source_reference="doi:example-3",
                evidence_kind=EvidenceKind.COMPUTATIONAL,
                confidence=1.1,
            )


if __name__ == "__main__":
    unittest.main()
