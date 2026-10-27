import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.triage import generate_triage_runs


class TriageTests(unittest.TestCase):
    def test_generates_four_named_passes(self) -> None:
        records = [
            {
                "canonical_composition": "BaCuO",
                "composition_key": "b acu o",
                "record_id": "uci:1",
                "reported_tc_k": 100,
                "crystal_structure": None,
            },
            {
                "canonical_composition": "BaCuO",
                "composition_key": "b acu o",
                "record_id": "3dsc_mp:1",
                "reported_tc_k": 90,
                "crystal_structure": "x.cif",
            },
            {
                "canonical_composition": "H2S",
                "composition_key": "h2s",
                "record_id": "uci:2",
                "reported_tc_k": 185,
                "crystal_structure": None,
            },
        ]

        runs = generate_triage_runs(records, limit=2)

        self.assertEqual(
            [run["pass"] for run in runs],
            [
                "highest_tc",
                "highest_tc_with_structure",
                "cross_source_evidence",
                "information_value",
            ],
        )
        self.assertEqual(runs[0]["candidates"][0]["canonical_composition"], "H2S")
        self.assertTrue(runs[1]["candidates"][0]["structure_count"] > 0)
        self.assertEqual(runs[2]["candidates"][0]["canonical_composition"], "BaCuO")


if __name__ == "__main__":
    unittest.main()
