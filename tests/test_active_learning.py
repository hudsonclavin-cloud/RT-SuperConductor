import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_3dsc import ingest_3dsc
from rt_superconductor.search import CandidateHypothesis, acquisition_score
from rt_superconductor.harmonized import load_harmonized


class ActiveLearningTests(unittest.TestCase):
    def test_3dsc_import_preserves_zero_negative_and_pressure_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "3DSC_MP.csv"
            source.write_text(
                '# "3DSC fixture"\n'
                "formula_sc,tc,sc_class,material_id_2,e_above_hull_2,spacegroup_2,cif\n"
                "FeSe,0,Other,mp-1,0.02,P 4/nmm,\n"
                "YBa2Cu3O7,92,Cuprate,mp-2,0.01,P m m m,\n",
                encoding="utf-8",
            )
            output = root / "harmonized.csv"
            report_path = root / "report.json"
            report = ingest_3dsc(
                input_path=source,
                output_path=output,
                report_path=report_path,
                structure_index_path=root / "index.csv",
            )
            records = load_harmonized(output)
            self.assertEqual(len(records), 2)
            self.assertEqual(report["zero_tc_row_count"], 1)
            self.assertEqual(records[0].tc_kelvin, 0)
            self.assertEqual(records[0].review_status, "curated_database")
            self.assertEqual(records[0].provenance, "curated")
            self.assertEqual(records[0].applied_pressure_gpa, 0.000101325)
            self.assertEqual(json.loads(report_path.read_text())["pressure_assumption"]["applied_pressure_gpa"], 0.000101325)

    def test_uncertainty_is_capped_and_plausibility_weighted(self) -> None:
        implausible = CandidateHypothesis(
            candidate_id="implausible",
            composition="FeSe",
            predicted_tc_k=290,
            tc_uncertainty_k=1000,
            pressure_gpa=0,
            structural_plausibility=0.0,
        )
        plausible = CandidateHypothesis(
            candidate_id="plausible",
            composition="LaH10",
            predicted_tc_k=290,
            tc_uncertainty_k=10,
            pressure_gpa=0,
            structural_plausibility=1.0,
        )
        implausible_score, implausible_reasons = acquisition_score(implausible)
        plausible_score, _ = acquisition_score(plausible)
        self.assertLess(implausible_score, plausible_score)
        self.assertTrue(any("capped" in reason for reason in implausible_reasons))
        self.assertTrue(any("weighted by plausibility" in reason for reason in implausible_reasons))


if __name__ == "__main__":
    unittest.main()
