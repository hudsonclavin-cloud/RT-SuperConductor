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
import evaluate_structure_features as structure_eval
import rank_candidates as candidate_ranker
from rt_superconductor.search import CandidateHypothesis, acquisition_score
from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized, write_harmonized

import numpy as np
from unittest.mock import patch


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

    def test_3dsc_import_skips_bad_rows_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "3DSC_MP.csv"
            source.write_text(
                "formula_sc,tc,sc_class,material_id_2,e_above_hull_2,spacegroup_2,cif\n"
                "FeSe,0,Other,mp-1,0.02,P 4/nmm,\n"
                "not-a-formula,missing,Other,mp-2,,,\n",
                encoding="utf-8",
            )
            output = root / "harmonized.csv"
            first = ingest_3dsc(input_path=source, output_path=output)
            second = ingest_3dsc(input_path=source, output_path=output)
            self.assertEqual(first["ingested_row_count"], 1)
            self.assertEqual(first["skipped_row_count"], 1)
            self.assertEqual(second["ingested_row_count"], 0)
            self.assertEqual(len(load_harmonized(output)), 1)

    def test_rank_candidates_writes_model_hypothesis_from_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.csv"
            write_harmonized(input_path, [HarmonizedRecord(
                record_id="a" * 24, source_db="supercon2", formula="YBa2Cu3O7",
                tc_kelvin=92, applied_pressure_gpa=0, measurement_method="resistivity",
                sample_form="single_crystal", doi="10.0000/example", review_status="confirmed",
                provenance="literature", is_synthetic=0, chemical_family="Cuprates",
            )])
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "pressure-aware-report.json").write_text(
                '{"training_status":"ready"}', encoding="utf-8"
            )
            (model_dir / "pressure-aware-model.joblib").write_bytes(b"test artifact placeholder")

            class Tree:
                def __init__(self, value: float) -> None:
                    self.value = value

                def predict(self, features: np.ndarray) -> np.ndarray:
                    return np.full(len(features), self.value)

            artifact = {
                "model": type("Forest", (), {"estimators_": [Tree(290), Tree(294)]})(),
                "hull_fill": 0.0,
                "dataset_sha256": "a" * 64,
            }
            candidate = {"formula": "YBa2Cu3S7", "chemical_family": "Cuprates", "parent_record_id": "a" * 24}
            output_path = root / "candidates" / "top_k.csv"
            with (
                patch.object(candidate_ranker, "enumerate_substitution_candidates", return_value=[candidate]),
                patch.object(candidate_ranker, "_mp_matches", return_value=({}, "degraded_no_mp_api_key", "no MP key")),
                patch.object(candidate_ranker.joblib, "load", return_value=artifact),
                patch.object(candidate_ranker, "build_features", return_value=(np.zeros((1, 2)), [], "test")),
            ):
                report = candidate_ranker.rank_candidates(
                    input_path=input_path, model_dir=model_dir, output_path=output_path, batch_size=1,
                )

            self.assertEqual(report["ranked_count"], 1)
            with output_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["formula"], "YBa2Cu3S7")
            self.assertAlmostEqual(float(row["predicted_tc_kelvin"]), 292.0)
            self.assertAlmostEqual(float(row["uncertainty_k"]), 2.0)
            self.assertEqual(row["status"], "review_required_hypothesis")

    def test_structure_feature_evaluation_reports_family_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.csv"
            records = [
                HarmonizedRecord(
                    record_id=f"{index:024x}", source_db="supercon2", formula=formula,
                    tc_kelvin=tc, applied_pressure_gpa=0, measurement_method="resistivity",
                    sample_form="single_crystal", doi=f"10.0000/{index}", review_status="confirmed",
                    provenance="literature", is_synthetic=0, chemical_family=family,
                )
                for index, (formula, tc, family) in enumerate((
                    ("YBa2Cu3O7", 92, "Cuprates"),
                    ("MgB2", 39, "Borides"),
                    ("Nb3Sn", 18, "A15"),
                ))
            ]
            write_harmonized(input_path, records)
            structure_root = root / "cifs"
            structure_root.mkdir()
            index_path = root / "index.csv"
            with index_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("record_id", "cif_path"))
                writer.writeheader()
                for index, record in enumerate(records):
                    cif = structure_root / f"{index}.cif"
                    cif.write_text("fixture", encoding="utf-8")
                    writer.writerow({"record_id": record.record_id, "cif_path": str(cif)})
            report_path = root / "structure-report.json"
            with patch.object(
                structure_eval, "_structure_features",
                side_effect=[([1.0, 0.1], ["density", "symmetry"]), ([2.0, 0.2], ["density", "symmetry"]), ([3.0, 0.3], ["density", "symmetry"])],
            ):
                report = structure_eval.evaluate_structure_features(
                    input_path=input_path, structure_index=index_path, output_path=report_path, n_splits=3,
                )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["cv"], "GroupKFold by chemical_family")
            self.assertEqual(len(report["folds"]), 3)
            self.assertEqual(report["structure_record_count"], 3)
            self.assertTrue(report_path.is_file())

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
