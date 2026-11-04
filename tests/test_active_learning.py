import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_3dsc import ingest_3dsc
from build_supercon2_review_queue import build_queue as build_supercon2_queue
import promotion_gate
import train_pressure_aware as pressure_training
import evaluate_structure_features as structure_eval
import enrich_materials_project as mp_enrichment
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
            self.assertIsNone(records[0].applied_pressure_gpa)  # Missing pressure must not become an ambient observation.
            self.assertIsNone(json.loads(report_path.read_text())["pressure_assumption"]["applied_pressure_gpa"])
            self.assertEqual(json.loads(report_path.read_text())["pressure_assumption"]["historical_screening_target_gpa"], 0.000101325)
            self.assertFalse(json.loads(report_path.read_text())["pressure_assumption"]["is_observed"])

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

    def test_family_ranker_filters_nonmetals_and_avoids_room_temperature_target_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.csv"
            write_harmonized(input_path, [HarmonizedRecord(
                record_id="f" * 24, source_db="3dsc", formula="Fe2O3",
                tc_kelvin=10, applied_pressure_gpa=0, measurement_method="unknown",
                sample_form="unknown", doi="10.0000/example", review_status="curated_database",
                provenance="curated", is_synthetic=0, chemical_family="Ferrites",
            )])
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "pressure-aware-report.json").write_text(
                '{"training_status":"ready"}', encoding="utf-8"
            )
            (model_dir / "pressure-aware-model.joblib").write_bytes(b"placeholder")

            class Tree:
                def __init__(self, value: float) -> None:
                    self.value = value

                def predict(self, features: np.ndarray) -> np.ndarray:
                    return np.full(len(features), self.value)

            artifact = {
                "model": type("Forest", (), {"estimators_": [Tree(30), Tree(34)]})(),
                "hull_fill": 0.0,
                "dataset_sha256": "b" * 64,
            }
            candidates = [
                {"formula": "Fe3O4", "chemical_family": "Ferrites", "parent_record_id": "f" * 24},
                {"formula": "FeO", "chemical_family": "Ferrites", "parent_record_id": "f" * 24},
            ]
            output_path = root / "candidates" / "ferrites_top25.csv"
            with (
                patch.object(candidate_ranker, "enumerate_substitution_candidates", return_value=candidates),
                patch.object(candidate_ranker, "_mp_matches", return_value=(
                    {
                        "Fe3O4": {"mp_id": "mp-1", "energy_above_hull": 0.01, "band_gap_ev": None, "is_metal": True},
                        "FeO": {"mp_id": "mp-2", "energy_above_hull": 0.01, "band_gap_ev": 0.2, "is_metal": False},
                    }, "ready", None,
                )),
                patch.object(candidate_ranker.joblib, "load", return_value=artifact),
                patch.object(candidate_ranker, "build_features", return_value=(np.zeros((1, 2)), [], "test")),
            ):
                report = candidate_ranker.rank_candidates(
                    input_path=input_path, model_dir=model_dir, output_path=output_path,
                    batch_size=25, family="Ferrites", api_key="test-key",
                )

            self.assertEqual(report["score_mode"], "within_family_predicted_tc_and_uncertainty")
            self.assertEqual(report["rejected_nonmetal_count"], 1)
            with output_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["formula"], "Fe3O4")
            self.assertEqual(row["band_gap_ev"], "")
            self.assertIn("family mode does not use distance", row["reasons"])

    def test_family_ranker_always_writes_validation_and_hypothesis_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.csv"
            write_harmonized(input_path, [HarmonizedRecord(
                record_id="f" * 24, source_db="3dsc", formula="Fe2O3",
                tc_kelvin=10, applied_pressure_gpa=0, measurement_method="unknown",
                sample_form="unknown", doi="10.0000/example", review_status="curated_database",
                provenance="curated", is_synthetic=0, chemical_family="Ferrites",
            )])
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "pressure-aware-report.json").write_text('{"training_status":"ready"}', encoding="utf-8")
            (model_dir / "pressure-aware-model.joblib").write_bytes(b"placeholder")

            class Tree:
                def __init__(self, value: float) -> None:
                    self.value = value

                def predict(self, features: np.ndarray) -> np.ndarray:
                    return np.full(len(features), self.value)

            artifact = {
                "model": type("Forest", (), {"estimators_": [Tree(30), Tree(32)]})(),
                "hull_fill": 0.0,
                "dataset_sha256": "c" * 64,
            }
            candidates = [
                {"formula": "Fe3O4", "chemical_family": "Ferrites", "parent_record_id": "f" * 24},
                {"formula": "FeO", "chemical_family": "Ferrites", "parent_record_id": "f" * 24},
            ]
            uci = root / "unique_m.csv"
            uci.write_text("material,critical_temp\nFe3O4,30\n", encoding="utf-8")
            output = root / "ferrites_top25.csv"
            validation = root / "ferrites_validation.csv"
            hypotheses = root / "ferrites_hypotheses.csv"
            with (
                patch.object(candidate_ranker, "enumerate_substitution_candidates", return_value=candidates),
                patch.object(candidate_ranker, "_mp_matches", return_value=(
                    {
                        "Fe3O4": {"mp_id": "mp-1", "energy_above_hull": 0.01, "band_gap_ev": None, "is_metal": True},
                        "FeO": {"mp_id": "mp-2", "energy_above_hull": 0.01, "band_gap_ev": None, "is_metal": True},
                    }, "ready", None,
                )),
                patch.object(candidate_ranker.joblib, "load", return_value=artifact),
                patch.object(candidate_ranker, "build_features", return_value=(np.zeros((2, 2)), [], "test")),
            ):
                report = candidate_ranker.rank_candidates(
                    input_path=input_path, model_dir=model_dir, output_path=output,
                    batch_size=25, family="Ferrites", api_key="test-key",
                    uci_path=uci, stanev_path=root / "missing-stanev.csv",
                    validation_output_path=validation, hypotheses_output_path=hypotheses,
                )

            self.assertEqual(report["validation_count"], 1)
            self.assertEqual(report["hypotheses_count"], 1)
            with validation.open(newline="", encoding="utf-8") as handle:
                validation_rows = list(csv.DictReader(handle))
            with hypotheses.open(newline="", encoding="utf-8") as handle:
                hypothesis_rows = list(csv.DictReader(handle))
            self.assertEqual(validation_rows[0]["formula"], "Fe3O4")
            self.assertEqual(hypothesis_rows[0]["formula"], "FeO")

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


    def test_promotion_gate_includes_curated_database_at_one_atmosphere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            predictions.write_text(
                "formula,chemical_family,tc_kelvin,predicted_tc_kelvin,applied_pressure_gpa,review_status,provenance,is_synthetic\n"
                "YBa2Cu3O7,Cuprates,92,90,0.000101325,curated_database,curated,0\n"
                "MgB2,Borides,39,38,0.000101326,curated_database,curated,0\n",
                encoding="utf-8",
            )
            model = root / "model.joblib"
            model.write_bytes(b"placeholder")
            report = promotion_gate.evaluate_gate(predictions_path=predictions, model_path=model)

            self.assertEqual(promotion_gate.AMBIENT_PRESSURE_MAX_GPA, 0.000101325)
            self.assertEqual(report["ambient_eligible_rows"], 1)
            self.assertAlmostEqual(report["ambient_mae_k"], 2.0)
            self.assertIn("Cuprates", report["ambient_by_family"])

    def test_promotion_gate_requires_twenty_five_percent_in_family_mean_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            predictions.write_text(
                "formula,chemical_family,tc_kelvin,predicted_tc_kelvin,applied_pressure_gpa,review_status,provenance,is_synthetic\n"
                "Fe2O3,Ferrites,100,95,0.000101325,curated_database,curated,0\n"
                "Fe3O4,Ferrites,20,25,0.000101325,curated_database,curated,0\n"
                "YBa2Cu3O7,Cuprates,100,60,0.000101325,curated_database,curated,0\n"
                "YBa2Cu4O8,Cuprates,20,60,0.000101325,curated_database,curated,0\n",
                encoding="utf-8",
            )
            model = root / "model.joblib"
            model.write_bytes(b"placeholder")

            report = promotion_gate.evaluate_gate(predictions_path=predictions, model_path=model)

            self.assertFalse(report["checks"]["ambient_mean_improvement_25pct"])
            self.assertTrue(report["ambient_by_family"]["Ferrites"]["meets_25pct_improvement"])
            self.assertFalse(report["ambient_by_family"]["Cuprates"]["meets_25pct_improvement"])
            self.assertAlmostEqual(report["ambient_by_family"]["Ferrites"]["mean_predictor_mae_k"], 40.0)

    def test_supercon2_queue_skips_source_records_already_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "supercon2.csv"
            source.write_text(
                "id,formula,criticalTemperature,appliedPressure,doi,title,year,criticalTemperatureMeasurementMethod,shape,materialClass\n"
                "reviewed-id,YBa2Cu3O7,92 K,ambient,10.1/example,Reviewed source,2020,resistivity,pellet,cuprate\n"
                "new-id,H3S,200 K,ambient,10.1/new,Unreviewed source,2021,resistivity,pellet,hydride\n",
                encoding="utf-8",
            )
            reviewed = root / "reviewed_records.csv"
            reviewed.write_text(
                "record_id,source_record_id\nreviewed-row,reviewed-id\n",
                encoding="utf-8",
            )
            output = root / "queue.csv"

            count = build_supercon2_queue(source, output, limit=20, reviewed_path=reviewed)

            self.assertEqual(count, 1)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_record_id"] for row in rows], ["new-id"])

    def test_within_family_evaluation_groups_whole_elemental_systems(self) -> None:
        records = [
            SimpleNamespace(
                chemical_family="Test family",
                formula=formula,
            )
            for formula in ("H2O", "NaCl", "FeSe", "MgB2", "Nb3Sn")
            for _ in range(20)
        ]
        y = np.arange(100, dtype=float)
        X = y.reshape(-1, 1)

        class FeatureModel:
            def fit(self, features: np.ndarray, targets: np.ndarray) -> "FeatureModel":
                return self

            def predict(self, features: np.ndarray) -> np.ndarray:
                return features[:, 0]

        with patch.object(pressure_training, "_model", side_effect=lambda _: FeatureModel()):
            report = pressure_training.evaluate_within_family(
                records, X, y, n_splits=5, random_state=7,
            )

        family = report["families"]["Test family"]
        self.assertEqual(report["status"], "ready")
        self.assertEqual(family["row_count"], 100)
        self.assertEqual(family["elemental_system_count"], 5)
        self.assertEqual(len(family["folds"]), 5)
        self.assertAlmostEqual(family["mae_k"], 0.0)
        self.assertGreater(family["mean_predictor_mae_k"], 0.0)
        self.assertAlmostEqual(family["spearman_rho"], 1.0)
        all_test_groups = [
            group for fold in family["folds"] for group in fold["held_out_elemental_systems"]
        ]
        self.assertEqual(sorted(all_test_groups), ["B-Mg", "Cl-Na", "Fe-Se", "H-O", "Nb-Sn"])

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

    def test_materials_project_api_miss_preserves_known_mp_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            write_harmonized(input_path, [HarmonizedRecord(
                record_id="c" * 24, source_db="3dsc", formula="MgB2", tc_kelvin=39,
                applied_pressure_gpa=0.000101325, measurement_method="unknown", sample_form="unknown",
                doi="10.6084/m9.figshare.c.6914407", review_status="curated_database",
                provenance="curated", is_synthetic=0, chemical_family="Borides", mp_id="mp-known",
                energy_above_hull=0.012, spacegroup_number=191,
            )])

            class Summary:
                def search(self, *, formula: str, fields: list[str]):
                    raise RuntimeError("temporary Materials Project API failure")

            client = type("Client", (), {"materials": type("Materials", (), {"summary": Summary()})()})()
            report = mp_enrichment.enrich_csv(input_path, output_path, client=client)
            with output_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

            self.assertIn("MgB2", report["errors"])
            self.assertEqual(row["mp_id"], "mp-known")
            self.assertEqual(row["energy_above_hull"], "0.012")
            self.assertEqual(row["spacegroup_number"], "191")

    def test_materials_project_exports_additional_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            write_harmonized(input_path, [HarmonizedRecord(
                record_id="d" * 24, source_db="3dsc", formula="MgB2", tc_kelvin=39,
                applied_pressure_gpa=0.000101325, measurement_method="unknown", sample_form="unknown",
                doi="10.6084/m9.figshare.c.6914407", review_status="curated_database",
                provenance="curated", is_synthetic=0, chemical_family="Borides",
            )])

            class Summary:
                fields_requested: list[str] = []

                def search(self, *, formula: str, fields: list[str]):
                    self.fields_requested = fields
                    return [SimpleNamespace(
                        material_id="mp-new", formula_pretty=formula, energy_above_hull=0.01,
                        symmetry=SimpleNamespace(number=191), formation_energy_per_atom=-0.5,
                        band_gap=0.0, is_metal=True, total_magnetization=1.2,
                        density=4.5, volume=20.0,
                    )]

            summary = Summary()
            client = type("Client", (), {"materials": type("Materials", (), {"summary": summary})()})()
            mp_enrichment.enrich_csv(input_path, output_path, client=client)
            with output_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["mp_id"], "mp-new")
            self.assertEqual(row["formation_energy_per_atom"], "-0.5")
            self.assertEqual(row["band_gap_ev"], "0.0")
            self.assertEqual(row["is_metal"], "True")
            self.assertEqual(row["total_magnetization_muB"], "1.2")
            self.assertEqual(row["density_g_cm3"], "4.5")
            self.assertEqual(row["volume_angstrom3"], "20.0")
            self.assertIn("formation_energy_per_atom", summary.fields_requested)
            self.assertIn("total_magnetization", summary.fields_requested)


if __name__ == "__main__":
    unittest.main()
