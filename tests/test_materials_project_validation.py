import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.materials_project import (  # noqa: E402
    audit_materials_project_queries,
    normalize_fixed_formula,
)


class MaterialsProjectValidationTests(unittest.TestCase):
    def test_exact_formula_normalization(self):
        result = normalize_fixed_formula("Fe2O3")
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.normalized, "Fe2O3")
        self.assertEqual(result.signature, (("Fe", 2), ("O", 3)))

    def test_whitespace_and_formatting_variants(self):
        first = normalize_fixed_formula(" K D 2 P O 4 ")
        second = normalize_fixed_formula("Y Ba 2 Cu 3 O 7")
        self.assertEqual(first.status, "ambiguous_input")  # D is not guessed as an element.
        self.assertEqual(second.normalized, "Ba2Cu3O7Y")

    def test_fractional_stoichiometry_is_fixed_and_scale_invariant(self):
        result = normalize_fixed_formula("Ba0.5K0.5BiO3")
        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.signature, (("Ba", 1), ("Bi", 2), ("K", 1), ("O", 6)))

    def test_variable_and_ambiguous_formulas_are_skipped(self):
        for formula in ("La2-xSrxCuO4", "La2CuO4-delta", "Fe(III)Se", "YBCO", "A+B"):
            with self.subTest(formula=formula):
                self.assertEqual(normalize_fixed_formula(formula).status, "ambiguous_input")

    def test_composition_mismatch_is_rejected(self):
        class Summary:
            def search(self, **kwargs):
                return [SimpleNamespace(material_id="mp-wrong", formula_pretty="Fe3O4", energy_above_hull=0.01)]

        report = audit_materials_project_queries(
            [{"record_id": "source-1", "formula": "Fe2O3"}],
            summary_client=Summary(),
        )
        query = report["query_records"][0]
        self.assertEqual(query["query_status"], "no_match")
        self.assertEqual(query["validation_status"], "rejected_mismatch")
        self.assertEqual(query["returned_candidates"][0]["validation_status"], "rejected_mismatch")
        self.assertEqual(report["rejected_mismatch_count"], 1)

    def test_multiple_valid_structure_matches_are_preserved(self):
        class Summary:
            def search(self, **kwargs):
                return [
                    SimpleNamespace(material_id="mp-a", formula_pretty="Fe2O3", energy_above_hull=0.02),
                    SimpleNamespace(material_id="mp-b", formula_pretty="O3Fe2", energy_above_hull=0.01),
                ]

        report = audit_materials_project_queries(
            [{"record_id": "source-1", "formula": "Fe2 O3"}],
            summary_client=Summary(),
        )
        candidates = report["query_records"][0]["returned_candidates"]
        self.assertEqual(report["query_records"][0]["query_status"], "matched")
        self.assertEqual([candidate["material_id"] for candidate in candidates], ["mp-a", "mp-b"])
        self.assertEqual(report["safe_match_count"], 2)

    def test_no_match_is_recorded(self):
        class Summary:
            def search(self, **kwargs):
                return []

        report = audit_materials_project_queries(
            [{"record_id": "source-1", "formula": "MgB2"}],
            summary_client=Summary(),
        )
        self.assertEqual(report["query_records"][0]["query_status"], "no_match")
        self.assertEqual(report["no_match_count"], 1)
        self.assertEqual(report["api_error_count"], 0)

    def test_api_error_is_recorded_and_sanitized(self):
        class Summary:
            def search(self, **kwargs):
                raise RuntimeError("request failed?api_key=secret-value")

        report = audit_materials_project_queries(
            [{"record_id": "source-1", "formula": "MgB2"}],
            summary_client=Summary(),
            error_secrets=("secret-value",),
        )
        query = report["query_records"][0]
        self.assertEqual(query["query_status"], "query_error")
        self.assertEqual(report["api_error_count"], 1)
        self.assertNotIn("secret-value", query["error_detail"])

    def test_suspicious_kd_mapping_is_not_accepted(self):
        class Summary:
            def search(self, **kwargs):
                return [SimpleNamespace(material_id="mp-fuzzy", formula_pretty="KP(HO2)2", energy_above_hull=0.0)]

        report = audit_materials_project_queries(
            [{"record_id": "supercon2:example", "formula": "KD 2 PO 4"}],
            summary_client=Summary(),
        )
        query = report["query_records"][0]
        self.assertEqual(query["query_status"], "ambiguous_input")
        self.assertEqual(query["returned_candidates"], [])
        self.assertEqual(report["safe_match_count"], 0)


if __name__ == "__main__":
    unittest.main()
