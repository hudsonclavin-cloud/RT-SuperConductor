import sys
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_label_audit import (FAMILIES, HUMAN_FIELDS, SEED, build_rows,
                               select_audit_sample, fractional_formula)


class LabelAuditSampleTests(unittest.TestCase):
    def test_sample_is_fixed_seed_stratified_and_residual_blind(self):
        records = []
        for family in FAMILIES:
            for index in range(12):
                fractional = index >= 4
                records.append({
                    "record_id": f"{family}-{index:02d}",
                    "formula": f"Fe1.5As{index + 1}" if fractional else f"Fe{index + 1}As",
                    "chemical_family": family,
                    "tc_kelvin": "0" if index < 4 else str(index),
                    "mp_id": f"mp-{family}-{index}",
                })
        selected = select_audit_sample(records, seed=SEED)
        self.assertEqual(len(selected), 30)
        self.assertEqual([r["record_id"] for r in selected],
                         [r["record_id"] for r in select_audit_sample(records, seed=SEED)])
        for family in FAMILIES:
            rows = [row for row in selected if row["chemical_family"] == family]
            self.assertEqual(len(rows), 10)
            self.assertGreaterEqual(sum(float(row["tc_kelvin"]) == 0 for row in rows), 2)
            self.assertGreaterEqual(sum(fractional_formula(row["formula"]) for row in rows), 3)

    def test_fractional_formula_classifier(self):
        self.assertTrue(fractional_formula("LaFeAsO0.9F0.1"))
        self.assertFalse(fractional_formula("YBa2Cu3O7"))

    def test_build_rows_prefills_provenance_and_leaves_all_human_fields_blank(self):
        from pymatgen.core import Lattice, Structure
        from rt_superconductor.formula import canonicalize_formula

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_3dsc = root / "3dsc.csv"
            harmonized = []
            source_rows = []
            for family in FAMILIES:
                for i in range(10):
                    formula = "Fe1.5As1.5" if i >= 4 else f"Fe{i + 1}As{i + 1}"
                    tc = 0 if i < 3 else 20 + i
                    mp_id = f"mp-{family}-{i}"
                    record_id = hashlib.sha256(
                        "|".join(("3dsc", mp_id, canonicalize_formula(formula).canonical,
                                 f"{float(tc):.8g}", str(len(source_rows) + 2))).encode()
                    ).hexdigest()[:24]
                    harmonized.append({
                        "record_id": record_id, "formula": formula, "chemical_family": family,
                        "tc_kelvin": str(tc), "mp_id": mp_id,
                    })
                    source_rows.append({"formula_sc": formula, "tc": str(tc), "material_id_2": mp_id,
                                        "doi_2": "10.17188/123456", "synth_doped": "True"})
            with raw_3dsc.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(source_rows[0]))
                writer.writeheader(); writer.writerows(source_rows)
            supercon2 = root / "supercon2.csv"
            supercon2.write_text("formula,doi\nFeAs,10.1000/example\n")
            supermat = root / "supermat.csv"
            supermat.write_text("material,doi\nFeAs,10.1001/example\n")
            cif = root / "parent.cif"
            Structure(Lattice.cubic(4), ["Fe", "As"], [[0, 0, 0], [.5, .5, .5]]).to(filename=cif)
            index = root / "index.csv"
            with index.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["record_id", "cif_path"])
                writer.writeheader()
                writer.writerows({"record_id": row["record_id"], "cif_path": str(cif)} for row in harmonized)

            rows = build_rows(harmonized, source_3dsc=raw_3dsc, supercon2=supercon2,
                              supermat=supermat, structure_index=index)

        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["cif_parent_formula"], "FeAs")
        self.assertIn("10.17188/123456", rows[0]["doi_available"])
        self.assertIn("SuperCon2=10.1000/example", rows[0]["doi_available"])
        self.assertIn("SuperMat=10.1001/example", rows[0]["doi_available"])
        self.assertTrue(all(not row[field] for row in rows for field in HUMAN_FIELDS))


if __name__ == "__main__":
    unittest.main()
