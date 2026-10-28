import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.datasets import (
    iter_3dsc_mp_records,
    iter_kaggle_supercon_records,
    iter_supercon2_records,
    iter_uci_records,
)


class DatasetImporterTests(unittest.TestCase):
    def test_uci_preserves_unknown_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unique_m.csv"
            path.write_text("material,critical_temp\nX2Y,12.5\n", encoding="utf-8")

            records = list(iter_uci_records(path))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].composition, "X2Y")
        self.assertEqual(records[0].reported_tc_k, 12.5)
        self.assertIsNone(records[0].measurement_pressure_gpa)

    def test_3dsc_skips_preamble_and_keeps_multiline_fields(self) -> None:
        fieldnames = ["formula_sc", "tc", "cif", "material_id_2", "doi_2", "sc_class"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "3DSC_MP.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                handle.write("# generated dataset\n")
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "formula_sc": "X2Y",
                        "tc": "25",
                        "cif": "data/final/MP/cifs/X2Y.cif",
                        "material_id_2": "mp-1",
                        "doi_2": "10.example/record",
                        "sc_class": "Other",
                    }
                )

            records = list(iter_3dsc_mp_records(path))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].crystal_structure, "data/final/MP/cifs/X2Y.cif")
        self.assertEqual(records[0].source_reference, "10.example/record")

    def test_kaggle_supercon_keeps_journal_and_unknown_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.tsv"
            path.write_text(
                "1\t4\t6\t67\t85\t92\t198\n"
                "data number\tcommon formula of materials\tchemical formula\tcommon name of structure\tunit of Tc\tTc (of this sample) recommended\tjournal\n"
                "num\tname\telement\tstr3\tutc\ttc\tjournal\n"
                "7\tYBCO\tY1Ba2Cu3O7\tY123\tK\t92\tExample, 1(2020)1\n",
                encoding="utf-8",
            )
            records = list(iter_kaggle_supercon_records(path))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].reported_tc_k, 92.0)
        self.assertEqual(records[0].publication_reference, "Example, 1(2020)1")
        self.assertEqual(records[0].material_family, "Y123")
        self.assertIsNone(records[0].measurement_pressure_gpa)

    def test_supercon2_normalizes_pressure_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supercon2.csv"
            fieldnames = [
                "id", "rawMaterial", "materialId", "name", "formula", "doping", "shape",
                "materialClass", "fabrication", "substrate", "variables", "criticalTemperature",
                "criticalTemperatureMeasurementMethod", "appliedPressure", "section", "subsection",
                "hash", "title", "doi", "authors", "publisher", "journal", "year",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "abc", "rawMaterial": "LaH10", "materialId": "mat-1",
                        "name": "lanthanum hydride", "formula": "LaH10", "materialClass": "hydride",
                        "fabrication": "annealed", "substrate": "diamond", "variables": "x=1",
                        "criticalTemperature": "290 K", "criticalTemperatureMeasurementMethod": "resistivity",
                        "appliedPressure": "1 atm", "section": "body", "subsection": "figure",
                        "hash": "h", "title": "title", "doi": "10.example/doi", "authors": "a",
                        "publisher": "b", "journal": "c", "year": "2022",
                    }
                )
            records = list(iter_supercon2_records(path))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].reported_tc_k, 290.0)
        self.assertAlmostEqual(records[0].measurement_pressure_gpa, 0.000101325)
        self.assertEqual(records[0].publication_reference, "10.example/doi")
        self.assertEqual(records[0].measurement_method, "resistivity")
        self.assertIn("fabrication=annealed", records[0].synthesis_conditions or "")
