import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.datasets import (
    iter_3dsc_mp_records,
    iter_kaggle_supercon_records,
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
