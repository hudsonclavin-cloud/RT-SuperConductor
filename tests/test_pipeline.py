import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.pipeline import composition_key, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_composition_key_only_removes_whitespace(self) -> None:
        self.assertEqual(composition_key("Ba 2 Cu 3 O 7"), "ba2cu3o7")
        self.assertEqual(composition_key("CuBa"), "cuba")

    def test_run_pipeline_writes_records_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uci = root / "unique_m.csv"
            uci.write_text("material,critical_temp\nX2Y,12.5\n", encoding="utf-8")
            dsc = root / "3DSC_MP.csv"
            dsc.write_text(
                "# generated\n"
                "formula_sc,tc,cif,material_id_2,doi_2,sc_class\n"
                "X2Y,25,data/final/MP/cifs/X2Y.cif,mp-1,doi:record,Other\n",
                encoding="utf-8",
            )
            output = root / "processed" / "evidence.jsonl"
            report = root / "processed" / "report.json"

            summary = run_pipeline(
                uci_path=uci,
                threedsc_path=dsc,
                output_path=output,
                report_path=report,
            )

            records = [json.loads(line) for line in output.read_text().splitlines()]
            report_data = json.loads(report.read_text())

        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.duplicate_composition_groups, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["reported_tc_k"], 12.5)
        self.assertEqual(report_data["summary"]["records_with_structure"], 1)

    def test_run_pipeline_includes_optional_supermat_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uci = root / "unique_m.csv"
            uci.write_text("material,critical_temp\nX2Y,12.5\n", encoding="utf-8")
            dsc = root / "3DSC_MP.csv"
            dsc.write_text(
                "# generated\nformula_sc,tc,cif,material_id_2,doi_2,sc_class\n"
                "X2Y,25,,mp-1,doi:record,Other\n",
                encoding="utf-8",
            )
            supermat = root / "SuperMat-2.0.csv"
            supermat.write_text(
                "id,material,tcValue,pressure,me_method,filename\n"
                ",X2Y,30 K,2 GPa,resistivity,paper.xml\n",
                encoding="utf-8",
            )

            summary = run_pipeline(
                uci_path=uci,
                threedsc_path=dsc,
                supermat_path=supermat,
                output_path=root / "evidence.jsonl",
                report_path=root / "report.json",
            )

        self.assertEqual(summary.total_records, 3)
        self.assertEqual(summary.records_with_known_pressure, 1)


if __name__ == "__main__":
    unittest.main()
