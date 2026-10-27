import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.datasets import iter_supermat_records
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA


class SuperMatTests(unittest.TestCase):
    def test_pressure_and_method_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supermat.csv"
            path.write_text(
                "id,material,tcValue,pressure,me_method,filename\n"
                ",H2S,203 K,155 GPa,magnetization,paper.xml\n"
                ",X,30 K,ambient pressure,resistivity,paper.xml\n",
                encoding="utf-8",
            )
            records = list(iter_supermat_records(path))

        self.assertEqual(records[0].reported_tc_k, 203.0)
        self.assertEqual(records[0].measurement_pressure_gpa, 155.0)
        self.assertEqual(records[0].measurement_method, "magnetization")
        self.assertEqual(records[1].measurement_pressure_gpa, ATMOSPHERIC_PRESSURE_GPA)

    def test_pressure_range_remains_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supermat.csv"
            path.write_text(
                "id,material,tcValue,pressure,me_method,filename\n"
                ",H2S,203(1) K,145-155 GPa,,paper.xml\n",
                encoding="utf-8",
            )
            record = next(iter_supermat_records(path))

        self.assertIsNone(record.measurement_pressure_gpa)
        self.assertIsNone(record.reported_tc_k)
        self.assertTrue(any("raw_pressure" in caveat for caveat in record.caveats))

    def test_millikelvin_is_converted_to_kelvin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supermat.csv"
            path.write_text(
                "id,material,tcValue,pressure,me_method,filename\n"
                ",UPt3,530 mK,,specific heat,paper.xml\n",
                encoding="utf-8",
            )
            record = next(iter_supermat_records(path))

        self.assertEqual(record.reported_tc_k, 0.53)


if __name__ == "__main__":
    unittest.main()
