import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.datasets import _parse_supercon2_pressure


class SuperCon2ParsingTests(unittest.TestCase):
    def test_megabar_is_not_treated_as_millibar(self) -> None:
        pressure, caveat = _parse_supercon2_pressure("20 Mbar")

        self.assertEqual(pressure, 2000.0)
        self.assertIsNone(caveat)

    def test_pressure_range_remains_unknown(self) -> None:
        pressure, caveat = _parse_supercon2_pressure("7-9 GPa")

        self.assertIsNone(pressure)
        self.assertIn("range", caveat or "")


if __name__ == "__main__":
    unittest.main()
