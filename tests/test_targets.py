import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.targets import AMBIENT_TARGET, ENGINEERING_TARGET


class TargetTests(unittest.TestCase):
    def test_ambient_window_requires_atmospheric_pressure(self) -> None:
        self.assertTrue(AMBIENT_TARGET.accepts(273.15, 0.000101325))
        self.assertFalse(AMBIENT_TARGET.accepts(273.15, 0.001))

    def test_engineering_window_accepts_provisional_ceiling(self) -> None:
        self.assertTrue(ENGINEERING_TARGET.accepts(298.15, 0.1))
        self.assertFalse(ENGINEERING_TARGET.accepts(298.15, 0.100001))


if __name__ == "__main__":
    unittest.main()
