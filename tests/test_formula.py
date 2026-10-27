import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.formula import canonicalize_formula


class FormulaTests(unittest.TestCase):
    def test_equivalent_element_order_gets_same_key(self) -> None:
        left = canonicalize_formula("La1.8Ba0.2CuO4")
        right = canonicalize_formula("Ba0.2Cu1La1.8O4")

        self.assertEqual(left.status, "parsed")
        self.assertEqual(left.canonical, right.canonical)

    def test_implicit_one_and_decimal_zeros_are_normalized(self) -> None:
        self.assertEqual(canonicalize_formula("Cu1.0O4.00").canonical, "CuO4")

    def test_unsupported_grouping_is_sent_to_review(self) -> None:
        result = canonicalize_formula("Ca(OH)2")

        self.assertIsNone(result.canonical)
        self.assertEqual(result.status, "unsupported")

    def test_zero_stoichiometry_is_omitted(self) -> None:
        result = canonicalize_formula("In0.12Pb0Sn0.88Te1")

        self.assertEqual(result.status, "parsed")
        self.assertEqual(result.canonical, "In0.12Sn0.88Te")


if __name__ == "__main__":
    unittest.main()
