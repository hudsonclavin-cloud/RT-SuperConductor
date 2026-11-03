import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pretrained_alignn import evaluate_embeddings


def _row(jid, elements, lam, wlog):
    return {
        "jid": jid,
        "atoms": {
            "elements": elements,
            "lattice_mat": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
            "coords": [[0.0, 0.0, 0.0] for _ in elements],
        },
        "lamb": lam,
        "wlog": wlog,
    }


class PretrainedALIGNNProbeTests(unittest.TestCase):
    def test_structure_embeddings_use_same_element_system_group_kfold(self):
        pairs = [
            ("NaCl", ["Na", "Cl"]), ("KBr", ["K", "Br"]),
            ("LiF", ["Li", "F"]), ("MgO", ["Mg", "O"]),
            ("CaS", ["Ca", "S"]),
        ]
        rows = []
        for i, (_formula, elements) in enumerate(pairs):
            rows.append(_row(f"{i}-a", elements, 0.5 + i * 0.1, 100 + i * 10))
            rows.append(_row(f"{i}-b", elements, 0.55 + i * 0.1, 105 + i * 10))
        embeddings = np.arange(len(rows) * 4, dtype=float).reshape(len(rows), 4) / 10
        report = evaluate_embeddings(rows, embeddings, list(range(len(rows))))
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["rows_used"], 10)
        self.assertEqual(report["elemental_system_groups"], 5)
        self.assertEqual(len(report["folds"]), 5)
        self.assertTrue(all(fold["train_rows"] == 8 and fold["test_rows"] == 2 for fold in report["folds"]))
        self.assertIn("physics_tc_from_lambda_omega_log", report["metrics"])


if __name__ == "__main__":
    unittest.main()
