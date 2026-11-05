import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_pressure_aware as training


class SharedMaterialsProjectEmbargoTests(unittest.TestCase):
    def test_within_family_folds_purge_shared_mp_ids_before_fit(self):
        records = []
        for group_index, formula in enumerate(("H2O", "NaCl", "FeSe", "MgB2", "Nb3Sn")):
            for row_index in range(20):
                mp_id = f"mp-{group_index}-{row_index}"
                if group_index in (0, 1) and row_index == 0:
                    mp_id = "mp-shared-across-groups"
                records.append(SimpleNamespace(
                    record_id=f"r-{group_index}-{row_index}",
                    chemical_family="Test family",
                    formula=formula,
                    mp_id=mp_id,
                ))
        X = np.arange(len(records), dtype=float).reshape(-1, 1)
        y = np.arange(len(records), dtype=float)
        mp_by_row = {i: row.mp_id for i, row in enumerate(records)}
        fitted_rows = []
        checked_folds = []

        class SpyModel:
            def fit(self, features, targets):
                self.train_rows = {int(value) for value in features[:, 0]}
                fitted_rows.append(len(self.train_rows))
                return self

            def predict(self, features):
                test_rows = {int(value) for value in features[:, 0]}
                shared = {mp_by_row[i] for i in test_rows} & {mp_by_row[i] for i in self.train_rows}
                checked_folds.append((test_rows, self.train_rows, shared))
                if shared:
                    raise AssertionError(f"MP IDs leaked across a within-family fold: {shared}")
                return features[:, 0]

        with patch.object(training, "_model", side_effect=lambda _: SpyModel()):
            report = training.evaluate_within_family(records, X, y, n_splits=5, random_state=11)

        family = report["families"]["Test family"]
        folds = family["folds"]
        self.assertEqual(report["status"], "ready")
        self.assertIn("are purged", report["grouping"])
        self.assertIn("mp_id", report["shared_mp_embargo"])
        self.assertIn("test fold", report["shared_mp_embargo"])
        self.assertEqual(len(folds), 5)
        self.assertEqual(len(checked_folds), 5)
        self.assertEqual(sum(fold["shared_mp_rows_purged"] for fold in folds), 2)
        self.assertEqual(sum(fold["train_rows_before_embargo"] - fold["train_rows"] for fold in folds), 2)
        self.assertEqual(fitted_rows, [fold["train_rows"] for fold in folds])
        self.assertTrue(all(not shared for _, _, shared in checked_folds))

    def test_embargo_helper_keeps_blank_ids_and_removes_shared_only(self):
        rows = [
            SimpleNamespace(record_id="held", mp_id="mp-shared"),
            SimpleNamespace(record_id="purge", mp_id="mp-shared"),
            SimpleNamespace(record_id="blank", mp_id=""),
            SimpleNamespace(record_id="unique", mp_id="mp-unique"),
        ]
        kept, removed = training.purge_shared_mp_ids(rows, np.array([1, 2, 3]), np.array([0]))
        self.assertEqual(kept.tolist(), [2, 3])
        self.assertEqual(removed, ["purge"])


if __name__ == "__main__":
    unittest.main()
