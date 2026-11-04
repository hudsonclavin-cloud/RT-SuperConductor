import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from deep_uncertainty import (
    bins, independent_partitions, independent_calibration,
    physics_point_and_sd, validate_oof,
)
from train_physics_targets import allen_dynes_tc_k


class UncertaintyAuditTests(unittest.TestCase):
    def test_bins_preserve_every_held_out_error(self):
        sd = np.arange(20)[::-1].astype(float)
        error = np.arange(20, dtype=float)
        rows = bins(sd, error)
        self.assertEqual(sum(row['n'] for row in rows), 20)
        self.assertAlmostEqual(sum(row['n'] * row['observed_mae'] for row in rows) / 20, error.mean())
        self.assertLess(rows[0]['mean_sd'], rows[-1]['mean_sd'])

    def test_invalid_spread_is_not_reported_as_an_error_estimate(self):
        with self.assertRaises(ValueError):
            bins([float('nan')], [1])
        with self.assertRaises(ValueError):
            bins([-1], [1])
        with self.assertRaises(ValueError):
            bins([], [])

    def test_random_forest_cannot_exceed_its_training_label_range(self):
        x = np.array([[0.], [1.], [2.]])
        y = np.array([0., 10., 20.])
        model = RandomForestRegressor(n_estimators=20, random_state=2026).fit(x, y)
        predicted = model.predict(np.array([[-1e9], [1e9], [.5]]))
        self.assertTrue(np.all(predicted <= max(y)))
        self.assertTrue(np.all(predicted >= min(y)))

    def test_independent_group_partitions_are_deterministic_and_disjoint(self):
        groups = np.repeat(np.arange(20), 3)
        fit, cal, test, meta = independent_partitions(groups)
        again = independent_partitions(groups)
        for a, b in zip((fit, cal, test), again[:3]):
            np.testing.assert_array_equal(a, b)
        for a, b in ((fit, cal), (fit, test), (cal, test)):
            self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual([meta['fit_groups'], meta['calibration_groups'], meta['test_groups']], [12, 4, 4])

    def test_mp_ids_are_embargoed_between_all_three_partitions(self):
        groups = np.repeat(np.arange(20), 3)
        fit, cal, test, _ = independent_partitions(groups)
        refs = np.array(['mp-' + str(i) for i in range(len(groups))], dtype=object)
        refs[fit[0]] = refs[cal[0]] = 'shared-fit-cal'
        refs[cal[1]] = refs[test[0]] = 'shared-cal-test'
        fit, cal, test, meta = independent_partitions(groups, refs)
        for a, b in ((fit, cal), (fit, test), (cal, test)):
            self.assertFalse(set(refs[a]) & set(refs[b]))
        self.assertEqual(meta['fit_rows_purged_shared_reference'], 1)
        self.assertEqual(meta['calibration_rows_purged_shared_reference'], 1)

    def test_test_labels_cannot_change_model_or_calibrator_fit(self):
        groups = np.repeat(np.arange(30), 4)
        x = np.c_[np.arange(len(groups)) / 10, np.sin(np.arange(len(groups)))]
        y = 4 + x[:, 0] * 2 + x[:, 1]
        _, _, test, _ = independent_partitions(groups)
        first = independent_calibration(x, y, groups, trees=3)
        changed = y.copy()
        changed[test] += 1000
        second = independent_calibration(x, changed, groups, trees=3)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['model_fit_training_max_k'], second['model_fit_training_max_k'])
        self.assertEqual(first['calibration_observed_mean_absolute_error_k'], second['calibration_observed_mean_absolute_error_k'])
        self.assertEqual(first['predicted_expected_error_test_mean_k'], second['predicted_expected_error_test_mean_k'])
        self.assertGreater(second['prediction_test_mae_k'], first['prediction_test_mae_k'] + 900)

    def test_no_remaining_fit_data_is_reported_blocked(self):
        groups = np.repeat(np.arange(20), 3)
        result = independent_calibration(np.arange(60).reshape(-1, 1), np.arange(60), groups, ['same-parent'] * 60, trees=3)
        self.assertEqual(result['status'], 'blocked_insufficient_partition_rows')
        self.assertEqual(result['fit_rows'], 0)
        self.assertEqual(result['calibration_rows'], 0)
        self.assertNotIn('calibrated_expected_error_test_mae_k', result)

    def test_physics_point_uses_mean_lambda_and_geometric_mean_omega(self):
        point, sd = physics_point_and_sd(np.array([[1.], [2.]]), np.log(np.array([[100.], [400.]])))
        self.assertAlmostEqual(point[0], allen_dynes_tc_k(1.5, 200.))
        self.assertGreater(sd[0], 0)
        with self.assertRaises(ValueError):
            physics_point_and_sd(np.array([[1.], [2.]]), np.array([[1.]]))

    def test_oof_predictions_must_cover_the_same_targets_once(self):
        rows = [dict(index=0, actual=1., predicted=2., sd=1., training_max=5.)]
        self.assertEqual(validate_oof(rows, np.array([1.]), [0]), rows)
        with self.assertRaises(ValueError):
            validate_oof(rows * 2, np.array([1.]), [0])
        with self.assertRaises(ValueError):
            validate_oof(rows, np.array([5.]), [0])


class PhysicsReferenceGuardTests(unittest.TestCase):
    def test_generated_values_cannot_become_reference_labels(self):
        from deep_uncertainty import physics_audit
        rows=[{'jid':'candidate-unit','is_generated_candidate':True,'lambda':1.,'omega_log_k':200.}]
        with self.assertRaisesRegex(ValueError,'Generated candidates'):
            physics_audit(rows,trees=3)
        rows=[{'jid':'unit-scored','is_generated_candidate':' 1.0 ','lambda':1.,'omega_log_k':200.}]
        with self.assertRaisesRegex(ValueError,'Generated candidates'):
            physics_audit(rows,trees=3)
