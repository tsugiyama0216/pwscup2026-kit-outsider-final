"""Behavioural checks on controlled synthetic inputs, not preliminary truths."""
import unittest

import numpy as np
import pandas as pd

from explore_common.core import (densest_time_center, local_time_predictions,
    nearest_distance_scores, score_to_confidence, QI_CONTINUOUS)


def frame(ages, times, sexes=None, prefectures=None):
    n = len(ages)
    data = {col: np.zeros(n) for col in QI_CONTINUOUS}
    data.update(age=ages, time=times, sex=sexes if sexes is not None else [0] * n,
                prefecture=prefectures if prefectures is not None else [1] * n)
    return pd.DataFrame(data)


class GeometryTests(unittest.TestCase):
    def test_window_objective_beats_average_on_multimodal_times(self):
        values = np.array([2.0, 2.2, 2.4, 8.0])
        result = densest_time_center(values)
        self.assertAlmostEqual(result, 2.2)
        self.assertEqual(np.sum(abs(values - result) <= .5), 3)
        self.assertEqual(np.sum(abs(values - values.mean()) <= .5), 0)

    def test_closed_boundary_window_and_order_invariance(self):
        self.assertEqual(densest_time_center([1., 0., 4.]), .5)
        self.assertEqual(densest_time_center([4., 0., 1.]), .5)
        self.assertEqual(densest_time_center([10., 10.]), 10.)

    def test_exact_distance_membership_and_categorical_fallback(self):
        bg = frame([40, 50, 60], [1, 2, 3])
        release = frame([40, 50, 60], [1, 2, 3], [0, 0, 1], [1, 2, 1])
        queries = frame([40, 50, 60], [0, 0, 0], [0, 0, 2], [1, 99, 99])
        np.testing.assert_array_equal(nearest_distance_scores(release, queries, bg), [0., 0., 0.])
        np.testing.assert_array_equal(local_time_predictions(release, queries, bg, mode="nearest"), [1., 2., 3.])

    def test_kth_ties_are_all_included_with_nondefault_indices(self):
        bg = frame([40, 50, 60], [1, 2, 3])
        release = frame([50, 50, 50], [2.0, 2.2, 9.0])
        release.index = [7, 9, 12]
        query = frame([50], [0])
        np.testing.assert_allclose(local_time_predictions(release, query, bg, k=1), [2.1])
        np.testing.assert_allclose(local_time_predictions(release.iloc[::-1], query, bg, k=1), [2.1])
        np.testing.assert_allclose(local_time_predictions(release, query, bg, mode="nearest"), [4.4])

    def test_confidence_preserves_ties_and_order(self):
        actual = score_to_confidence([-10000, 2, 2, 10000])
        self.assertTrue(np.all((actual > 0) & (actual < 1)))
        self.assertLess(actual[0], actual[1])
        self.assertEqual(actual[1], actual[2])
        self.assertLess(actual[2], actual[3])
        with self.assertRaises(ValueError):
            score_to_confidence([np.nan])


if __name__ == "__main__":
    unittest.main()
