"""Public-only tests for whole-table bounded Cox target privacy invariants."""
import copy
from fractions import Fraction
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dp_cox_target as target


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # PUBLIC A_bg only; no actual B, B_self, utility reference or R artifact.
        data = HERE.parents[2]/"participant_data"/"A_bg.csv"
        public = pd.read_csv(data)
        cls.proxy = public.iloc[:1049].reset_index(drop=True)
        cls.reference = target.build_public_target(public.iloc[1049:].reset_index(drop=True))

    def test_exact_integer_bound_and_global_sensitivity(self):
        rng = np.random.default_rng(3217)
        for q in [16, 17, 65536]:
            points = [np.zeros(18), np.full(18, 1e300), np.full(18, -1e300),
                      np.eye(18)[0]*16, -np.eye(18)[0]*16]
            points += [rng.normal(size=18)*10**rng.uniform(-4, 100) for _ in range(100)]
            rounded = [target._integer_l1_projection(v, 16, q) for v in points]
            for integer in rounded:
                self.assertTrue(all(isinstance(x, int) for x in integer))
                self.assertLessEqual(sum(abs(x) for x in integer), q)
            for a, b in zip(rounded, rounded[1:]):
                self.assertLessEqual(sum(abs(x-y) for x, y in zip(a, b)), 2*q)
            # Worst-case replacement bound is attained by opposite range ends.
            self.assertEqual(sum(abs(x-y) for x,y in zip(rounded[3], rounded[4])), 2*q)

    def test_failed_private_fit_has_same_noise_path_and_no_flag(self):
        calls = []
        def fake_noise(values, epsilon, sensitivity):
            calls.append((values.copy(), epsilon, sensitivity))
            return [v+3 for v in values], {"test_only": True}
        def failure(*args):
            warnings.warn("PRIVATE DETAIL SHOULD NOT ESCAPE", RuntimeWarning)
            raise RuntimeError("PRIVATE DETAIL SHOULD NOT ESCAPE")
        capture = io.StringIO()
        with patch.object(target, "_fit_summary", side_effect=failure), redirect_stdout(capture), redirect_stderr(capture):
            result = target.measure_target(self.proxy, self.reference, "100", noise_callable=fake_noise)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], [0]*18)
        self.assertEqual(calls[0][1], Fraction(100))
        self.assertEqual(calls[0][2], 131072)
        self.assertEqual(capture.getvalue(), "")
        expected = target.postprocess_target([3]*18, self.reference, epsilon=100, mechanism={"test_only":True})
        self.assertEqual(result, expected)
        self.assertFalse(any("failure" in key or "clip_norm" in key for key in result))

    def test_noiseless_reconstruction_of_public_proxy_and_rounding(self):
        beta, se = target._fit_summary(self.proxy, self.reference)
        expected_se = np.asarray(self.reference["expected_se_standardized"])
        coordinates = np.r_[(beta-np.asarray(self.reference["beta_standardized"]))/expected_se,
                            2*(np.log(se)-np.log(expected_se))]
        radius = 64.0  # Fixed PUBLIC test choice; permits unclipped test reconstruction.
        self.assertLess(np.abs(coordinates).sum(), radius)
        seen = []
        def fake_noise(values, epsilon, sensitivity):
            seen.append(values.copy())
            return values.copy(), {"test_only":True}
        before = self.proxy.copy(deep=True)
        result = target.measure_target(self.proxy, self.reference, 100, radius=radius, noise_callable=fake_noise)
        self.assertEqual(len(seen), 1)
        np.testing.assert_allclose(np.asarray(result["beta_standardized"]), beta,
                                   atol=radius/65536*expected_se.max()/2+1e-12, rtol=0)
        np.testing.assert_allclose(np.log(result["standard_errors_standardized"]), np.log(se),
                                   atol=radius/65536/2/2+1e-12, rtol=0)
        pd.testing.assert_frame_equal(before, self.proxy)

    def test_default_exact_mechanism_called_once(self):
        with patch.object(target.dc, "release_integer_vector", return_value=([0]*18, {"mock":True})) as noise:
            result = target.measure_target(self.proxy, self.reference, Fraction(100))
        noise.assert_called_once()
        args = noise.call_args.args
        self.assertEqual(args[1:], (Fraction(100), 131072))
        self.assertLessEqual(sum(abs(v) for v in args[0]), 65536)
        self.assertEqual(result["epsilon_exact"], "100")
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["integer_l1_sensitivity"], 131072)

    def test_extreme_noisy_values_have_finite_public_clipping(self):
        result = target.postprocess_target([10**1000]*9+[-10**1000]*9, self.reference, epsilon=100)
        self.assertTrue(np.isfinite(result["beta_standardized"]).all())
        se = np.asarray(result["standard_errors_standardized"])
        np.testing.assert_allclose(se/np.asarray(self.reference["expected_se_standardized"]), .1)
        self.assertEqual(result["noisy_target_integer"][0], 10**1000)

    def test_public_size_and_parameter_rejections_precede_noise(self):
        with patch.object(target.dc, "release_integer_vector") as noise:
            with self.assertRaises(ValueError):
                target.measure_target(self.proxy.iloc[:-1], self.reference, 100)
            with self.assertRaises(ValueError):
                target.measure_target(self.proxy, self.reference, 100, radius=0)
            with self.assertRaises(ValueError):
                target.measure_target(self.proxy, self.reference, 0)
        noise.assert_not_called()


if __name__ == "__main__":
    unittest.main()
