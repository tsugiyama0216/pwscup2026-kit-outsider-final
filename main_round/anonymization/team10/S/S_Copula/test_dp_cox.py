"""Public artificial-data checks of the bounded Cox extension."""
import unittest
import numpy as np
import pandas as pd
import dp_cox as dc


def artificial():
    rng = np.random.default_rng(814)
    n = 400
    frame = pd.DataFrame({
        "age": rng.integers(40, 75, n), "sex": rng.integers(0, 2, n),
        "BMI": rng.normal(24, 3, n), "SBP": rng.normal(125, 14, n),
        "TG": rng.lognormal(4.5, .3, n), "HDL": rng.normal(48, 5, n),
        "ALT": rng.lognormal(3, .2, n), "FPG": rng.normal(92, 8, n),
        "smoking": rng.integers(0, 2, n),
    })
    rate = np.exp(.03*(frame.age-55) + .05*(frame.FPG-92))*.035
    time = rng.exponential(1/rate)
    frame["time"] = np.minimum(time, 10)
    frame["onset"] = (time < 10).astype(int)
    frame["death"] = 0
    return frame


class TestCox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = artificial()
        cls.model = dc.build_public_cox_model(cls.frame)

    def test_integer_bound_and_replacement_sensitivity(self):
        rows = dc._integer_rows(self.frame, self.model)
        self.assertLessEqual(int(np.abs(rows).sum(axis=1).max()), self.model["quantization"])
        changed = self.frame.copy()
        changed.loc[0, ["FPG", "onset", "time"]] = [400, 1, .001]
        other = dc._integer_rows(changed, self.model)
        self.assertTrue(np.array_equal(rows[1:], other[1:]))
        distance = int(np.abs(rows.sum(axis=0)-other.sum(axis=0)).sum())
        self.assertLessEqual(distance, 2*self.model["quantization"])

    def test_one_measurement_and_no_raw_statistics(self):
        calls=[]
        def noise(values, sensitivity, epsilon):
            calls.append((sensitivity,epsilon))
            return [0]*len(values)
        release=dc.measure_cox_update(self.frame,self.model,epsilon=2,public_n=len(self.frame),noise_integer_vector=noise)
        self.assertEqual(calls,[(256,2)])
        self.assertEqual(release["noisy_score_integer"],[0]*10)
        self.assertNotIn("raw_score",release)

    def test_unbounded_noise_safe_postprocess(self):
        release=dc.postprocess_cox_update([10**500]*10,self.model,epsilon=1,public_n=400)
        self.assertTrue(np.isfinite(release["beta_standardized"]).all())
        self.assertEqual(release["noisy_score_integer"][0],10**500)

    def test_rank_calibration_preserves_marginals_and_categorical_outcomes(self):
        target={"beta_standardized":self.model["beta_standardized"]}
        result,trace=dc.calibrate_synthetic_cox(self.frame,target,self.model,iterations=2)
        for col in self.frame.columns:
            np.testing.assert_array_equal(np.sort(result[col]),np.sort(self.frame[col]))
        for col in ["age","sex","smoking","onset","death","time"]:
            np.testing.assert_array_equal(result[col],self.frame[col])
        self.assertEqual(trace["selection_input"],"synthetic_C_and_DP_target_only")


if __name__ == "__main__":
    unittest.main()
