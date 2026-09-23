"""Mechanism and postprocessing checks using fabricated/public-only records."""
import copy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "dp_copula.py"
spec = importlib.util.spec_from_file_location("dp_copula", MODULE)
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)


def fake_inputs():
    rng = np.random.default_rng(1986)
    n = dp.PRIVATE_N
    mu = np.array([56, 24, 123, 105, 49, 24, 97], float)
    sd = np.array([10, 4, 20, 50, 10, 15, 15], float)
    x = rng.normal(mu, sd, (n, 7))
    schema = {
        "submitted_c_columns": ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"],
        "ranges": {"BMI": [10, 55], "SBP": [40, 250], "TG": [1, 3000],
                   "HDL": [10, 160], "ALT": [0.1, 700], "FPG": [40, 400]},
        "age_range": [40, 74],
        "rare_detector": {"columns": dp.CONT, "mu": mu.tolist(),
                          "sigma": np.diag(sd**2).tolist(), "band": {"lo": 4, "hi": 10, "cap": 12}, "gate": 8},
    }
    frame = pd.DataFrame(x, columns=dp.CONT)
    frame["age"] = np.rint(np.clip(frame.age, 40, 74)).astype(int)
    for name, limits in schema["ranges"].items():
        frame[name] = np.clip(frame[name], *limits)
    frame["sex"] = rng.integers(0, 2, n)
    frame["smoking"] = rng.integers(0, 2, n)
    frame["prefecture"] = np.array(["公開県A", "公開県B"])[rng.integers(0, 2, n)]
    event = rng.choice(3, n, p=[.65, .1, .25])
    frame["onset"] = (event == 2).astype(int)
    frame["death"] = (event == 1).astype(int)
    frame["time"] = np.where(event == 0, 9.99, rng.uniform(.1, 9.9, n))
    frame["record_id"] = [f"NOT_FOR_RELEASE_{i}" for i in range(n)]
    return frame, schema


class PrivacyMechanismTests(unittest.TestCase):
    def test_exact_budget_composition(self):
        for eps in [1, 3, 10, Fraction(4, 5)]:
            plan = dp.privacy_plan(eps)
            self.assertEqual(sum(Fraction(q["epsilon_allocated_exact"]) for q in plan["queries"]), eps)
            self.assertEqual(plan["delta"], 0)
            for query in plan["queries"]:
                allocation = Fraction(query["epsilon_allocated_exact"])
                q = Fraction(query["geometric_q_numerator"], query["geometric_q_denominator"])
                self.assertEqual(q, 2 / (2 + allocation))
                self.assertLessEqual(2 * math.log(1 / float(q)), float(allocation) + 1e-14)

    def test_two_sided_geometric_likelihood_bound(self):
        q = Fraction(7, 9)
        def pmf(k):
            return (1 - q) / (1 + q) * q ** abs(k)
        for k in range(-30, 31):
            for delta in [-2, -1, 0, 1, 2]:
                self.assertLessEqual(pmf(k) / pmf(k + delta), q ** (-abs(delta)))

    def test_rational_bernoulli_unbounded_loop(self):
        with patch.object(dp.secrets, "randbelow", side_effect=[0, 1, 0, 4]) as secret:
            self.assertEqual(dp._geometric(3, 5), 3)
            self.assertEqual(secret.call_count, 4)

    def test_noise_retains_arbitrary_integers(self):
        query = dp.privacy_plan(1)["queries"][0]
        with patch.object(dp, "_geometric", side_effect=[10**40, 0, 0, 10**40]):
            noisy = dp._noise_counts(np.array([0, 1]), query)
        self.assertEqual(noisy, [10**40, 1-10**40])
        self.assertEqual(dp._released_float(noisy).tolist(), [1e12, -1e12])

    def test_histogram_group_crossing_sensitivity(self):
        # Any row contributes one unit to exactly one group/bin cell. This
        # enumerates within-group, cross-group, endpoint and identity changes.
        for old in range(30):
            for new in range(30):
                before, after = np.zeros(30, int), np.zeros(30, int)
                before[old] = 1
                after[new] = 1
                self.assertLessEqual(np.abs(before-after).sum(), 2)

    def test_general_vector_budget(self):
        with patch.object(dp, "_geometric", return_value=0):
            values, plan = dp.release_integer_vector([1, -5, 8], Fraction(1, 5), 256)
        self.assertEqual(values, [1, -5, 8])
        self.assertEqual(Fraction(plan["geometric_q_numerator"], plan["geometric_q_denominator"]), Fraction(1280, 1281))


class PublicPostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.schema = fake_inputs()
        # Mock noise only in fabricated fixtures: do not use in a real release.
        with patch.object(dp, "_geometric", return_value=0):
            cls.model = dp.fit(cls.frame, cls.frame, cls.schema, 1)

    def test_model_json_roundtrip_and_no_record_ids(self):
        content = json.dumps(self.model, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("NOT_FOR_RELEASE", content)
        self.assertEqual(json.loads(content)["privacy"]["epsilon_exact"], "1")

    def test_private_data_only_enters_noise_queries(self):
        changed = self.frame.copy()
        changed.loc[0, "FPG"] = 400
        changed.loc[0, "prefecture"] = "unapproved_private_category"
        def blank(counts, query):
            return np.zeros_like(counts).tolist()
        with patch.object(dp, "_noise_counts", side_effect=blank):
            a = dp.fit(self.frame, self.frame, self.schema, 1)
            b = dp.fit(changed, self.frame, self.schema, 1)
        self.assertEqual(a, b)

    def test_fixed_n_contract_before_release(self):
        with patch.object(dp, "_noise_counts") as noise:
            with self.assertRaises(ValueError):
                dp.fit(self.frame.iloc[:-1], self.frame, self.schema, 1)
            noise.assert_not_called()

    def test_sample_is_reproducible_valid_postprocessing(self):
        frozen = copy.deepcopy(self.model)
        a = dp.sample(self.model, 1049, seed=91)
        b = dp.sample(self.model, 1049, seed=91)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(frozen, self.model)
        self.assertEqual(a.columns.tolist(), self.schema["submitted_c_columns"])
        self.assertEqual(len(a), 1049)
        self.assertFalse(a.isna().any().any())
        self.assertTrue(((a.onset + a.death) <= 1).all())
        for name, limits in dp._bounds(self.schema).items():
            self.assertTrue(a[name].between(*limits).all(), name)
        self.assertTrue(set(a.prefecture).issubset(set(self.frame.prefecture)))

    def test_huge_released_counts_remain_finite(self):
        changed = copy.deepcopy(self.model)
        changed["dp_group_counts"][0] = 10**400
        changed["dp_prefecture_counts"][0] = -(10**400)
        a = dp.sample(changed, 30, seed=1)
        self.assertTrue(np.isfinite(a[dp.NUMERIC]).all().all())

    def test_public_radial_map_keeps_output_in_domain(self):
        a = dp.sample(self.model, 1049, seed=91, public_radial_strength=1.0)
        self.assertTrue(np.isfinite(a[dp.NUMERIC]).all().all())
        for name, limits in dp._bounds(self.schema).items():
            self.assertTrue(a[name].between(*limits).all(), name)
        for name, limits in self.model["public_support"].items():
            self.assertTrue(a[name].between(*limits).all(), name)

    def test_empty_public_bins_do_not_invent_schema_wide_tails(self):
        for group in self.model["groups"]:
            for name, marginal in group["margins"].items():
                grid = np.asarray(marginal["public_within_bin_quantiles"])
                lo, hi = self.model["public_support"][name]
                self.assertGreaterEqual(float(grid.min()), lo)
                self.assertLessEqual(float(grid.max()), hi)


if __name__ == "__main__":
    unittest.main()
