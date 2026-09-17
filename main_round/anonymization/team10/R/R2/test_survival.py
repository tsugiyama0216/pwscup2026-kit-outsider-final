"""Synthetic-data tests for the R2 survival component; no contest file access."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError

import survival


def toy_data(n=700):
    rng = np.random.default_rng(81)
    age = rng.integers(40, 75, n)
    bmi = rng.normal(24, 3, n)
    b = pd.DataFrame({"age": age, "sex": rng.integers(0, 2, n),
                      "BMI": bmi, "SBP": 90 + age + rng.normal(0, 8, n),
                      "TG": rng.lognormal(4.5, 0.4, n), "HDL": rng.normal(50, 6, n),
                      "ALT": rng.lognormal(3, 0.3, n), "FPG": 90 + 2 * (bmi - 24) + rng.normal(0, 10, n),
                      "smoking": rng.integers(0, 2, n)})
    onset_t = rng.exponential(25 * np.exp(-0.03 * (b.FPG - 90)), n)
    death_t = rng.exponential(65 * np.exp(-0.03 * (age - 55)), n)
    b["onset"] = ((onset_t <= death_t) & (onset_t <= 10)).astype(int)
    b["death"] = ((death_t < onset_t) & (death_t <= 10)).astype(int)
    b["time"] = np.minimum(np.minimum(onset_t, death_t), 10)
    return b


class SurvivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = toy_data()
        cls.c = toy_data().sample(frac=1, random_state=9)[survival.RAW_FEATURES].reset_index(drop=True)
        cls.out, cls.audit = survival.synthesize_outcomes(cls.b, cls.c, 10)

    def test_counts_exclusivity_and_admin_censoring(self):
        self.assertEqual(self.out.columns.tolist(), ["time", "onset", "death"])
        self.assertEqual(len(self.out), len(self.b))
        for cause in survival.CAUSES:
            self.assertEqual(int(self.out[cause].sum()), int(self.b[cause].sum()))
        self.assertTrue((self.out.onset + self.out.death <= 1).all())
        self.assertTrue((self.out.loc[self.out.onset + self.out.death == 0, "time"] == 10).all())
        self.assertTrue(self.out.time.between(survival.TIME_FLOOR, 10).all())

    def test_time_bins_preserved_without_source_time_copy(self):
        edges = np.asarray(self.audit["schedule"]["bin_edges"])
        for cause in survival.CAUSES:
            actual = np.histogram(self.out.loc[self.out[cause] == 1, "time"], bins=edges)[0]
            np.testing.assert_array_equal(actual, self.audit["schedule"]["cause_bin_counts"][cause])
        event_times = self.out.loc[self.out.onset + self.out.death == 1, "time"]
        self.assertEqual(len(event_times.unique()), len(event_times))
        self.assertEqual(len(np.intersect1d(event_times, self.b.loc[self.b.time < 10, "time"])), 0)

    def test_reproducible_and_ignores_identifiers_labels(self):
        b = self.b.assign(record_id=np.arange(len(self.b)), is_rare=True)
        got, audit = survival.synthesize_outcomes(b, self.c, 10)
        pd.testing.assert_frame_equal(got, self.out, check_exact=True)
        self.assertEqual(audit, self.audit)
        got, _ = survival.synthesize_outcomes(b, self.c, 10, seed=123)
        self.assertFalse(got.equals(self.out))

    def test_input_frames_unchanged(self):
        b, c = self.b.copy(deep=True), self.c.copy(deep=True)
        survival.synthesize_outcomes(b, c, 10)
        pd.testing.assert_frame_equal(b, self.b)
        pd.testing.assert_frame_equal(c, self.c)

    def test_no_event_source(self):
        b = self.b.assign(time=10.0, onset=0, death=0)
        got, audit = survival.synthesize_outcomes(b, self.c, 10)
        self.assertTrue((got.time == 10).all())
        self.assertEqual(int(got.onset.sum() + got.death.sum()), 0)
        self.assertEqual(audit["fits"]["onset"]["status"], "no_events")

    def test_constant_predictors_get_uniform_assignment(self):
        b, c = self.b.copy(), self.c.copy()
        for col in survival.RAW_FEATURES:
            b[col] = c[col] = 1.0
        got, audit = survival.synthesize_outcomes(b, c, 10)
        self.assertEqual(int(got.onset.sum()), int(b.onset.sum()))
        self.assertEqual(audit["fits"]["onset"]["status"], "no_variable_predictors")

    def test_log_floor(self):
        b = self.b.copy()
        b.loc[0, ["TG", "ALT"]] = 0
        self.assertTrue(np.isfinite(survival.cox_features(b).to_numpy()).all())

    def test_day_rounded_censoring_is_accepted_and_audited(self):
        for delta in [0.5 / 365.25, 1.0 / 365.25]:
            with self.subTest(delta=delta):
                b = self.b.copy()
                admin = b.onset + b.death == 0
                b.loc[admin, "time"] = 10 - delta
                saved = b.copy(deep=True)
                got, audit = survival.synthesize_outcomes(b, self.c, 10)
                generated_admin = got.onset + got.death == 0
                self.assertTrue((got.loc[generated_admin, "time"] == 10).all())
                pd.testing.assert_frame_equal(b, saved)
                details = audit["administrative_censoring"]
                self.assertEqual(details["output_times_canonicalized_count"], int(admin.sum()))
                self.assertAlmostEqual(details["maximum_adjustment_years"], delta)
                self.assertFalse(details["training_times_modified"])
        b.loc[admin, "time"] = 10 - 1.01 / 365.25
        with self.assertRaisesRegex(ValueError, "within one day"):
            survival.synthesize_outcomes(b, self.c, 10)

    def test_tolerance_does_not_relax_strict_horizon_upper_bound(self):
        b = self.b.copy()
        b.loc[b.onset + b.death == 0, "time"] = 10 + 0.5 / 365.25
        with self.assertRaisesRegex(ValueError, "lie in"):
            survival.synthesize_outcomes(b, self.c, 10)

    def test_minimal_ridge_fallback_is_reported(self):
        original_fit = survival.CoxPHFitter.fit

        def fail_unpenalized(model, *args, **kwargs):
            if model.penalizer == 0:
                raise ConvergenceError("Deliberate synthetic-test failure")
            return original_fit(model, *args, **kwargs)

        with patch.object(survival.CoxPHFitter, "fit", fail_unpenalized):
            _, audit = survival.synthesize_outcomes(self.b, self.c, 10)
        for cause in survival.CAUSES:
            fit = audit["fits"][cause]
            self.assertTrue(fit["ridge_fallback"])
            self.assertEqual(fit["penalizer"], 1e-8)
            self.assertEqual(fit["attempts"][0]["reason"], "ConvergenceError")

    def test_failed_all_fits_is_not_silently_replaced(self):
        with patch.object(survival.CoxPHFitter, "fit", side_effect=ConvergenceError("Test failure")):
            with self.assertRaisesRegex(RuntimeError, "every predeclared penalizer"):
                survival.synthesize_outcomes(self.b, self.c, 10)

    def test_invalid_inputs_and_invalid_calibration_bounds_fail(self):
        with self.assertRaises(ValueError):
            survival.synthesize_outcomes(self.b, self.c.iloc[:-1], 10)
        for steps in [-1, 9, 0.5, True]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                survival.synthesize_outcomes(self.b, self.c, 10, calibration_steps=steps)
        with self.assertRaises(ValueError):
            survival.synthesize_outcomes(self.b.assign(onset=1, death=1), self.c, 10)
        b = self.b.copy()
        b.loc[b.onset + b.death == 0, "time"] = 9
        with self.assertRaisesRegex(ValueError, "administrative censoring"):
            survival.synthesize_outcomes(b, self.c, 10)

    def test_sequential_assignment_preserves_risk_direction(self):
        n = 2000
        x = pd.DataFrame(0.0, index=range(n), columns=survival.FEATURES)
        x["FPG"] = np.linspace(-1, 1, n)
        coefficients = {cause: pd.Series(0.0, index=survival.FEATURES) for cause in survival.CAUSES}
        coefficients["onset"]["FPG"] = 4.0
        times = np.linspace(0.1, 9, 200)
        out = survival._assign_events(x, x.mean(), coefficients, times, np.zeros(200, dtype=int),
                                      10, np.random.default_rng(44).random(200))
        self.assertGreater(x.loc[out.onset == 1, "FPG"].mean(), 0.6)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = toy_data()
        cls.c = toy_data().sample(frac=1, random_state=9)[survival.RAW_FEATURES].reset_index(drop=True)
        cls.out, cls.audit = survival.synthesize_outcomes(cls.b, cls.c, 10, calibration_steps=4)

    def test_loss_selection_is_global_and_includes_uncalibrated(self):
        audit = self.audit["calibration"]
        history = audit["history"]
        self.assertEqual(len(history), 5)
        self.assertEqual(audit["evaluated_candidates"], 5)
        expected = min(history, key=lambda record: record["loss"])
        self.assertEqual(audit["selected_iteration"], expected["iteration"])
        self.assertEqual(audit["selected_loss"], expected["loss"])
        self.assertLessEqual(audit["selected_loss"], audit["initial_loss"])
        self.assertFalse(audit["official_score_used"])
        self.assertFalse(audit["significance_threshold_used"])

    def test_both_causes_all_nine_coefficients_and_se_are_audited(self):
        for record in self.audit["calibration"]["history"]:
            detail = record["loss_detail"]
            self.assertEqual(detail["coefficient_count"], 18)
            errors, information = [], []
            for cause in survival.CAUSES:
                self.assertEqual(set(detail["per_coefficient"][cause]), set(survival.FEATURES))
                for entry in detail["per_coefficient"][cause].values():
                    self.assertGreater(entry["target_standard_error"], 0)
                    self.assertGreater(entry["generated_standard_error"], 0)
                    errors.append(((entry["generated_fit_coefficient"] - entry["target_coefficient"])
                                   / entry["target_standard_error"]) ** 2)
                    information.append((2 * np.log(entry["target_standard_error"]
                                                    / entry["generated_standard_error"])) ** 2)
            expected = np.mean(errors) + survival.CALIBRATION_INFORMATION_WEIGHT * np.mean(information)
            self.assertAlmostEqual(record["loss"], expected)

    def test_updates_are_bounded_in_target_se_units(self):
        history = self.audit["calibration"]["history"]
        for record in history:
            for cause in survival.CAUSES:
                target = self.audit["fits"][cause]
                for feature, se in target["standard_errors"].items():
                    current = record["generation_coefficients"][cause][feature]
                    original = target["coefficients"][feature]
                    self.assertLessEqual(abs(current - original) / se, 3.0 + 1e-12)
                    if record["iteration"]:
                        previous = history[record["iteration"] - 1]["generation_coefficients"][cause][feature]
                        self.assertLessEqual(abs(current - previous) / se, 1.0 + 1e-12)

    def test_calibration_reuses_schedule_and_preserves_event_margins(self):
        uncalibrated, initial = survival.synthesize_outcomes(self.b, self.c, 10)
        self.assertEqual(self.audit["schedule"], initial["schedule"])
        self.assertEqual(self.audit["event_counts"], initial["event_counts"])
        for cause in survival.CAUSES:
            np.testing.assert_array_equal(np.sort(self.out.loc[self.out[cause] == 1, "time"]),
                                          np.sort(uncalibrated.loc[uncalibrated[cause] == 1, "time"]))
        self.assertTrue((self.out.onset + self.out.death <= 1).all())

    def test_calibration_is_reproducible_and_ignores_private_labels(self):
        b = self.b.assign(record_id=np.arange(len(self.b)), is_rare=True)
        got, audit = survival.synthesize_outcomes(b, self.c, 10, calibration_steps=4)
        pd.testing.assert_frame_equal(got, self.out, check_exact=True)
        self.assertEqual(audit, self.audit)

    def test_zero_steps_bypasses_calibration_and_retains_original_bytes(self):
        expected, expected_audit = survival.synthesize_outcomes(self.b, self.c, 10)
        with patch.object(survival, "_calibrate", side_effect=AssertionError("Must not run")):
            actual, audit = survival.synthesize_outcomes(self.b, self.c, 10, calibration_steps=0)
        self.assertEqual(actual.to_csv(index=False).encode(), expected.to_csv(index=False).encode())
        self.assertEqual(audit, expected_audit)
        self.assertNotIn("calibration", audit)

    def test_no_event_calibration_selects_first_tie_without_invalid_loss(self):
        b = self.b.assign(time=10.0, onset=0, death=0)
        got, audit = survival.synthesize_outcomes(b, self.c, 10, calibration_steps=8)
        self.assertEqual(audit["calibration"]["selected_iteration"], 0)
        self.assertEqual(audit["calibration"]["selected_loss"], 0.0)
        self.assertEqual(len(audit["calibration"]["history"]), 9)
        self.assertTrue((got.time == 10).all())


if __name__ == "__main__":
    unittest.main()
