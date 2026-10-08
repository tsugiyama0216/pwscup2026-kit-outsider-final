"""Focused behavioral checks; these fixtures contain no contest records or truth."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "plugin.py"
spec = importlib.util.spec_from_file_location("predictor_under_test", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


def rows(n=40, low=2.0, high=8.0, prefecture="A"):
    half = np.arange(n) >= n // 2
    return pd.DataFrame({
        "record_id": np.arange(n) + 100,
        "age": np.where(half, 65.0, 25.0),
        "SBP": np.where(half, 140.0, 110.0),
        "TG": np.full(n, 100.0), "HDL": np.full(n, 50.0),
        "ALT": np.full(n, 20.0), "sex": np.zeros(n, dtype=int),
        "prefecture": [prefecture] * n,
        "time": np.where(half, high, low),
    })


def world():
    background = rows(low=2.0, high=8.0)
    release = rows(low=4.0, high=6.0, prefecture="UNSEEN")
    candidates = rows(n=4, low=4.0, high=8.0)
    challenge = candidates.drop(columns=["record_id", "time"]).copy()
    challenge["challenge_row_id"] = [7, 3, 11, 2]
    return dict(candidates=candidates, background=background,
                targets={2: {"release": release, "challenge": challenge}},
                attacker_id=24)


class PredictorTests(unittest.TestCase):
    def test_residual_score_direction_and_shared_aia_model(self):
        result = plugin.predict_world(**world())
        # Background predicts [2, 2, 8, 8], C predicts [4, 4, 6, 6].
        # The first pair is closer to C, the second pair to background.
        self.assertGreater(result["F_mia"]["2"].iloc[0], result["F_mia"]["2"].iloc[2])
        np.testing.assert_allclose(result["F_aia"].time_hat, [4, 4, 6, 6])
        np.testing.assert_allclose(plugin.residual_difference(
            np.array([4, 8]), np.array([2, 8]), np.array([4, 6])), [2, -2])

    def test_preprocessor_is_fitted_on_background_only_and_is_not_refitted(self):
        data = world()
        processor = plugin.fit_preprocessor(data["background"])
        encoder = processor.named_transformers_["categorical"]
        before = [array.copy() for array in encoder.categories_]
        model = plugin.SyntheticPredictor(processor).fit(data["targets"][2]["release"])
        for actual, expected in zip(encoder.categories_, before):
            np.testing.assert_array_equal(actual, expected)
        self.assertNotIn("UNSEEN", encoder.categories_[1])
        self.assertTrue(np.isfinite(model.predict(data["targets"][2]["challenge"])).all())

    def test_candidate_time_affects_only_mia_not_training_or_aia(self):
        original = world()
        first = plugin.predict_world(**original)
        changed = copy.deepcopy(original)
        changed["candidates"]["time"] = [2.0, 2.0, 6.0, 6.0]
        second = plugin.predict_world(**changed)
        assert_frame_equal(first["F_aia"], second["F_aia"])
        self.assertFalse(first["F_mia"]["2"].equals(second["F_mia"]["2"]))
        self.assertLess(second["F_mia"]["2"].iloc[0], second["F_mia"]["2"].iloc[2])

    def test_non_qi_attributes_and_unused_arguments_have_no_effect(self):
        original = world()
        first = plugin.predict_world(**original)
        changed = copy.deepcopy(original)
        for frame in [changed["candidates"], changed["background"],
                      changed["targets"][2]["release"], changed["targets"][2]["challenge"]]:
            for col in ["BMI", "FPG", "smoking", "onset", "death", "is_rare"]:
                frame[col] = "this field must never be read"
        changed.update(self_labels=object(), self_challenge=object(), schema=object(),
                       cfg=object(), self_release=object(), utility_reference=object())
        second = plugin.predict_world(**changed)
        assert_frame_equal(first["F_mia"], second["F_mia"])
        assert_frame_equal(first["F_aia"], second["F_aia"])

    def test_output_identifiers_finite_bounds_and_determinism(self):
        data = world()
        snapshot = copy.deepcopy(data)
        data["targets"][24] = copy.deepcopy(data["targets"][2])
        first = plugin.predict_world(**data)
        second = plugin.predict_world(**data)
        assert_frame_equal(first["F_mia"], second["F_mia"])
        assert_frame_equal(first["F_aia"], second["F_aia"])
        self.assertEqual(list(first["F_mia"].columns), ["record_id", "2"])
        self.assertEqual(first["F_aia"].challenge_row_id.tolist(), [7, 3, 11, 2])
        self.assertEqual(first["F_mia"].record_id.tolist(), snapshot["candidates"].record_id.tolist())
        self.assertTrue(first["F_mia"]["2"].between(0, 1).all())
        self.assertTrue(first["F_aia"].time_hat.between(0, 10, inclusive="right").all())
        assert_frame_equal(data["candidates"], snapshot["candidates"])
        assert_frame_equal(data["background"], snapshot["background"])

    def test_invalid_public_values_are_rejected(self):
        for column, bad in [("time", np.nan), ("time", 0.0), ("SBP", np.inf), ("sex", 2)]:
            with self.subTest(column=column, bad=bad):
                data = world()
                data["targets"][2]["release"].loc[0, column] = bad
                with self.assertRaises(ValueError):
                    plugin.predict_world(**data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
