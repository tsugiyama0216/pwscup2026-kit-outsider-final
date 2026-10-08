"""Meaningful small-data checks; no contest tables or truth are opened."""
from __future__ import annotations

import sys
import unittest
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from explore_3_gen_lra.model import FEATURES, FixedKernelGenLRA, _log_softplus
from explore_3_gen_lra.plugin import CONFIG, predict_world


def table(values, ids=False):
    frame = pd.DataFrame(values, columns=list(FEATURES))
    frame["sex"] = 0
    frame["prefecture"] = 1
    frame["time"] = np.linspace(1.0, 5.0, len(frame))
    if ids:
        frame["record_id"] = np.arange(len(frame)) + 100
    return frame


class GenLRATests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(90210)
        self.ref = table(self.rng.normal(size=(35, 7)))
        self.synth = table(self.rng.normal(size=(13, 7)))
        self.candidates = table(self.rng.normal(size=(5, 7)), ids=True)

    def test_equivalence_to_explicit_fixed_kernel_augmented_density(self):
        model = FixedKernelGenLRA(self.ref, k=4, batch_size=2)
        actual = model.score(self.synth, self.candidates)
        r = model.reference
        s = model._transform(self.synth)
        q = model._transform(self.candidates)
        covariance = model.covariance * model.bandwidth ** 2
        density = multivariate_normal(mean=np.zeros(7), cov=covariance)
        base = np.array([np.mean(density.pdf(point - r)) for point in s])
        expected = []
        for point in q:
            nn = np.argsort(cdist(point[None], s)[0], kind="stable")[:4]
            augmented = (len(r) * base[nn] + density.pdf(s[nn] - point)) / (len(r) + 1)
            gen_lra = np.sum(np.log(augmented) - np.log(base[nn]))
            positive_evidence = gen_lra + 4 * np.log1p(1 / len(r))
            expected.append(np.log(positive_evidence))
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    def test_negative_tail_scores_stay_finite_and_distinct(self):
        model = FixedKernelGenLRA(self.ref, k=1)
        release = table(np.zeros((1, 7)))
        far = table(np.repeat(np.array([[1e3], [2e3], [3e3]]), 7, axis=1))
        raw = model.score(release, far)
        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(np.all(np.diff(raw) < 0))
        # Direct exponentiation would erase all distinctions in these tails.
        np.testing.assert_array_equal(np.exp(raw), np.zeros(3))

    def test_log_softplus_handles_both_extreme_tails(self):
        output = _log_softplus(np.array([-1000000.0, 0.0, 1000000.0]))
        np.testing.assert_allclose(output, [-1000000.0, np.log(np.log(2)), np.log(1000000.0)])

    def test_duplicate_release_rows_add_evidence_without_changing_order(self):
        small = self.synth.iloc[:3]
        duplicated = pd.concat([small, small], ignore_index=True)
        once = FixedKernelGenLRA(self.ref, k=3).score(small, self.candidates)
        twice = FixedKernelGenLRA(self.ref, k=6).score(duplicated, self.candidates)
        np.testing.assert_allclose(twice, once + np.log(2), atol=1e-12)

    def test_batch_size_does_not_change_result(self):
        a = FixedKernelGenLRA(self.ref, batch_size=1).score(self.synth, self.candidates)
        b = FixedKernelGenLRA(self.ref, batch_size=256).score(self.synth, self.candidates)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_constant_columns_and_missing_values_are_supported(self):
        ref = self.ref.copy()
        ref["age"] = 42
        ref["TG"] = 2 * ref["HDL"]
        ref.loc[0, "BMI"] = np.nan
        query = self.candidates.copy()
        query.loc[0, "BMI"] = np.nan
        raw = FixedKernelGenLRA(ref).score(self.synth, query)
        self.assertTrue(np.isfinite(raw).all())

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            FixedKernelGenLRA(self.ref.iloc[:1])
        with self.assertRaisesRegex(ValueError, "Missing continuous"):
            FixedKernelGenLRA(self.ref.drop(columns="BMI"))
        missing = self.ref.copy()
        missing["age"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite observation"):
            FixedKernelGenLRA(missing)
        infinite = self.ref.copy()
        infinite.loc[0, "age"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            FixedKernelGenLRA(infinite)
        with self.assertRaisesRegex(ValueError, "nonempty release"):
            FixedKernelGenLRA(self.ref).score(self.synth.iloc[:0], self.candidates)

    def test_repeat_candidates_have_equal_scores_without_random_tie_breaks(self):
        duplicated = pd.concat([self.candidates.iloc[:1]] * 2, ignore_index=True)
        raw = FixedKernelGenLRA(self.ref).score(self.synth, duplicated)
        self.assertEqual(raw[0], raw[1])

    def test_public_api_shape_order_and_input_preservation(self):
        challenge = self.candidates.drop(columns=["record_id", "time"]).copy()
        challenge["challenge_row_id"] = [90, 40, 10, 70, 20]
        originals = [df.copy(deep=True) for df in (self.ref, self.synth, self.candidates, challenge)]
        targets = {3: {"release": self.synth, "challenge": challenge}}
        # Unused private/calibration arguments must not be inspected.
        unused = object()
        with patch("builtins.open", side_effect=AssertionError("Inference cannot open files")):
            result = predict_world(candidates=self.candidates, self_labels=unused,
                                   background=self.ref, self_challenge=unused,
                                   targets=targets, schema=unused, cfg=unused)
        self.assertEqual(list(result), ["F_mia", "F_aia", "diagnostics"])
        self.assertEqual(result["diagnostics"]["fixed_config"], CONFIG)
        self.assertEqual(result["diagnostics"]["reference_rows"], len(self.ref))
        json.dumps(result["diagnostics"], allow_nan=False)
        self.assertEqual(list(result["F_mia"]), ["record_id", "3"])
        np.testing.assert_array_equal(result["F_mia"].record_id, self.candidates.record_id)
        self.assertTrue(result["F_mia"]["3"].between(0, 1).all())
        self.assertEqual(list(result["F_aia"]), ["target_k", "challenge_row_id", "time_hat"])
        np.testing.assert_array_equal(result["F_aia"].challenge_row_id, challenge.challenge_row_id)
        self.assertTrue(np.isfinite(result["F_aia"].time_hat).all())
        for original, current in zip(originals, (self.ref, self.synth, self.candidates, challenge)):
            pd.testing.assert_frame_equal(original, current)


    def test_distance_boundary_ties_use_feature_order_not_release_order(self):
        # Mean age is exactly zero; +/-1 therefore tie in standardized distance.
        # Reference density is asymmetric, so selecting +1 vs -1 changes evidence.
        values = np.zeros((5, 7))
        values[:, 0] = [-2, -2, 1, 1, 2]
        model = FixedKernelGenLRA(table(values), k=1)
        release_values = np.zeros((3, 7))
        release_values[:, 0] = [1, -1, 1]
        release = table(release_values)
        query = table(np.zeros((1, 7)))
        left = model.score(release.iloc[[1]], query)
        right = model.score(release.iloc[[0]], query)
        self.assertNotEqual(left[0], right[0])
        for order in itertools.permutations(range(3)):
            np.testing.assert_array_equal(
                model.score(release.iloc[list(order)], query), left
            )

    def test_general_release_permutation_invariance(self):
        model = FixedKernelGenLRA(self.ref, k=4)
        expected = model.score(self.synth, self.candidates)
        for seed in range(3):
            permuted = self.synth.sample(frac=1, random_state=seed)
            np.testing.assert_array_equal(model.score(permuted, self.candidates), expected)

    def test_fixed_metadata_matches_saved_config(self):
        saved = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(CONFIG, saved)

    def test_numeric_target_order_and_progress(self):
        challenge = self.candidates.drop(columns=["record_id", "time"]).copy()
        challenge["challenge_row_id"] = np.arange(len(challenge))
        target = {"release": self.synth, "challenge": challenge}
        with patch("builtins.print") as progress:
            result = predict_world(candidates=self.candidates, self_labels=None,
                                   background=self.ref, self_challenge=None,
                                   targets={"12": target, "3": target}, schema=None, cfg=None)
        self.assertEqual(list(result["F_mia"]), ["record_id", "3", "12"])
        self.assertEqual(progress.call_count, 2)
        self.assertIn("target 3", progress.call_args_list[0].args[0])
        self.assertIn("target 12", progress.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
