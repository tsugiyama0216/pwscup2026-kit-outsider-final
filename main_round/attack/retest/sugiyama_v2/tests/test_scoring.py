"""Aggregation, row identity, floor provenance and reference-cache regressions."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from pwscup2026.attack.aia import aia_predict
from pwscup2026.kit.codabench.scoring_io import score_attack
from sugiyama_v2.scoring import (FLOOR_METADATA, FLOOR_METHOD, TARGET_IDS,
                                 _aia_result, _reference_time_samples, _scratch_directory,
                                 estimate_floors, score_world, validate_predictions)


CFG = {"scoring": {"low_fpr": .01, "aia_rare_se_z": 1.64},
       "aia": {"tau_time": .5, "qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"],
               "block_keys": ["sex", "prefecture"], "k_nn_onset": 5}}


def fixture(directory):
    root = Path(directory)
    candidates = pd.DataFrame({"record_id": np.arange(1049)})
    attacker_pool = pd.DataFrame({"record_id": np.arange(1049), "pool_id": np.arange(10000, 11049),
                                  "is_rare": np.arange(1049) < 24})
    attacker_pool.to_csv(root / "pool_ids_10.csv", index=False)
    public = {"team_id": 10, "candidates": candidates, "targets": {}}
    mia = candidates.copy()
    aia = []
    floors = []
    floor_meta = {"method": FLOOR_METHOD, "mock_truth_only": True,
                  "official_hidden_floors_known": False, "repetitions": 1000,
                  "sample_rows": 1049, "per_target": {}}
    for index, k in enumerate(TARGET_IDS):
        pool_ids = np.arange(200000 + k * 2000, 200000 + k * 2000 + 1049)
        # Only the first target has a rare positive among attacker candidates.
        pool_ids[24:524] = attacker_pool.pool_id.to_numpy()[24:524]
        if index == 0:
            pool_ids[0] = attacker_pool.pool_id.iloc[0]
        target = pd.DataFrame({"record_id": np.arange(1049), "pool_id": pool_ids,
                               "is_rare": np.arange(1049) < 24})
        target.to_csv(root / f"pool_ids_{k}.csv", index=False)
        mia[str(k)] = .5
        if index == 0:
            mia.loc[0, str(k)] = 1.
        query_ids = np.arange(3000, 3326)
        public["targets"][k] = {"challenge": pd.DataFrame({"challenge_row_id": query_ids})}
        member = np.arange(326) < 163
        rare = (np.arange(326) % 163) < 13
        truth = pd.DataFrame({"challenge_row_id": query_ids, "label": np.where(member, "member", "control"),
                              "time": 2., "onset": 0, "is_rare": rare})
        truth.to_csv(root / f"aia_truth_{k}.csv", index=False)
        pred = np.where(member & (index < 11), 2., 5.)
        aia.append(pd.DataFrame({"target_k": k, "challenge_row_id": query_ids, "time_hat": pred}))
        floors.append({"team_id": k, "aia_floor": .2})
        floor_meta["per_target"][str(k)] = {"mean": .2, "monte_carlo_se": .01}
    pd.DataFrame(floors).to_csv(root / "aia_floor.csv", index=False)
    (root / FLOOR_METADATA).write_text(json.dumps(floor_meta), encoding="utf-8")
    return public, mia, pd.concat(aia, ignore_index=True)


class OfficialAggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp = _scratch_directory("test-")
        self.root = self.temp.__enter__()
        self.addCleanup(self.temp.__exit__, None, None, None)
        self.public, self.mia, self.aia = fixture(self.root)

    def test_same_saved_truth_official_parity_and_max_before_mean(self):
        actual = score_world(self.mia, self.aia, self.public, self.root, CFG)
        expected = score_attack({"F_mia.csv": self.mia, "F_aia.csv": self.aia}, self.root, CFG, team_id=10)
        for key, value in expected.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual["denominators"]["mia_rare_defined_targets"], 1)
        self.assertEqual(actual["mia_rare"], 1.)
        self.assertAlmostEqual(actual["mia_power"], (1. + 21 * .01) / 22)
        self.assertLess(actual["mia_power"], max(actual["mia_all"], actual["mia_rare"]))
        self.assertAlmostEqual(actual["aia_power"], .4)
        self.assertAlmostEqual(actual["pre_floor"]["aia_power"], .5)
        # Per-target clipping retains 11*(1-.2) and 11*0, unlike clipping mean.
        self.assertNotAlmostEqual(actual["aia_power"], max(.5 - .2, 0.))
        self.assertEqual(actual["leaderboard"]["score_1"], actual["attack_power"])
        self.assertEqual(actual["leaderboard"]["score_4"], actual["aia_all"])

    def test_row_and_target_column_permutation_preserve_score(self):
        first = score_world(self.mia, self.aia, self.public, self.root, CFG)
        permuted_m = self.mia.sample(frac=1, random_state=1)[self.mia.columns[::-1]]
        permuted_a = self.aia.sample(frac=1, random_state=2)
        second = score_world(permuted_m, permuted_a, self.public, self.root, CFG)
        self.assertEqual(first["leaderboard"], second["leaderboard"])

    def test_none_is_explicit_and_missing_floor_fails_closed(self):
        (self.root / "aia_floor.csv").unlink()
        with self.assertRaisesRegex(ValueError, "missing is not zero"):
            score_world(self.mia, self.aia, self.public, self.root, CFG)
        result = score_world(self.mia, self.aia, self.public, self.root, CFG, floor_mode="none")
        self.assertEqual(result["floor_mode"], "none")
        self.assertIsNone(result["floor_estimate"])
        self.assertAlmostEqual(result["aia_power"], .5)

    def test_incomplete_and_nonfinite_floor_rejected(self):
        for replacement in (pd.DataFrame({"team_id": [TARGET_IDS[0]], "aia_floor": [.2]}),
                            pd.DataFrame({"team_id": TARGET_IDS, "aia_floor": [np.inf] * 22})):
            with self.subTest(rows=len(replacement)):
                replacement.to_csv(self.root / "aia_floor.csv", index=False)
                with self.assertRaises(ValueError):
                    score_world(self.mia, self.aia, self.public, self.root, CFG)

    def test_floor_metadata_mean_mismatch_rejected(self):
        frame = pd.read_csv(self.root / "aia_floor.csv")
        frame.loc[0, "aia_floor"] = .7
        frame.to_csv(self.root / "aia_floor.csv", index=False)
        with self.assertRaisesRegex(ValueError, "recorded mean"):
            score_world(self.mia, self.aia, self.public, self.root, CFG)

    def test_invalid_prediction_sets_rejected(self):
        invalid = []
        invalid.append((self.mia.iloc[:-1], self.aia))
        invalid.append((self.mia, self.aia.iloc[:-1]))
        invalid.append((self.mia.drop(columns=str(TARGET_IDS[-1])), self.aia))
        invalid.append((self.mia.assign(**{"10": .1}), self.aia))
        for value in (np.nan, np.inf, -1., 1.01):
            frame = self.mia.copy()
            frame.loc[0, str(TARGET_IDS[0])] = value
            invalid.append((frame, self.aia))
        for value in (np.nan, np.inf, -1.):
            frame = self.aia.copy()
            frame.loc[0, "time_hat"] = value
            invalid.append((self.mia, frame))
        frame = self.mia.copy()
        frame.loc[0, "record_id"] = 1
        invalid.append((frame, self.aia))
        frame = self.aia.copy()
        frame["challenge_row_id"] = frame["challenge_row_id"].astype(float)
        frame.loc[0, "challenge_row_id"] = 3000.5
        invalid.append((self.mia, frame))
        for index, (mia, aia) in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_predictions(mia, aia, self.public)

    def test_missing_or_wrong_truth_is_not_silently_inner_joined(self):
        path = self.root / f"aia_truth_{TARGET_IDS[0]}.csv"
        frame = pd.read_csv(path).iloc[:-1]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "challenge IDs differ"):
            score_world(self.mia, self.aia, self.public, self.root, CFG)

    def test_truth_rare_labels_required_not_reconstructed(self):
        path = self.root / "pool_ids_10.csv"
        frame = pd.read_csv(path).drop(columns="is_rare")
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "explicit rare truth"):
            score_world(self.mia, self.aia, self.public, self.root, CFG)

    def test_floor_estimator_matches_direct_official_draws(self):
        rng = np.random.default_rng(419)
        bg = pd.DataFrame({name: rng.normal(50, 10, 1100) for name in CFG["aia"]["qi_continuous"]})
        bg["sex"] = rng.integers(0, 2, 1100)
        bg["prefecture"] = rng.integers(0, 3, 1100)
        bg["onset"] = rng.integers(0, 2, 1100)
        bg["time"] = rng.uniform(0, 10, 1100)
        self.public["background"] = bg
        query = bg.iloc[:326].drop(columns=["time", "onset"]).copy()
        query.insert(0, "challenge_row_id", np.arange(3000, 3326))
        for target in self.public["targets"].values():
            target["challenge"] = query
        floors, metadata = estimate_floors(self.public, self.root, CFG, seed=712, repetitions=3)
        draws = np.random.default_rng(712)
        truth = pd.read_csv(self.root / f"aia_truth_{TARGET_IDS[0]}.csv")
        expected = []
        for _ in range(3):
            selected = draws.choice(len(bg), 1049, replace=False)
            _, predictions = aia_predict(bg.iloc[selected].reset_index(drop=True), query, bg, CFG)
            expected.append(_aia_result(predictions, truth, CFG).score)
        self.assertEqual(floors.iloc[0].aia_floor, float(np.mean(expected)))
        self.assertEqual(metadata["per_target"][str(TARGET_IDS[0])]["monte_carlo_se"],
                         float(np.std(expected, ddof=1)) / np.sqrt(3))
        self.assertTrue(metadata["diagnostic_reduced_repetitions"])
        self.assertFalse(metadata["official_hidden_floors_known"])


class CachedReferenceTests(unittest.TestCase):
    def test_cached_reference_matches_official_ties_and_fallbacks(self):
        bg = pd.DataFrame({"age": [45., 45., 55., 60., 70., 60.], "SBP": [110., 110., 130., 140., 150., 140.],
                           "TG": [100.] * 6, "HDL": [50.] * 6, "ALT": [20.] * 6,
                           "sex": [0, 0, 0, 1, 1, 1], "prefecture": [1, 1, 2, 1, 2, 1],
                           "onset": [0, 1, 0, 1, 0, 0], "time": [1., 3., 4., 5., 9., 7.]})
        query = pd.concat([bg.iloc[[0]], bg.iloc[[3]], bg.iloc[[4]]], ignore_index=True)
        query.loc[2, "prefecture"] = 999  # first-key fallback
        query = pd.concat([query, query.iloc[[0]]], ignore_index=True)
        query.loc[3, "sex"] = 7  # full-background fallback
        samples = np.array([[0, 1, 2], [3, 4, 5], [0, 3, 5], [1, 2, 4]])
        cached = _reference_time_samples(bg, query, samples, CFG)
        for r, selected in enumerate(samples):
            _, expected = aia_predict(bg.iloc[selected].reset_index(drop=True), query, bg, CFG)
            np.testing.assert_array_equal(cached[r], expected)


if __name__ == "__main__":
    unittest.main()
