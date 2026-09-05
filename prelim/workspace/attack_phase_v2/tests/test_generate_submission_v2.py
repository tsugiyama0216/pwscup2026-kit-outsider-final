from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase.generate_submission import build_aia, fit_view
from attack_phase_v2.generate_submission_v2 import (
    LocalDensityCalibrator,
    _rare_aware_transport,
    blockwise_neighbour_distances,
    build_aia_v2,
    clean_json_v2,
    decode_linked_times,
    effective_link_trust_distance,
    load_candidate_rare_mask,
    safe_abstain_time,
    select_and_link,
    validate_output_frames,
)
from attack_phase.generate_submission import ViewSpec


class AttackV2Tests(unittest.TestCase):
    @staticmethod
    def _population(rng: np.random.Generator, n: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "record_id": np.arange(n),
                "age": rng.integers(40, 75, size=n),
                "BMI": rng.normal(23, 3, size=n),
                "SBP": rng.normal(125, 15, size=n),
                "TG": rng.lognormal(4.5, 0.45, size=n),
                "HDL": rng.normal(50, 9, size=n),
                "ALT": rng.lognormal(2.8, 0.5, size=n),
                "FPG": rng.normal(95, 15, size=n),
                "sex": rng.integers(0, 2, size=n),
                "prefecture": np.repeat("01", n),
                "smoking": rng.integers(0, 2, size=n),
                "time": np.full(n, 10.0),
                "onset": np.zeros(n, dtype=int),
                "death": np.zeros(n, dtype=int),
            }
        )
        return frame

    def test_safe_abstain_is_outside_every_valid_time_window(self) -> None:
        value = safe_abstain_time(10.0, 0.5)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value - 10.0, 0.5)

    def test_json_cleaner_preserves_boolean_type(self) -> None:
        cleaned = clean_json_v2(
            {"python": True, "numpy": np.bool_(False), "integer": np.int64(1)}
        )
        self.assertIs(cleaned["python"], True)
        self.assertIs(cleaned["numpy"], False)
        self.assertEqual(cleaned["integer"], 1)
        self.assertIs(type(cleaned["integer"]), int)

    def test_outcome_anomaly_guard_caps_link_distance(self) -> None:
        normal = pd.DataFrame(
            {"onset": [0, 0, 1, 0], "death": [0, 0, 0, 0]}
        )
        anomalous = pd.DataFrame(
            {"onset": [1, 1, 1, 0], "death": [0, 0, 0, 0]}
        )
        normal_distance, normal_rate, normal_guard = (
            effective_link_trust_distance(normal, 0.20)
        )
        anomaly_distance, anomaly_rate, anomaly_guard = (
            effective_link_trust_distance(anomalous, 0.20)
        )
        self.assertEqual(normal_distance, 0.20)
        self.assertEqual(normal_rate, 0.25)
        self.assertFalse(normal_guard)
        self.assertEqual(anomaly_distance, 0.08)
        self.assertEqual(anomaly_rate, 0.75)
        self.assertTrue(anomaly_guard)

    def test_selection_and_linkage_are_separate_and_unique(self) -> None:
        pair_cost = np.array(
            [
                [0.0, 9.0, 9.0],
                [9.0, 0.0, 9.0],
                [9.0, 9.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        membership = np.array([0.1, 0.9, 0.8, 0.2])
        assigned = select_and_link(pair_cost, membership, 2, np.arange(4))
        self.assertTrue(np.array_equal(np.flatnonzero(assigned >= 0), [1, 2]))
        self.assertEqual(len(np.unique(assigned[assigned >= 0])), 2)

    def test_adaptive_decoder_uses_horizon_for_uncertain_or_censored_rows(self) -> None:
        released = pd.DataFrame(
            {
                "time": [2.0, 9.2, 4.0],
                "onset": [1, 0, 0],
                "death": [0, 0, 1],
            }
        )
        assigned = np.array([0, 1, 2, -1])
        strength = np.zeros(4)
        pair_cost = np.full((4, 3), 5.0)
        pair_cost[0, 0] = 0.02  # trusted event
        pair_cost[1, 1] = 0.01  # censored: always horizon
        pair_cost[2, 2] = 0.20  # event but too far: horizon
        decoded = decode_linked_times(
            released,
            assigned,
            pair_cost,
            strength,
            abstain_time=11.0,
            horizon_time=10.0,
            trust_distance=0.08,
        )
        self.assertTrue(np.allclose(decoded["adaptive"], [2.0, 10.0, 10.0, 11.0]))
        self.assertTrue(np.allclose(decoded["linked"], [2.0, 9.2, 4.0, 11.0]))

    def test_local_density_ratio_prefers_a_released_member(self) -> None:
        rng = np.random.default_rng(123)
        background = self._population(rng, 300)
        query = self._population(rng, 20)
        query["record_id"] += 10_000
        released = self._population(rng, 80)
        released.iloc[0] = query.iloc[0]
        spec = ViewSpec("small", ("age", "SBP", "HDL"))
        fitted = fit_view(background, spec)
        calibrator = LocalDensityCalibrator(
            background,
            query,
            [fitted],
            (1, 3),
            query["record_id"].to_numpy(),
        )
        score, _ = calibrator.score(released)
        self.assertEqual(int(np.argmax(score)), 0)

    def test_blockwise_distances_return_requested_shape(self) -> None:
        rng = np.random.default_rng(456)
        reference = self._population(rng, 50)
        query = self._population(rng, 7)
        spec = ViewSpec("small", ("age", "SBP", "HDL"))
        fitted = fit_view(reference, spec)
        result = blockwise_neighbour_distances(
            reference,
            fitted.transform(reference),
            query,
            fitted.transform(query),
            (1, 3, 5),
        )
        self.assertEqual(result.distances.shape, (7, 3))
        self.assertTrue(np.isfinite(result.distances).all())

    def test_rare_transport_changes_only_rare_rows_before_final_rerank(self) -> None:
        rng = np.random.default_rng(789)
        general = rng.uniform(size=(12, 3))
        hybrid = rng.uniform(size=(12, 3))
        rare = np.zeros(12, dtype=bool)
        rare[:4] = True
        ids = np.arange(12)
        result, diagnostics = _rare_aware_transport(
            general,
            hybrid,
            rare,
            ids,
            total_shared=6,
            epsilon=0.35,
            rare_weight=0.65,
        )
        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(result.shape, general.shape)
        for target in range(result.shape[1]):
            self.assertEqual(len(np.unique(result[:, target])), len(result))

    def test_b_self_is_aligned_by_record_id(self) -> None:
        candidates = pd.DataFrame({"record_id": [20, 10, 30]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame(
                {"record_id": [10, 20, 30], "is_rare": [1, 0, 1]}
            ).to_csv(path / "B_self.csv", index=False)
            mask, diagnostics = load_candidate_rare_mask(path, candidates)
        self.assertTrue(diagnostics["available"])
        self.assertTrue(np.array_equal(mask, [False, True, True]))

    def test_strict_output_validation_accepts_exact_tie_free_frames(self) -> None:
        candidates = pd.DataFrame({"record_id": [10, 20, 30]})
        targets = {"1": pd.DataFrame(), "2": pd.DataFrame()}
        challenges = {
            "1": pd.DataFrame({"challenge_row_id": [1, 2]}),
            "2": pd.DataFrame({"challenge_row_id": [4]}),
        }
        f_mia = pd.DataFrame(
            {
                "record_id": [10, 20, 30],
                "1": [0.1, 0.2, 0.3],
                "2": [0.3, 0.1, 0.2],
            }
        )
        f_aia = pd.DataFrame(
            {
                "target_k": [1, 1, 2],
                "challenge_row_id": [1, 2, 4],
                "time_hat": [10.0, 11.0, 3.0],
            }
        )
        checks = validate_output_frames(
            f_mia, f_aia, candidates, targets, challenges
        )
        self.assertTrue(checks["strict_preflight_passed"])

    def test_strict_output_validation_rejects_mia_ties(self) -> None:
        candidates = pd.DataFrame({"record_id": [10, 20]})
        targets = {"1": pd.DataFrame()}
        challenges = {"1": pd.DataFrame({"challenge_row_id": [1, 2]})}
        f_mia = pd.DataFrame(
            {"record_id": [10, 20], "1": [0.5, 0.5]}
        )
        f_aia = pd.DataFrame(
            {
                "target_k": [1, 1],
                "challenge_row_id": [1, 2],
                "time_hat": [10.0, 11.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "tie-free"):
            validate_output_frames(f_mia, f_aia, candidates, targets, challenges)

    def test_strict_output_validation_rejects_duplicate_aia_key(self) -> None:
        candidates = pd.DataFrame({"record_id": [10, 20]})
        targets = {"1": pd.DataFrame()}
        challenges = {"1": pd.DataFrame({"challenge_row_id": [1, 2]})}
        f_mia = pd.DataFrame(
            {"record_id": [10, 20], "1": [0.4, 0.6]}
        )
        f_aia = pd.DataFrame(
            {
                "target_k": [1, 1],
                "challenge_row_id": [1, 1],
                "time_hat": [10.0, 11.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "composite key"):
            validate_output_frames(f_mia, f_aia, candidates, targets, challenges)

    def test_v1_linked_variant_matches_v1_generator(self) -> None:
        rng = np.random.default_rng(2468)
        background = self._population(rng, 120)
        released = self._population(rng, 10)
        challenge = self._population(rng, 12).drop(
            columns=["time", "onset", "death"]
        )
        challenge = challenge.rename(columns={"record_id": "challenge_row_id"})
        targets = {"1": released}
        challenges = {"1": challenge}
        _, v1_variants, _ = build_aia(
            background,
            targets,
            challenges,
            main_fraction=0.5,
            variant_fractions=[0.5],
            abstain_time=0.0,
        )
        _, v2_variants, _ = build_aia_v2(
            background,
            targets,
            challenges,
            main_fraction=0.5,
            variant_fractions=[0.5],
            abstain_time=11.0,
            baseline_abstain_time=0.0,
            horizon_time=10.0,
        )
        pd.testing.assert_frame_equal(
            v2_variants["v1_linked_0p5"].reset_index(drop=True),
            v1_variants["0p5"].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
