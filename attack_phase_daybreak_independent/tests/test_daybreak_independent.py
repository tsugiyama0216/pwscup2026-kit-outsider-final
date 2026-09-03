from __future__ import annotations

import argparse
import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (
    AIA_SUBSPACES,
    EXPECTED_AIA_MEMBERS,
    EXPECTED_TARGET_IDS,
    OutcomeIntegrityFingerprint,
    RobustAtomEncoder,
    _consensus_codebook_link,
    _route_outcome_integrity,
    discover_known_private,
    strict_output_checks,
    strict_rank,
    validate_settings,
)


class DaybreakIndependentTests(unittest.TestCase):
    def expected_settings(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "self_target_id": "24",
            "exclude_targets": ("24",),
            "mia_total_shared": 508,
            "aia_member_count": 163,
            "aia_residual_weight": 0.25,
            "aia_background_neighbours": 48,
            "aia_integrity_auc_threshold": 0.60,
            "acknowledge_prelim_private_block_rule": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_settings_are_locked_to_preliminary_team24(self) -> None:
        validate_settings(self.expected_settings())
        with self.assertRaises(ValueError):
            validate_settings(self.expected_settings(self_target_id="23"))
        with self.assertRaises(ValueError):
            validate_settings(self.expected_settings(mia_total_shared=736))

    def test_production_module_has_no_legacy_attack_imports(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "generate_submission_daybreak_independent.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden = (
            "attack_phase",
            "attack_phase_v2",
            "attack_phase_v3",
            "attack_phase_daybreak",
        )
        self.assertFalse(
            any(name == prefix or name.startswith(prefix + ".") for name in imported for prefix in forbidden)
        )

    def test_strict_rank_is_tie_free_and_deterministic(self) -> None:
        values = np.asarray([1.0, 1.0, 0.0, 2.0])
        keys = np.asarray([2, 1, 3, 0])
        first = strict_rank(values, keys)
        second = strict_rank(values, keys)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), len(values))
        self.assertGreater(first[3], first[0])

    def test_encoder_supports_qi_only_challenge(self) -> None:
        background = pd.DataFrame(
            {
                "age": [40, 50, 60], "BMI": [20.0, 25.0, 30.0],
                "SBP": [110.0, 120.0, 130.0], "TG": [50.0, 100.0, 150.0],
                "HDL": [40.0, 50.0, 60.0], "ALT": [10.0, 20.0, 30.0],
                "FPG": [80.0, 90.0, 100.0], "time": [3.0, 5.0, 10.0],
                "sex": [0, 1, 0], "smoking": [0, 1, 0],
                "onset": [1, 0, 0], "death": [0, 0, 1],
                "prefecture": ["A", "B", "A"],
            }
        )
        challenge = background[["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]]
        encoder = RobustAtomEncoder.fit(background)
        matrix = encoder.matrix(challenge, AIA_SUBSPACES[0])
        self.assertEqual(matrix.shape[0], len(challenge))
        self.assertTrue(np.isfinite(matrix).all())

    def test_known_private_exactly_recovers_163_qi_rows(self) -> None:
        rows = 1_049
        candidates = pd.DataFrame(
            {
                "age": 40 + np.arange(rows) % 35,
                "SBP": 100.0 + np.arange(rows) * 0.01,
                "TG": 50.0 + np.arange(rows) * 0.02,
                "HDL": 30.0 + np.arange(rows) * 0.03,
                "ALT": 5.0 + np.arange(rows) * 0.04,
                "sex": np.arange(rows) % 2,
                "prefecture": [f"P{index}" for index in range(rows)],
            }
        )
        self_part = candidates.iloc[:EXPECTED_AIA_MEMBERS].copy()
        controls = candidates.iloc[EXPECTED_AIA_MEMBERS : 2 * EXPECTED_AIA_MEMBERS].copy()
        controls["prefecture"] = [f"X{index}" for index in range(len(controls))]
        challenge = pd.concat([self_part, controls], ignore_index=True)
        mask, details = discover_known_private(candidates, challenge)
        self.assertEqual(int(mask.sum()), EXPECTED_AIA_MEMBERS)
        self.assertEqual(details["known_negative_rows"], EXPECTED_AIA_MEMBERS)

    def test_consensus_link_allows_reuse_without_hungarian(self) -> None:
        nearest = np.asarray([[1, 1, 2], [1, 1, 0]], dtype=int)
        pair = np.asarray([[3.0, 0.1, 0.2], [0.4, 0.1, 0.3]])
        linked, agreement = _consensus_codebook_link(nearest, pair)
        self.assertTrue(np.array_equal(linked, np.asarray([1, 1])))
        self.assertTrue(np.allclose(agreement, 2.0 / 3.0))

    def test_integrity_router_uses_horizon_for_single_class_release(self) -> None:
        rows = 40
        event = np.arange(rows) >= rows // 2
        background = pd.DataFrame(
            {
                "age": 30 + np.arange(rows),
                "SBP": 100 + 2 * np.arange(rows),
                "TG": 50 + 3 * np.arange(rows),
                "HDL": 80 - np.arange(rows),
                "ALT": 10 + np.arange(rows),
                "sex": np.arange(rows) % 2,
                "prefecture": np.where(np.arange(rows) % 2, "B", "A"),
                "time": np.where(event, 4.0, 10.0),
                "onset": event.astype(int),
                "death": np.zeros(rows, dtype=int),
            }
        )
        aligned = background.copy()
        single_class = background.copy()
        single_class[["onset", "death"]] = 0
        measure = OutcomeIntegrityFingerprint(background).measure(aligned)
        self.assertGreater(measure["auc"], 0.90)

        keys = pd.DataFrame(
            {
                "target_k": [1, 1, 2, 2],
                "challenge_row_id": [0, 1, 0, 1],
            }
        )
        adaptive = keys.assign(time_hat=[1.0, 2.0, 3.0, 4.0])
        horizon = keys.assign(time_hat=[10.0, 10.0, 10.0, 10.0])
        routed, profiles = _route_outcome_integrity(
            background,
            {"1": aligned, "2": single_class},
            adaptive,
            horizon,
            auc_threshold=0.60,
        )
        self.assertEqual(profiles["1"]["decoder"], "adaptive_residual")
        self.assertEqual(profiles["2"]["decoder"], "horizon")
        self.assertTrue(
            np.array_equal(
                routed.loc[routed["target_k"].eq(2), "time_hat"].to_numpy(),
                np.asarray([10.0, 10.0]),
            )
        )

    def test_strict_output_rejects_infinite_aia_and_duplicate_keys(self) -> None:
        record_ids = np.arange(1_049)
        mia = pd.DataFrame({"record_id": record_ids})
        for offset, target_id in enumerate(EXPECTED_TARGET_IDS):
            mia[target_id] = (record_ids + 1 + offset / 100.0) / 1_051.0
        challenges: dict[str, pd.DataFrame] = {}
        aia_parts: list[pd.DataFrame] = []
        for target_id in EXPECTED_TARGET_IDS:
            challenge = pd.DataFrame({"challenge_row_id": np.arange(326)})
            challenges[target_id] = challenge
            aia_parts.append(
                pd.DataFrame(
                    {
                        "target_k": int(target_id),
                        "challenge_row_id": np.arange(326),
                        "time_hat": np.full(326, 11.0),
                    }
                )
            )
        aia = pd.concat(aia_parts, ignore_index=True)
        candidates = pd.DataFrame({"record_id": record_ids})
        strict_output_checks(mia, aia, candidates, challenges)
        broken = aia.copy()
        broken.loc[0, "time_hat"] = np.inf
        with self.assertRaises(ValueError):
            strict_output_checks(mia, broken, candidates, challenges)
        duplicate = pd.concat([aia, aia.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            strict_output_checks(mia, duplicate, candidates, challenges)


if __name__ == "__main__":
    unittest.main()
