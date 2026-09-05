from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v3.generate_submission_v3 import (
    PRIVATE_LINK_COLUMNS,
    blend_mia_frames,
    demote_known_private,
    discover_known_private_records,
)


class AttackV3Tests(unittest.TestCase):
    @staticmethod
    def _candidates(n: int = 8) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "record_id": np.arange(100, 100 + n),
                "age": np.arange(40, 40 + n),
                "SBP": np.arange(110.0, 110.0 + n),
                "TG": np.arange(70.0, 70.0 + n),
                "HDL": np.arange(45.0, 45.0 + n),
                "ALT": np.arange(15.0, 15.0 + n),
                "sex": np.arange(n) % 2,
                "prefecture": [f"P{i}" for i in range(n)],
            }
        )

    @staticmethod
    def _challenge(candidates: pd.DataFrame) -> pd.DataFrame:
        members = candidates.iloc[[1, 3, 5, 7]][list(PRIVATE_LINK_COLUMNS)].copy()
        controls = members.copy()
        controls["age"] += 100
        controls["prefecture"] = [f"C{i}" for i in range(len(controls))]
        challenge = pd.concat([members, controls], ignore_index=True)
        challenge.insert(0, "challenge_row_id", np.arange(len(challenge)))
        return challenge

    def test_self_challenge_exact_half_recovers_known_private_ids(self) -> None:
        candidates = self._candidates()
        challenge = self._challenge(candidates)
        mask, diagnostics = discover_known_private_records(candidates, challenge)
        self.assertEqual(candidates.loc[mask, "record_id"].tolist(), [101, 103, 105, 107])
        self.assertEqual(diagnostics["matched_rows"], 4)
        self.assertEqual(diagnostics["unmatched_controls"], 4)

    def test_self_challenge_rejects_non_half_match_count(self) -> None:
        candidates = self._candidates()
        challenge = self._challenge(candidates)
        challenge.loc[0, "age"] += 200
        with self.assertRaisesRegex(ValueError, "半数"):
            discover_known_private_records(candidates, challenge)

    def test_self_challenge_rejects_duplicate_composite_keys(self) -> None:
        candidates = self._candidates()
        challenge = self._challenge(candidates)
        challenge.loc[1, list(PRIVATE_LINK_COLUMNS)] = challenge.loc[
            0, list(PRIVATE_LINK_COLUMNS)
        ].to_numpy()
        with self.assertRaisesRegex(ValueError, "一意"):
            discover_known_private_records(candidates, challenge)

    def test_self_challenge_rejects_missing_link_value(self) -> None:
        candidates = self._candidates()
        challenge = self._challenge(candidates)
        challenge.loc[0, "SBP"] = np.nan
        with self.assertRaisesRegex(ValueError, "欠測"):
            discover_known_private_records(candidates, challenge)

    def test_demotion_puts_private_rows_below_all_eligible_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "record_id": [10, 20, 30, 40, 50, 60],
                "1": [0.91, 0.20, 0.80, 0.70, 0.60, 0.10],
                "2": [0.10, 0.99, 0.20, 0.70, 0.60, 0.50],
            }
        )
        private = np.array([True, True, False, False, False, False])
        result = demote_known_private(frame, private)
        for target in ("1", "2"):
            self.assertLess(
                result.loc[private, target].max(),
                result.loc[~private, target].min(),
            )
            eligible_before = frame.loc[~private].sort_values(target)["record_id"].tolist()
            eligible_after = result.loc[~private].sort_values(target)["record_id"].tolist()
            self.assertEqual(eligible_before, eligible_after)
            self.assertEqual(result[target].nunique(), len(result))

    def test_demotion_preserves_record_order_and_valid_range(self) -> None:
        frame = pd.DataFrame(
            {
                "record_id": [50, 10, 30, 20],
                "1": [0.4, 0.3, 0.2, 0.1],
            }
        )
        result = demote_known_private(frame, np.array([False, True, False, False]))
        self.assertEqual(result["record_id"].tolist(), frame["record_id"].tolist())
        self.assertTrue(result["1"].between(0.0, 1.0).all())
        self.assertEqual(result["1"].nunique(), len(result))

    def test_demotion_rejects_empty_or_misaligned_mask(self) -> None:
        frame = pd.DataFrame({"record_id": [1, 2], "1": [0.1, 0.2]})
        with self.assertRaisesRegex(ValueError, "empty"):
            demote_known_private(frame, np.array([False, False]))
        with self.assertRaisesRegex(ValueError, "length"):
            demote_known_private(frame, np.array([True]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            demote_known_private(frame, np.array([1, 0]))

    def test_demotion_rejects_nonfinite_confidence(self) -> None:
        frame = pd.DataFrame({"record_id": [1, 2], "1": [np.nan, 0.2]})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            demote_known_private(frame, np.array([True, False]))

    def test_light_blend_is_tie_free_and_preserves_record_order(self) -> None:
        primary = pd.DataFrame(
            {"record_id": [3, 1, 2], "1": [0.75, 0.25, 0.50]}
        )
        secondary = pd.DataFrame(
            {"record_id": [3, 1, 2], "1": [0.25, 0.75, 0.50]}
        )
        result = blend_mia_frames(primary, secondary, secondary_weight=0.10)
        self.assertEqual(result["record_id"].tolist(), [3, 1, 2])
        self.assertEqual(result["1"].nunique(), 3)
        self.assertTrue(result["1"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
