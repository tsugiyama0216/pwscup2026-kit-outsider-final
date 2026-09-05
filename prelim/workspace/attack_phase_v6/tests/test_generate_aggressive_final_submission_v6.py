from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS
from attack_phase_v6.generate_aggressive_final_submission_v6 import (
    EXPECTED_OUTPUT_HASHES,
    POSITIVE_TARGET_IDS,
    ZERO_TARGET_IDS,
    complement_zero_target_blocks,
)


class AggressiveFinalSubmissionV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        team_dir = cls.repo_root / "submissions" / "prelim" / "team24"
        cls.safe_dir = team_dir / "attack-candidate-v6-final-feedback-winners"
        cls.routed_dir = team_dir / "attack-candidate-v6-routed-guided-probe"
        cls.aggressive_dir = (
            team_dir / "attack-candidate-v6-final-aggressive-complements"
        )

    def test_builder_rejects_nonbinary_zero_block(self) -> None:
        safe = pd.DataFrame(
            {
                "target_k": [1, 1, 2, 2],
                "challenge_row_id": [0, 1, 0, 1],
                "time_hat": [10.0, 11.0, 10.0, 11.0],
            }
        )
        routed = safe.copy()
        routed.loc[2, "time_hat"] = 9.0
        with self.assertRaisesRegex(ValueError, "not a balanced 10/11"):
            complement_zero_target_blocks(
                safe,
                routed,
                target_ids=("1", "2"),
                zero_target_ids=("2",),
                rows_per_target=2,
            )

    def test_production_changes_exactly_the_seven_zero_blocks(self) -> None:
        safe = pd.read_csv(self.safe_dir / "F_aia.csv")
        routed = pd.read_csv(self.routed_dir / "F_aia.csv")
        aggressive = pd.read_csv(self.aggressive_dir / "F_aia.csv")
        changed_targets: list[str] = []
        for target_id in EXPECTED_TARGET_IDS:
            target = int(target_id)
            safe_block = safe.loc[safe["target_k"].eq(target)].reset_index(drop=True)
            routed_block = routed.loc[routed["target_k"].eq(target)].reset_index(
                drop=True
            )
            aggressive_block = aggressive.loc[
                aggressive["target_k"].eq(target)
            ].reset_index(drop=True)
            if not aggressive_block.equals(safe_block):
                changed_targets.append(target_id)
            if target_id in ZERO_TARGET_IDS:
                pd.testing.assert_frame_equal(
                    aggressive_block[["target_k", "challenge_row_id"]],
                    routed_block[["target_k", "challenge_row_id"]],
                )
                np.testing.assert_array_equal(
                    aggressive_block["time_hat"].to_numpy(dtype=float),
                    21.0 - routed_block["time_hat"].to_numpy(dtype=float),
                )
                self.assertEqual(
                    aggressive_block["time_hat"].value_counts().to_dict(),
                    {10.0: 163, 11.0: 163},
                )
            else:
                pd.testing.assert_frame_equal(aggressive_block, safe_block)
        self.assertEqual(tuple(changed_targets), ZERO_TARGET_IDS)
        self.assertEqual(
            set(POSITIVE_TARGET_IDS), set(EXPECTED_TARGET_IDS) - set(ZERO_TARGET_IDS)
        )

    def test_production_candidate_has_only_two_locked_files(self) -> None:
        self.assertEqual(
            sorted(path.name for path in self.aggressive_dir.iterdir()),
            ["F_aia.csv", "F_mia.csv"],
        )
        observed = {
            name: hashlib.sha256((self.aggressive_dir / name).read_bytes())
            .hexdigest()
            .upper()
            for name in ("F_mia.csv", "F_aia.csv")
        }
        self.assertEqual(observed, EXPECTED_OUTPUT_HASHES)
        self.assertEqual(
            (self.aggressive_dir / "F_mia.csv").read_bytes(),
            (self.safe_dir / "F_mia.csv").read_bytes(),
        )

    def test_positive_target_csv_rows_are_byte_identical_to_safe(self) -> None:
        safe_lines = (self.safe_dir / "F_aia.csv").read_bytes().splitlines(
            keepends=True
        )
        aggressive_lines = (
            self.aggressive_dir / "F_aia.csv"
        ).read_bytes().splitlines(keepends=True)

        def target_lines(lines: list[bytes], target_id: str) -> list[bytes]:
            target = int(target_id)
            return [
                line
                for line in lines[1:]
                if int(line.split(b",", maxsplit=1)[0]) == target
            ]

        for target_id in POSITIVE_TARGET_IDS:
            with self.subTest(target_id=target_id):
                self.assertEqual(
                    target_lines(aggressive_lines, target_id),
                    target_lines(safe_lines, target_id),
                )


if __name__ == "__main__":
    unittest.main()
