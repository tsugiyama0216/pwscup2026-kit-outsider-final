from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase_v5.generate_submission_v5 import load_public_targets, sha256_file
from attack_phase_v6.generate_aggressive_final_submission_v6 import ZERO_TARGET_IDS
from attack_phase_v7.generate_submission_v7 import (
    EVENT_LINK_TARGET,
    EXPECTED_ACTIVE_ROWS,
    EXPECTED_INPUT_HASHES,
    EXPECTED_OUTPUT_HASHES,
    assert_zero_feedback,
    stack_aia_feedback,
    validate_variable_aia,
)
from pwscup2026.kit import validate as official_validate


ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "submissions" / "prelim" / "team24"
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"


class GenerateSubmissionV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.releases, cls.challenges = load_public_targets(TARGETS)
        cls.base = pd.read_csv(
            TEAM / "attack-candidate-v6-final-horizon-edge" / "F_aia.csv"
        )
        cls.v4 = pd.read_csv(TEAM / "attack-candidate-v4" / "F_aia.csv")
        cls.independent = pd.read_csv(
            TEAM / "attack-candidate-daybreak-independent" / "F_aia.csv"
        )
        cls.routed = pd.read_csv(
            TEAM / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv"
        )
        cls.output, cls.details = stack_aia_feedback(
            cls.base,
            cls.v4,
            cls.independent,
            cls.routed,
            cls.releases,
            cls.challenges,
        )

    def test_hash_locked_inputs_and_outputs_exist(self) -> None:
        paths = {
            "base_mia": TEAM
            / "attack-candidate-v6-final-horizon-edge"
            / "F_mia.csv",
            "base_aia": TEAM
            / "attack-candidate-v6-final-horizon-edge"
            / "F_aia.csv",
            "v4_aia": TEAM / "attack-candidate-v4" / "F_aia.csv",
            "independent_aia": TEAM
            / "attack-candidate-daybreak-independent"
            / "F_aia.csv",
            "routed_aia": TEAM
            / "attack-candidate-v6-routed-guided-probe"
            / "F_aia.csv",
            "feedback_v4": ROOT / "attack_phase_v6" / "official_feedback" / "v4.json",
            "feedback_independent": ROOT
            / "attack_phase_v6"
            / "official_feedback"
            / "probe_independent.json",
            "feedback_routed": ROOT
            / "attack_phase_v6"
            / "official_feedback"
            / "routed_guided_probe.json",
            "targets_manifest": TARGETS / "SHA256SUMS.txt",
        }
        self.assertEqual(
            {name: sha256_file(path) for name, path in paths.items()},
            EXPECTED_INPUT_HASHES,
        )
        candidate = TEAM / "attack-candidate-v7-feedback-stack"
        self.assertEqual(
            {
                "F_mia.csv": sha256_file(candidate / "F_mia.csv"),
                "F_aia.csv": sha256_file(candidate / "F_aia.csv"),
            },
            EXPECTED_OUTPUT_HASHES,
        )

    def test_zero_targets_are_complements_of_unanimous_direct_mask(self) -> None:
        for target_id in ZERO_TARGET_IDS:
            ids = (
                self.challenges[target_id]
                .sort_values("challenge_row_id")["challenge_row_id"]
                .to_numpy(dtype=int)
            )

            def active(frame: pd.DataFrame) -> np.ndarray:
                block = frame.loc[frame["target_k"].eq(int(target_id))].set_index(
                    "challenge_row_id"
                )
                return ~np.isclose(block.loc[ids, "time_hat"].to_numpy(), 11.0)

            expected = ~(active(self.v4) & active(self.independent) & active(self.routed))
            observed_block = self.output.loc[
                self.output["target_k"].eq(int(target_id))
            ].set_index("challenge_row_id")
            observed_values = observed_block.loc[ids, "time_hat"].to_numpy(dtype=float)
            self.assertTrue(np.array_equal(~np.isclose(observed_values, 11.0), expected))
            self.assertEqual(set(observed_values), {9.5, 11.0})
            self.assertEqual(int(expected.sum()), EXPECTED_ACTIVE_ROWS[target_id])

    def test_only_locked_targets_change(self) -> None:
        changed = []
        for target_id in sorted(self.challenges, key=int):
            base = self.base.loc[self.base["target_k"].eq(int(target_id))].reset_index(
                drop=True
            )
            output = self.output.loc[
                self.output["target_k"].eq(int(target_id))
            ].reset_index(drop=True)
            if not output.equals(base):
                changed.append(target_id)
        self.assertEqual(changed, sorted([*ZERO_TARGET_IDS, EVENT_LINK_TARGET], key=int))
        self.assertEqual(
            self.details["per_target"][EVENT_LINK_TARGET]["changed_challenge_row_ids"],
            [37, 102, 104, 114, 243, 255, 290],
        )

    def test_variable_cardinality_format_and_official_validator_pass(self) -> None:
        details = validate_variable_aia(self.output, self.challenges)
        self.assertEqual(details["rows"], 20 * 326)
        dist = official_validate.load_dist(TARGETS)
        result = official_validate.validate_aia_submission(self.output, {}, dist)
        self.assertTrue(result.ok, result.errors)

    def test_feedback_gate_rejects_a_nonzero_source(self) -> None:
        feedback_dir = ROOT / "attack_phase_v6" / "official_feedback"
        reports = {
            "v4": json.loads((feedback_dir / "v4.json").read_text(encoding="utf-8")),
            "independent": json.loads(
                (feedback_dir / "probe_independent.json").read_text(encoding="utf-8")
            ),
            "routed": json.loads(
                (feedback_dir / "routed_guided_probe.json").read_text(encoding="utf-8")
            ),
        }
        assert_zero_feedback(reports)
        reports["routed"]["aia_per_target"]["6"] = 0.01
        with self.assertRaises(ValueError):
            assert_zero_feedback(reports)


if __name__ == "__main__":
    unittest.main()
