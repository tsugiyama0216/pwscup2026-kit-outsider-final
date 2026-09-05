from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from attack_phase_v6.generate_final_submission_v6 import (
    COMBINED_MIA_ROUNDING_WIDTH,
    EXPECTED_FINAL_AIA_POWER,
    EXPECTED_OUTPUT_HASHES,
    SOURCE_ORDER,
    assert_expected_output_hashes,
    choose_final_mia,
    load_complete_feedback,
    verify_named_hashes,
)


class FinalSubmissionV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.team_dir = cls.repo_root / "submissions" / "prelim" / "team24"
        cls.feedback_dir = cls.repo_root / "attack_phase_v6" / "official_feedback"

    def test_input_hash_gate_accepts_exact_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes(b"stable\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            self.assertEqual(
                verify_named_hashes({"sample": path}, expected={"sample": digest}),
                {"sample": digest},
            )
            path.write_bytes(b"drifted\n")
            with self.assertRaisesRegex(ValueError, "final input hash mismatch"):
                verify_named_hashes({"sample": path}, expected={"sample": digest})

    def test_complete_feedback_rejects_a_missing_target(self) -> None:
        source = json.loads(
            (self.feedback_dir / "routed_guided_probe.json").read_text(
                encoding="utf-8"
            )
        )
        incomplete = copy.deepcopy(source)
        del incomplete["aia_per_target"]["29"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.json"
            path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target set mismatch"):
                load_complete_feedback(
                    path,
                    expected_candidate="routed_guided_probe",
                    require_score_1=True,
                    require_all_displayed_scores=True,
                )

    def test_mia_gate_reverts_to_v4_within_combined_rounding(self) -> None:
        baseline = load_complete_feedback(
            self.feedback_dir / "probe_independent.json",
            expected_candidate="probe_independent",
            require_score_1=True,
        )
        guided = load_complete_feedback(
            self.feedback_dir / "routed_guided_probe.json",
            expected_candidate="routed_guided_probe",
            require_score_1=True,
            require_all_displayed_scores=True,
        )

        decision = choose_final_mia(baseline, guided)

        self.assertEqual(decision["selected"], "v4")
        self.assertTrue(decision["equivalent_within_rounding"])
        self.assertAlmostEqual(
            decision["guided_minus_v4"], 0.000007360034977255, places=15
        )
        self.assertEqual(
            decision["combined_rounding_width"], COMBINED_MIA_ROUNDING_WIDTH
        )

    def test_output_hash_gate_rejects_recipe_drift(self) -> None:
        assert_expected_output_hashes(EXPECTED_OUTPUT_HASHES)
        changed = dict(EXPECTED_OUTPUT_HASHES)
        changed["F_aia.csv"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "before write"):
            assert_expected_output_hashes(changed)

    def test_production_candidate_uses_whole_measured_winner_blocks(self) -> None:
        candidate = self.team_dir / "attack-candidate-v6-final-feedback-winners"
        self.assertEqual(
            sorted(path.name for path in candidate.iterdir()),
            ["F_aia.csv", "F_mia.csv"],
        )
        self.assertEqual(
            (candidate / "F_mia.csv").read_bytes(),
            (self.team_dir / "attack-candidate-v4" / "F_mia.csv").read_bytes(),
        )
        observed_hashes = {
            name: hashlib.sha256((candidate / name).read_bytes()).hexdigest().upper()
            for name in ("F_mia.csv", "F_aia.csv")
        }
        self.assertEqual(observed_hashes, EXPECTED_OUTPUT_HASHES)

        expected_route = {
            "v4": {"1", "2", "4", "5", "6", "7", "8", "9", "12", "17", "18", "19", "29"},
            "independent": {"3", "13", "22", "23", "26"},
            "routed": {"10", "20"},
        }
        source_paths = {
            "v4": self.team_dir / "attack-candidate-v4" / "F_aia.csv",
            "independent": self.team_dir
            / "attack-candidate-daybreak-independent"
            / "F_aia.csv",
            "routed": self.team_dir
            / "attack-candidate-v6-routed-guided-probe"
            / "F_aia.csv",
        }
        sources = {label: pd.read_csv(path) for label, path in source_paths.items()}
        final = pd.read_csv(candidate / "F_aia.csv")
        for label in SOURCE_ORDER:
            for target_id in expected_route[label]:
                actual = final.loc[final["target_k"].eq(int(target_id))].reset_index(
                    drop=True
                )
                expected = sources[label].loc[
                    sources[label]["target_k"].eq(int(target_id))
                ].reset_index(drop=True)
                pd.testing.assert_frame_equal(actual, expected)

    def test_locked_final_aia_power(self) -> None:
        self.assertAlmostEqual(EXPECTED_FINAL_AIA_POWER, 0.1506204125415067)


if __name__ == "__main__":
    unittest.main()
