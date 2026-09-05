from __future__ import annotations

import json
import unittest
from pathlib import Path

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS


class OfficialFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        feedback_dir = cls.repo_root / "attack_phase_v6" / "official_feedback"
        cls.feedback = {
            name: json.loads((feedback_dir / f"{name}.json").read_text(encoding="utf-8"))
            for name in (
                "v3",
                "v4",
                "probe_independent",
                "daybreak",
                "routed_guided_probe",
            )
        }

    def test_every_feedback_file_has_all_target_keys(self) -> None:
        expected = set(EXPECTED_TARGET_IDS)
        for name, payload in self.feedback.items():
            with self.subTest(name=name):
                self.assertEqual(set(payload["mia_per_target"]), expected)
                self.assertEqual(set(payload["aia_per_target"]), expected)

    def test_duplicate_aia_runs_reconcile_exactly(self) -> None:
        self.assertEqual(
            self.feedback["daybreak"]["aia"],
            self.feedback["probe_independent"]["aia"],
        )
        self.assertEqual(
            self.feedback["daybreak"]["aia_per_target"],
            self.feedback["probe_independent"]["aia_per_target"],
        )
        team_dir = self.repo_root / "submissions" / "prelim" / "team24"
        self.assertEqual(
            (team_dir / "attack-candidate-daybreak-independent" / "F_aia.csv").read_bytes(),
            (team_dir / "attack-candidate-v5-probe-independent" / "F_aia.csv").read_bytes(),
        )

    def test_daybreak_mia_has_the_observed_tradeoff(self) -> None:
        daybreak = self.feedback["daybreak"]["mia"]
        v4 = self.feedback["v4"]["mia"]
        self.assertGreater(daybreak["tpr_at_low_fpr"], v4["tpr_at_low_fpr"])
        self.assertLess(daybreak["tpr_rare"], v4["tpr_rare"])
        self.assertLess(daybreak["auc"], v4["auc"])

    def test_v3_and_v4_aia_per_target_are_identical(self) -> None:
        self.assertEqual(
            self.feedback["v3"]["aia_per_target"],
            self.feedback["v4"]["aia_per_target"],
        )

    def test_routed_guided_feedback_preserves_all_user_supplied_values(self) -> None:
        routed = self.feedback["routed_guided_probe"]
        self.assertEqual(
            [routed[f"score_{number}_displayed"] for number in range(1, 6)],
            [0.3004, 0.1024, 0.3333, 0.1256, 0.1],
        )
        self.assertEqual(routed["mia"]["tpr_at_low_fpr"], 0.10236752136752136)
        self.assertEqual(routed["mia"]["tpr_rare"], 0.3333333333333333)
        self.assertEqual(routed["mia"]["auc"], 0.6493444718960911)
        self.assertEqual(routed["aia"]["r_time_all"], 0.12555367549200885)
        self.assertEqual(routed["aia"]["r_time_rare"], 0.1)

    def test_routed_guided_derived_mia_reconciles(self) -> None:
        routed = self.feedback["routed_guided_probe"]
        aia_power = sum(routed["aia_per_target"].values()) / len(
            routed["aia_per_target"]
        )
        mia_power = routed["score_1_displayed"] - aia_power
        self.assertAlmostEqual(aia_power, 0.14248115204199743)
        self.assertAlmostEqual(mia_power, 0.15791884795800257)
        self.assertAlmostEqual(
            routed["derived_mia_power"]["value"], mia_power, places=15
        )


if __name__ == "__main__":
    unittest.main()
