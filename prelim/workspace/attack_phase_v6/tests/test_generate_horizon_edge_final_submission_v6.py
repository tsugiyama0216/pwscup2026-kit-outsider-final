from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS
from attack_phase_v6.generate_horizon_edge_final_submission_v6 import (
    EDGE_TIME,
    EXPECTED_OUTPUT_HASHES,
    SHIFT_TARGET_IDS,
    SOURCE_TIME,
    UNCHANGED_TARGET_IDS,
    shift_exact_horizon_predictions,
)


class HorizonEdgeFinalSubmissionV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        team_dir = cls.repo_root / "submissions" / "prelim" / "team24"
        cls.source_dir = (
            team_dir / "attack-candidate-v6-final-aggressive-complements"
        )
        cls.edge_dir = team_dir / "attack-candidate-v6-final-horizon-edge"

    def test_builder_changes_only_exact_ten_on_selected_targets(self) -> None:
        source = pd.DataFrame(
            {
                "target_k": [1, 1, 2, 2],
                "challenge_row_id": [0, 1, 0, 1],
                "time_hat": [SOURCE_TIME, 11.0, SOURCE_TIME, 8.0],
            }
        )
        shifted, diagnostics = shift_exact_horizon_predictions(
            source,
            target_ids=("1", "2"),
            shift_target_ids=("2",),
            rows_per_target=2,
        )
        self.assertEqual(shifted["time_hat"].tolist(), [10.0, 11.0, EDGE_TIME, 8.0])
        self.assertEqual(diagnostics["total_shifted_rows"], 1)

    def test_production_candidate_preserves_selection_and_non_horizon_values(self) -> None:
        source = pd.read_csv(self.source_dir / "F_aia.csv")
        edge = pd.read_csv(self.edge_dir / "F_aia.csv")
        pd.testing.assert_frame_equal(
            source[["target_k", "challenge_row_id"]],
            edge[["target_k", "challenge_row_id"]],
        )
        for target_id in EXPECTED_TARGET_IDS:
            mask = source["target_k"].eq(int(target_id))
            source_values = source.loc[mask, "time_hat"].reset_index(drop=True)
            edge_values = edge.loc[mask, "time_hat"].reset_index(drop=True)
            if target_id in UNCHANGED_TARGET_IDS:
                pd.testing.assert_series_equal(source_values, edge_values)
                continue
            expected = source_values.mask(source_values.eq(SOURCE_TIME), EDGE_TIME)
            pd.testing.assert_series_equal(expected, edge_values)
        self.assertEqual(
            set(SHIFT_TARGET_IDS), set(EXPECTED_TARGET_IDS) - set(UNCHANGED_TARGET_IDS)
        )

    def test_production_candidate_has_only_two_locked_files(self) -> None:
        self.assertEqual(
            sorted(path.name for path in self.edge_dir.iterdir()),
            ["F_aia.csv", "F_mia.csv"],
        )
        observed = {
            name: hashlib.sha256((self.edge_dir / name).read_bytes())
            .hexdigest()
            .upper()
            for name in ("F_mia.csv", "F_aia.csv")
        }
        self.assertEqual(observed, EXPECTED_OUTPUT_HASHES)
        self.assertEqual(
            (self.edge_dir / "F_mia.csv").read_bytes(),
            (self.source_dir / "F_mia.csv").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
