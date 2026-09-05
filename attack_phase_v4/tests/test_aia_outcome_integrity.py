from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v4.aia_outcome_integrity import (
    fixed_selection_horizon,
    route_target_blocks,
)


class _FingerprintStub:
    def __init__(self, auc_by_marker: dict[int, float | None]) -> None:
        self.auc_by_marker = auc_by_marker

    def measure(self, released: pd.DataFrame) -> dict[str, float | None]:
        auc = self.auc_by_marker[int(released.loc[0, "marker"])]
        return {"auc": auc, "correlation": 0.0, "event_rate": 0.5}


class AiaOutcomeIntegrityTests(unittest.TestCase):
    def test_fixed_selection_horizon_preserves_active_rows(self) -> None:
        adaptive = pd.DataFrame(
            {
                "target_k": [1, 1, 1],
                "challenge_row_id": [10, 11, 12],
                "time_hat": [2.0, 10.0, 11.0],
            }
        )
        output = fixed_selection_horizon(
            adaptive, abstain_time=11.0, horizon_time=10.0
        )
        np.testing.assert_allclose(output["time_hat"], [10.0, 10.0, 11.0])

    def test_router_uses_horizon_for_low_or_degenerate_auc(self) -> None:
        adaptive = pd.DataFrame(
            {
                "target_k": [1, 1, 2, 2, 3, 3],
                "challenge_row_id": [10, 11, 20, 21, 30, 31],
                "time_hat": [2.0, 11.0, 3.0, 11.0, 4.0, 11.0],
            }
        )
        horizon = fixed_selection_horizon(
            adaptive, abstain_time=11.0, horizon_time=10.0
        )
        releases = {
            "1": pd.DataFrame({"marker": [1]}),
            "2": pd.DataFrame({"marker": [2]}),
            "3": pd.DataFrame({"marker": [3]}),
        }
        output, diagnostics = route_target_blocks(
            adaptive,
            horizon,
            releases,
            _FingerprintStub({1: 0.70, 2: 0.57, 3: None}),
            auc_threshold=0.58,
        )
        np.testing.assert_allclose(
            output["time_hat"], [2.0, 11.0, 10.0, 11.0, 10.0, 11.0]
        )
        self.assertEqual(diagnostics["routed_targets"], ["2", "3"])
        self.assertEqual(diagnostics["targets"]["1"]["decoder"], "adaptive")

    def test_router_rejects_different_key_order(self) -> None:
        adaptive = pd.DataFrame(
            {
                "target_k": [1, 1],
                "challenge_row_id": [1, 2],
                "time_hat": [1.0, 11.0],
            }
        )
        horizon = adaptive.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            route_target_blocks(
                adaptive,
                horizon,
                {"1": pd.DataFrame({"marker": [1]})},
                _FingerprintStub({1: 0.50}),
                auc_threshold=0.58,
            )


if __name__ == "__main__":
    unittest.main()
