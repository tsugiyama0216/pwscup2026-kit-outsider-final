from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd

from attack_phase_v6.generate_submission_v6 import (
    COPULA_HORIZON_VARIANT,
    DEFAULT_PROBE_ROUTE,
    EQ_COP_IND_EVENT_VARIANT,
    EQ_COP_V4_EVENT_VARIANT,
    EQ_MIX_IND_EVENT_VARIANT,
    EXPECTED_ROUTED_AIA_HASH,
    GLOBAL_DENSITY_ALL5_VARIANT,
    GLOBAL_DENSITY_STABLE3_VARIANT,
    INVERTED_DENSITY_STABLE3_VARIANT,
    SEX_CONSENSUS_HORIZON_VARIANT,
    SOFT_COP_IND_EVENT_VARIANT,
    V4_MIX_V4_EVENT_VARIANT,
    feedback_winner_power,
    route_target_blocks,
)


def _source(times_by_target: dict[str, list[float]]) -> pd.DataFrame:
    parts = []
    for target_id, values in times_by_target.items():
        parts.append(
            pd.DataFrame(
                {
                    "target_k": int(target_id),
                    "challenge_row_id": list(range(len(values))),
                    "time_hat": values,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


class GenerateSubmissionV6Tests(unittest.TestCase):
    def test_default_route_protects_ceiling_and_routes_independent_winners(self) -> None:
        self.assertEqual(DEFAULT_PROBE_ROUTE["1"], "v4_exact")
        self.assertEqual(DEFAULT_PROBE_ROUTE["2"], "v4_exact")
        self.assertEqual(DEFAULT_PROBE_ROUTE["3"], EQ_MIX_IND_EVENT_VARIANT)
        for target_id in ("4", "5", "12"):
            self.assertEqual(DEFAULT_PROBE_ROUTE[target_id], EQ_COP_V4_EVENT_VARIANT)
        for target_id in ("13", "22", "23"):
            self.assertEqual(DEFAULT_PROBE_ROUTE[target_id], EQ_COP_IND_EVENT_VARIANT)
        self.assertEqual(DEFAULT_PROBE_ROUTE["26"], SOFT_COP_IND_EVENT_VARIANT)
        self.assertEqual(DEFAULT_PROBE_ROUTE["10"], V4_MIX_V4_EVENT_VARIANT)
        self.assertEqual(DEFAULT_PROBE_ROUTE["29"], "v4_exact")
        self.assertEqual(DEFAULT_PROBE_ROUTE["9"], GLOBAL_DENSITY_STABLE3_VARIANT)
        for target_id in ("7", "20"):
            self.assertEqual(
                DEFAULT_PROBE_ROUTE[target_id], GLOBAL_DENSITY_ALL5_VARIANT
            )
        for target_id in ("6", "8", "17"):
            self.assertEqual(DEFAULT_PROBE_ROUTE[target_id], COPULA_HORIZON_VARIANT)
        self.assertEqual(DEFAULT_PROBE_ROUTE["18"], SEX_CONSENSUS_HORIZON_VARIANT)
        self.assertEqual(
            DEFAULT_PROBE_ROUTE["19"], INVERTED_DENSITY_STABLE3_VARIANT
        )

    def test_route_copies_complete_target_blocks(self) -> None:
        left = _source({"1": [1.0, 11.0], "2": [2.0, 11.0]})
        right = _source({"1": [3.0, 11.0], "2": [4.0, 11.0]})

        output, detail = route_target_blocks(
            {"left": left, "right": right},
            {"1": "right", "2": "left"},
            target_ids=("1", "2"),
            rows_per_target=2,
        )

        self.assertEqual(output["time_hat"].tolist(), [3.0, 11.0, 2.0, 11.0])
        self.assertEqual(detail["source_counts"], {"left": 1, "right": 1})

    def test_route_rejects_incomplete_target_map(self) -> None:
        source = _source({"1": [1.0, 11.0], "2": [2.0, 11.0]})
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            route_target_blocks(
                {"source": source},
                {"1": "source"},
                target_ids=("1", "2"),
                rows_per_target=2,
            )

    def test_feedback_winner_power_is_mean_of_target_maxima(self) -> None:
        score_maps = {
            "left": {target_id: 0.1 for target_id in DEFAULT_PROBE_ROUTE},
            "right": {target_id: 0.2 for target_id in DEFAULT_PROBE_ROUTE},
        }
        score_maps["left"]["1"] = 1.0

        actual = feedback_winner_power(score_maps)

        self.assertAlmostEqual(actual, (1.0 + 19 * 0.2) / 20)

    def test_production_probe_preserves_v4_mia_and_ceiling_blocks(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        v4_dir = (
            repo_root / "submissions" / "prelim" / "team24" / "attack-candidate-v4"
        )
        probe_dir = (
            repo_root
            / "submissions"
            / "prelim"
            / "team24"
            / "attack-candidate-v6-routed-probe"
        )
        guided_dir = (
            repo_root
            / "submissions"
            / "prelim"
            / "team24"
            / "attack-candidate-v6-routed-guided-probe"
        )
        self.assertEqual(
            (probe_dir / "F_mia.csv").read_bytes(),
            (v4_dir / "F_mia.csv").read_bytes(),
        )
        self.assertEqual(
            (guided_dir / "F_aia.csv").read_bytes(),
            (probe_dir / "F_aia.csv").read_bytes(),
        )
        self.assertNotEqual(
            (guided_dir / "F_mia.csv").read_bytes(),
            (probe_dir / "F_mia.csv").read_bytes(),
        )

        v4 = pd.read_csv(v4_dir / "F_aia.csv").sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        )
        probe = pd.read_csv(probe_dir / "F_aia.csv").sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        )
        for target_id in (1, 2, 29):
            pd.testing.assert_frame_equal(
                probe.loc[probe["target_k"].eq(target_id)].reset_index(drop=True),
                v4.loc[v4["target_k"].eq(target_id)].reset_index(drop=True),
            )
        self.assertEqual(
            hashlib.sha256((probe_dir / "F_aia.csv").read_bytes())
            .hexdigest()
            .upper(),
            EXPECTED_ROUTED_AIA_HASH,
        )


if __name__ == "__main__":
    unittest.main()
