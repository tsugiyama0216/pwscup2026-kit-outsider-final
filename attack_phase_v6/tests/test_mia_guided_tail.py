from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd

from attack_phase_v6.mia_guided_tail import build_guided_tail


EXPECTED_GUIDED_HASH = (
    "70E8E689E476E9EEF54F5821846EC1B93DE04FD6A63B680849D34F720DB78AFA"
)


def _ordered_subset(
    frame: pd.DataFrame, target: str, record_ids: set[int]
) -> list[int]:
    return (
        frame.sort_values(target, ascending=False, kind="mergesort")
        .loc[lambda value: value["record_id"].isin(record_ids), "record_id"]
        .astype(int)
        .tolist()
    )


class GuidedMiaTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        team_dir = repo_root / "submissions" / "prelim" / "team24"

        def load(name: str) -> pd.DataFrame:
            return pd.read_csv(team_dir / name / "F_mia.csv")

        cls.v4 = load("attack-candidate-v4")
        cls.output, cls.detail = build_guided_tail(
            load("attack-candidate-v3"),
            cls.v4,
            load("attack-candidate-daybreak-independent"),
        )

    def test_production_owner_switches_match_equal_rank_fusion(self) -> None:
        self.assertEqual(self.detail["switch_count"], 5)
        self.assertEqual(
            self.detail["switched_record_ids"], [410, 647, 807, 998, 1001]
        )
        switched = [row for row in self.detail["owners"] if row["switched"]]
        self.assertTrue(all(row["guide_gain"] >= 5 for row in switched))
        self.assertTrue(all(row["v4_penalty"] <= 3 for row in switched))
        for row in self.detail["owners"]:
            owner = row["guided_owner"]
            owner_sum = row["ranks"][owner]["v4"] + row["ranks"][owner]["guide"]
            self.assertEqual(
                owner_sum,
                min(
                    ranks["v4"] + ranks["guide"]
                    for ranks in row["ranks"].values()
                ),
            )

    def test_private_rare_and_ceiling_rankings_are_preserved(self) -> None:
        baseline = self.detail["baseline"]
        private = set(int(value) for value in baseline["private_record_ids"])
        rare = set(int(value) for value in baseline["eligible_rare_record_ids"])
        targets = [column for column in self.v4.columns if column != "record_id"]
        for target in targets:
            self.assertEqual(
                _ordered_subset(self.output, target, private),
                _ordered_subset(self.v4, target, private),
            )
            self.assertEqual(
                _ordered_subset(self.output, target, rare),
                _ordered_subset(self.v4, target, rare),
            )
        for target in ("1", "2"):
            self.assertEqual(
                _ordered_subset(
                    self.output,
                    target,
                    set(self.output["record_id"].astype(int)),
                ),
                _ordered_subset(
                    self.v4,
                    target,
                    set(self.v4["record_id"].astype(int)),
                ),
            )

    def test_production_output_is_reproducible_and_tail_is_unique(self) -> None:
        raw = self.output.to_csv(index=False).encode("utf-8")
        self.assertEqual(hashlib.sha256(raw).hexdigest().upper(), EXPECTED_GUIDED_HASH)
        self.assertEqual(self.detail["final_duplicate_slots"], 0)
        self.assertEqual(self.detail["final_unique_tail_records"], 300)

    def test_rejects_misaligned_guide_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "guide MIA columns"):
            build_guided_tail(
                self.v4,
                self.v4,
                self.v4.drop(columns=[self.v4.columns[-1]]),
            )


if __name__ == "__main__":
    unittest.main()
