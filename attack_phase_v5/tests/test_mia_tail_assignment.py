from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase_v5.mia_tail_assignment import (
    assign_unique_top_tail,
    infer_private_bottom_ids,
    recover_eligible_rare_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = REPO_ROOT / "submissions" / "prelim" / "team24"
V3_PATH = SUBMISSIONS / "attack-candidate-v3" / "F_mia.csv"
V4_PATH = SUBMISSIONS / "attack-candidate-v4" / "F_mia.csv"
EXPECTED_ELIGIBLE_RARE_IDS = (
    23,
    249,
    336,
    393,
    455,
    475,
    502,
    548,
    655,
    707,
    1022,
)


class MiaTailAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = pd.read_csv(V3_PATH)
        cls.v4 = pd.read_csv(V4_PATH)
        cls.targets = [column for column in cls.v4.columns if column != "record_id"]

    def test_recovers_private_and_unique_eligible_rare_sets(self) -> None:
        private_ids = infer_private_bottom_ids(self.v3)
        self.assertEqual(len(private_ids), 163)
        rare_ids = recover_eligible_rare_ids(self.v3, self.v4, private_ids)
        self.assertEqual(rare_ids, EXPECTED_ELIGIBLE_RARE_IDS)
        self.assertTrue(set(private_ids).isdisjoint(rare_ids))

    def test_assignment_preserves_invariants_for_supported_tail_widths(self) -> None:
        private_ids = set(infer_private_bottom_ids(self.v3))
        rare_ids = set(EXPECTED_ELIGIBLE_RARE_IDS)
        for top_l, expected_original_duplicates in ((12, 9), (15, 16)):
            with self.subTest(top_l=top_l):
                output, diagnostics = assign_unique_top_tail(
                    self.v3, self.v4, top_l=top_l
                )
                self.assertTrue(output["record_id"].equals(self.v4["record_id"]))
                self.assertEqual(list(output.columns), list(self.v4.columns))
                self.assertEqual(
                    diagnostics["original_duplicate_slots"],
                    expected_original_duplicates,
                )
                self.assertEqual(diagnostics["changed_tail_slots"], expected_original_duplicates)
                self.assertEqual(diagnostics["final_duplicate_slots"], 0)

                top_ids: list[int] = []
                for target in self.targets:
                    values = output[target].to_numpy(dtype=float)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values > 0.0) & (values < 1.0)).all())
                    self.assertEqual(len(np.unique(values)), len(values))

                    descending = output.sort_values(
                        target, ascending=False, kind="mergesort"
                    )["record_id"].astype(int).tolist()
                    top_ids.extend(descending[:top_l])
                    self.assertEqual(set(descending[-163:]), private_ids)

                    v3_rare_order = [
                        value
                        for value in self.v3.sort_values(
                            target, ascending=False, kind="mergesort"
                        )["record_id"].astype(int)
                        if value in rare_ids
                    ]
                    output_rare_order = [
                        value for value in descending if value in rare_ids
                    ]
                    self.assertEqual(output_rare_order, v3_rare_order)
                self.assertEqual(len(top_ids), len(set(top_ids)))

    def test_rejects_tail_width_outside_low_fpr_window(self) -> None:
        for top_l in (11, 16):
            with self.subTest(top_l=top_l):
                with self.assertRaisesRegex(ValueError, "top_l"):
                    assign_unique_top_tail(self.v3, self.v4, top_l=top_l)


if __name__ == "__main__":
    unittest.main()
