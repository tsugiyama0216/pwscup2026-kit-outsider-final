import unittest

import numpy as np
import pandas as pd

from attack_phase_v9.surrogate_mia import (
    assign_surrogate_donors,
    donor_counts,
    exact_challenge_membership,
    map_release_to_donor_pools,
    occupancy_scores,
    rerank_top_tail,
    stratified_discovery_mask,
    tpr_auc,
)


def frame(points):
    values = np.asarray(points, dtype=float)
    return pd.DataFrame({"age": values[:, 0], "SBP": values[:, 1], "TG": values[:, 2]})


class DonorMappingTests(unittest.TestCase):
    def test_tolerant_mixed_pool_mapping_and_multiplicity(self):
        actual = frame([[10, 100, 1], [20, 110, 2], [30, 120, 3]])
        toy = frame([[40, 130, 4], [50, 140, 5]])
        release = frame([
            [10 + 1e-12, 100, 1],
            [40, 130 - 1e-12, 4],
            [40, 130, 4],
            [99, 999, 9],
        ])
        mapping = map_release_to_donor_pools(release, {"toy": toy, "actual": actual})
        self.assertEqual(mapping.pool.tolist(), ["actual", "toy", "toy", "unmatched"])
        self.assertEqual(mapping.donor_index.tolist(), [0, 0, 0, -1])
        self.assertEqual(mapping.matched.tolist(), [True, True, True, False])
        np.testing.assert_array_equal(donor_counts(mapping, "toy", len(toy)), [2, 0])

    def test_cross_pool_duplicate_is_flagged_ambiguous(self):
        shared = frame([[10, 100, 1]])
        mapping = map_release_to_donor_pools(shared, {"a": shared, "b": shared})
        self.assertTrue(mapping.ambiguous[0])

    def test_replacement_modes(self):
        source = frame([[0, 0, 0], [0.01, 0, 0], [9.9, 0, 0]])
        donors = frame([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
        repeated = assign_surrogate_donors(source, donors, without_replacement=False)
        unique = assign_surrogate_donors(source, donors, without_replacement=True)
        self.assertLess(len(np.unique(repeated)), len(repeated))
        self.assertEqual(len(np.unique(unique)), len(unique))


class OccupancyScoringTests(unittest.TestCase):
    def test_selected_donor_cell_scores_above_unselected(self):
        donors = frame([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
        queries = frame([[0.01, 0, 0], [9.99, 0, 0]])
        scores = occupancy_scores(queries, donors, np.array([2, 0, 0]), neighbours=1)
        self.assertGreater(scores["occupancy"][0], scores["occupancy"][1])
        self.assertGreater(scores["nearest_selected"][0], scores["nearest_selected"][1])

    def test_inverse_cell_requires_reference(self):
        donors = frame([[0, 0, 0], [5, 0, 0]])
        with self.assertRaises(ValueError):
            occupancy_scores(donors, donors, np.ones(2), neighbours=1, inverse_cell=True)

    def test_tpr_auc_ties_match_official_interpolation(self):
        tpr, auc = tpr_auc(np.ones(4), np.array([1, 1, 0, 0]), low_fpr=0.01)
        self.assertAlmostEqual(tpr, 0.01)
        self.assertAlmostEqual(auc, 0.5)


class MembershipAndCandidateTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_balanced_and_deterministic(self):
        truth = np.array([False] * 5 + [True] * 7)
        keys = np.arange(len(truth)) + 100
        first = stratified_discovery_mask(keys, truth)
        second = stratified_discovery_mask(keys, truth)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int((first & ~truth).sum()), 2)
        self.assertEqual(int((first & truth).sum()), 3)
        self.assertFalse(np.any(first & ~first))

    def test_exact_membership_never_uses_row_id(self):
        columns = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
        candidates = pd.DataFrame([
            [999, 10, 100, 1, 2, 3, 0, "A"],
            [998, 20, 110, 2, 3, 4, 1, "B"],
        ], columns=["record_id", *columns])
        challenge = pd.DataFrame([
            [998, 10, 100, 1, 2, 3, 0, "A"],
            [999, 30, 120, 3, 4, 5, 1, "C"],
        ], columns=["challenge_row_id", *columns])
        np.testing.assert_array_equal(exact_challenge_membership(challenge, candidates), [True, False])

    def test_tail_rerank_preserves_locks_ceiling_targets_and_multisets(self):
        baseline = pd.DataFrame({
            "record_id": np.arange(8),
            "1": np.linspace(0.1, 0.8, 8),
            "2": np.linspace(0.8, 0.1, 8),
            "18": np.linspace(0.1, 0.8, 8),
            "19": np.linspace(0.15, 0.85, 8),
            "29": np.linspace(0.2, 0.9, 8),
        })
        locked = np.array([True, False, False, False, False, False, False, True])
        signals = {target: -np.arange(8, dtype=float) for target in ("18", "19", "29")}
        result = rerank_top_tail(baseline, signals, locked, width=4, weight=2.0)
        pd.testing.assert_series_equal(result.record_id, baseline.record_id)
        pd.testing.assert_frame_equal(result.loc[locked], baseline.loc[locked])
        pd.testing.assert_frame_equal(result[["1", "2"]], baseline[["1", "2"]])
        for target in ("18", "19", "29"):
            np.testing.assert_array_equal(np.sort(result[target]), np.sort(baseline[target]))
        self.assertGreater(sum((result[target] != baseline[target]).sum() for target in signals), 0)


if __name__ == "__main__":
    unittest.main()
