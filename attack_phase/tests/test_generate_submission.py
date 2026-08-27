from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase.generate_submission import (
    AIA_VIEW_SPECS,
    MIA_REQUIRED_COLUMNS,
    aia_pair_cost,
    cardinality_link_assignment,
    choose_low_hit_time,
    constrained_capacity_assignment,
    fit_view,
    independent_link_assignment,
    linkage_component,
    sinkhorn_transport,
    strict_rank,
)


class CapacityAssignmentTests(unittest.TestCase):
    def test_exact_capacity_and_candidate_exclusivity(self) -> None:
        rng = np.random.default_rng(123)
        utility = rng.normal(size=(12, 3))
        utility[0, :] = 10.0  # one common outlier must still choose at most one target
        assigned, counts = constrained_capacity_assignment(
            utility, minimum=2, maximum=3, total=7
        )
        self.assertEqual(int((assigned >= 0).sum()), 7)
        self.assertTrue(np.all((counts >= 2) & (counts <= 3)))
        self.assertEqual(int(counts.sum()), 7)
        self.assertIn(int(assigned[0]), (-1, 0, 1, 2))

    def test_infeasible_capacity_is_rejected(self) -> None:
        utility = np.zeros((8, 3), dtype=float)
        with self.assertRaises(ValueError):
            constrained_capacity_assignment(utility, minimum=3, maximum=4, total=8)

    def test_transport_marginals_and_tie_free_ranks(self) -> None:
        rng = np.random.default_rng(456)
        utility = rng.normal(size=(12, 3))
        _, counts = constrained_capacity_assignment(utility, minimum=2, maximum=3, total=7)
        plan, diagnostics = sinkhorn_transport(
            utility, counts, null_count=5, epsilon=0.5
        )
        self.assertLess(diagnostics["max_row_error"], 1e-7)
        self.assertLess(diagnostics["max_column_error"], 1e-7)
        self.assertTrue(np.allclose(plan.sum(axis=0), counts, atol=1e-7))
        for target in range(plan.shape[1]):
            confidence = strict_rank(plan[:, target], np.arange(len(plan)))
            self.assertEqual(len(np.unique(confidence)), len(confidence))
            self.assertTrue(np.all((confidence > 0.0) & (confidence < 1.0)))

    def test_obvious_membership_structure_is_recovered(self) -> None:
        utility = np.full((9, 3), -5.0)
        utility[0:2, 0] = 8.0
        utility[2:4, 1] = 8.0
        utility[4:6, 2] = 8.0
        assigned, counts = constrained_capacity_assignment(
            utility, minimum=2, maximum=2, total=6
        )
        self.assertTrue(np.array_equal(assigned[:6], np.repeat(np.arange(3), 2)))
        self.assertTrue(np.all(assigned[6:] == -1))
        plan, _ = sinkhorn_transport(utility, counts, null_count=3, epsilon=0.25)
        for target in range(3):
            expected = {2 * target, 2 * target + 1}
            observed = set(np.argsort(plan[:, target])[-2:].tolist())
            self.assertEqual(observed, expected)


class AIALinkageTests(unittest.TestCase):
    @staticmethod
    def _population(rng: np.random.Generator, n: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "age": rng.integers(40, 75, size=n),
                "BMI": rng.normal(23, 3, size=n),
                "SBP": rng.normal(125, 15, size=n),
                "TG": rng.lognormal(4.5, 0.45, size=n),
                "HDL": rng.normal(50, 9, size=n),
                "ALT": rng.lognormal(2.8, 0.5, size=n),
                "FPG": rng.normal(95, 15, size=n),
                "sex": rng.integers(0, 2, size=n),
                "prefecture": rng.integers(1, 8, size=n),
                "smoking": rng.integers(0, 2, size=n),
                "time": rng.uniform(0.1, 10.0, size=n),
                "onset": rng.integers(0, 2, size=n),
                "death": np.zeros(n, dtype=int),
            }
        )
        return frame.loc[:, MIA_REQUIRED_COLUMNS]

    def test_density_normalized_linkage_separates_positive_control(self) -> None:
        rng = np.random.default_rng(789)
        background = self._population(rng, 800)
        source = self._population(rng, 60)
        released = source.iloc[:40].copy().reset_index(drop=True)
        for column in ("SBP", "TG", "HDL", "ALT"):
            released[column] += rng.normal(0.0, 0.01, size=len(released))
        members = source.iloc[:20].copy().reset_index(drop=True)
        controls = self._population(rng, 20)
        query = pd.concat([members, controls], ignore_index=True)
        fitted = fit_view(background, AIA_VIEW_SPECS[1])
        component = linkage_component(released, query, background, fitted)
        top = np.argsort(component["strength"])[-20:]
        self.assertGreaterEqual(int(np.sum(top < 20)), 17)

    def test_cardinality_assignment_uses_unique_release_rows(self) -> None:
        # All real edges are better than a zero-cost dummy.  The construction must
        # still select exactly three rather than greedily linking every query.
        cost = np.full((6, 8), -1.0)
        cost[0, 0] = -5.0
        cost[1, 1] = -4.0
        cost[2, 2] = -3.0
        assigned = cardinality_link_assignment(cost, active_count=3)
        selected = assigned[assigned >= 0]
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(np.unique(selected)), 3)
        self.assertTrue(np.array_equal(assigned[:3], np.array([0, 1, 2])))

    def test_independent_variant_keeps_exact_cardinality_but_can_reuse(self) -> None:
        cost = np.full((5, 3), 2.0)
        cost[:, 0] = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        assigned = independent_link_assignment(cost, active_count=2, tie_key=np.arange(5))
        self.assertEqual(int((assigned >= 0).sum()), 2)
        self.assertTrue(np.array_equal(assigned[:2], np.array([0, 0])))
        self.assertTrue(np.all(assigned[2:] == -1))

    def test_low_hit_time_stays_inside_horizon(self) -> None:
        background = pd.DataFrame({"time": [1.0, 1.2, 5.0, 5.1, 10.0]})
        value = choose_low_hit_time(background, horizon=10.0, tau=0.5)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 10.0)

    def test_end_to_end_positive_control_and_row_order_invariance(self) -> None:
        rng = np.random.default_rng(2468)
        background = self._population(rng, 1_000)
        released = self._population(rng, 80)
        members = released.iloc[:20].copy().reset_index(drop=True)
        controls = self._population(rng, 20)
        query_source = pd.concat([members, controls], ignore_index=True)
        challenge = query_source[["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]].copy()
        challenge.insert(0, "challenge_row_id", np.arange(len(challenge)))
        fitted = [fit_view(background, spec) for spec in AIA_VIEW_SPECS]

        cost, _, _ = aia_pair_cost(released, challenge, background, fitted)
        assigned = cardinality_link_assignment(cost, active_count=20)
        self.assertTrue(np.all(assigned[:20] >= 0))
        self.assertTrue(np.all(assigned[20:] == -1))
        predicted = released["time"].to_numpy()[assigned[:20]]
        self.assertTrue(np.allclose(predicted, members["time"].to_numpy()))

        shuffled = released.sample(frac=1.0, random_state=99).reset_index(drop=True)
        shuffled_cost, _, _ = aia_pair_cost(shuffled, challenge, background, fitted)
        shuffled_assigned = cardinality_link_assignment(shuffled_cost, active_count=20)
        shuffled_predicted = shuffled["time"].to_numpy()[shuffled_assigned[:20]]
        self.assertTrue(np.allclose(shuffled_predicted, predicted))


if __name__ == "__main__":
    unittest.main()
