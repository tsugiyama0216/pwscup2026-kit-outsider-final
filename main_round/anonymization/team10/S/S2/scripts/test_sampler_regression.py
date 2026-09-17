"""Public-only regression tests for the S1 conditional-round sampler defect.

The model below encodes exactly independent variables. It uses no contest data,
does not invoke a DP measurement, and does not spend a privacy budget. The round
control deliberately bypasses S2's guard to demonstrate the defect in the pinned
mbi 1.1.0 dependency; the sample path calls S2's actual synthesis entry point.
"""

from __future__ import annotations

from itertools import combinations
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
from mbi import CliqueVector, Domain, MarkovRandomField, marginal_oracles
from opendp.extras.mbi import ContingencyTable

from generate_pwscup import (
    ConfigurationError,
    synthesize_table,
    validate_synthesis,
)


ROWS = 1049
SEEDS = (0, 1, 2)
CHILDREN = ("A", "B", "C")


def sample_config() -> dict:
    return {"output_rows": ROWS, "synthesis": {"method": "sample"}}


def independent_star_table() -> ContingencyTable:
    """Zero potentials imply a uniform joint, despite the star factorization."""
    sizes = {"H": 2, "A": 8, "B": 8, "C": 8}
    domain = Domain.fromdict(sizes)
    potentials = CliqueVector.zeros(
        domain, [("H", "A"), ("H", "B"), ("H", "C")]
    )
    marginals = marginal_oracles.message_passing_stable(potentials)
    model = MarkovRandomField(potentials=potentials, marginals=marginals, total=1)
    return ContingencyTable(
        keys={name: np.arange(size) for name, size in sizes.items()},
        cuts={},
        marginals={},
        model=model,
    )


class SynthesisConfigurationTests(unittest.TestCase):
    def test_sample_is_accepted(self):
        validate_synthesis(sample_config())

    def test_round_missing_and_misspelled_methods_are_rejected(self):
        invalid = {
            "round": {"synthesis": {"method": "round"}},
            "missing synthesis": {},
            "missing method": {"synthesis": {}},
            "misspelled sample": {"synthesis": {"method": "sampl"}},
        }
        for label, config in invalid.items():
            with self.subTest(configuration=label):
                with self.assertRaises(ConfigurationError):
                    validate_synthesis({"output_rows": ROWS, **config})

    def test_synthesis_explicitly_dispatches_sample_and_preserves_the_frame(self):
        expected = pd.DataFrame({"A": [1, 3, 7], "B": [6, 2, 0]})
        table = Mock()
        table.synthesize.return_value = pl.from_pandas(expected)
        config = {"output_rows": len(expected), "synthesis": {"method": "sample"}}

        actual = synthesize_table(table, config)

        table.synthesize.assert_called_once_with(rows=len(expected), method="sample")
        pd.testing.assert_frame_equal(actual, expected)

    def test_materialization_rejects_round_before_calling_the_dependency(self):
        table = Mock()
        config = {"output_rows": ROWS, "synthesis": {"method": "round"}}
        with self.assertRaises(ConfigurationError):
            synthesize_table(table, config)
        table.synthesize.assert_not_called()


class IndependentStarSamplerTests(unittest.TestCase):
    def setUp(self):
        self.numpy_state = np.random.get_state()
        self.table = independent_star_table()

    def tearDown(self):
        np.random.set_state(self.numpy_state)

    def test_actual_sample_path_does_not_create_child_correlations(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                frame = synthesize_table(self.table, sample_config())
                self.assertIsInstance(frame, pd.DataFrame)
                self.assertEqual(frame.shape, (ROWS, 4))
                self.assertEqual(set(frame.columns), {"H", *CHILDREN})
                self.assertFalse(frame.isna().to_numpy().any())
                for column, upper in [("H", 1), *[(c, 7) for c in CHILDREN]]:
                    self.assertTrue(frame[column].between(0, upper).all())
                correlations = frame[list(CHILDREN)].corr(method="spearman")
                for left, right in combinations(CHILDREN, 2):
                    observed = float(correlations.loc[left, right])
                    self.assertTrue(np.isfinite(observed))
                    self.assertLess(
                        abs(observed),
                        0.15,
                        f"seed={seed}, pair={left}/{right}, Spearman={observed}",
                    )

    def test_pinned_round_control_reproduces_the_known_defect(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                np.random.seed(seed)
                frame = self.table.synthesize(rows=ROWS, method="round").to_pandas()
                correlations = frame[list(CHILDREN)].corr(method="spearman")
                for left, right in combinations(CHILDREN, 2):
                    observed = float(correlations.loc[left, right])
                    self.assertGreater(
                        observed,
                        0.99,
                        f"Pinned round defect changed: seed={seed}, "
                        f"pair={left}/{right}, Spearman={observed}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
