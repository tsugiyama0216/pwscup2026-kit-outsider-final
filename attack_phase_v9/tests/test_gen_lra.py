import unittest

import numpy as np
from sklearn.neighbors import KernelDensity

from attack_phase_v9.gen_lra import LocalLikelihoodInfluence, reference_bandwidth


class LocalLikelihoodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(617)
        self.reference = rng.normal(size=(31, 3))
        self.released = rng.normal(size=(19, 3))
        self.queries = rng.normal(size=(7, 3))

    def test_matches_explicit_kde_refits(self):
        h = 0.42
        model = LocalLikelihoodInfluence(self.reference, h)
        observed = model.score_many(self.queries, self.released, neighbours=(1, 3, 19))
        for i, query in enumerate(self.queries):
            nearest = np.argsort(np.linalg.norm(self.released - query, axis=1))
            augmented = KernelDensity(bandwidth=h).fit(np.vstack([self.reference, query]))
            delta = augmented.score_samples(self.released) - model.model.score_samples(self.released)
            for count in (1, 3, 19):
                self.assertAlmostEqual(observed[f"gen_lra_k{count}"][i], delta[nearest[:count]].sum(), places=11)

    def test_query_and_release_order_invariance(self):
        model = LocalLikelihoodInfluence(self.reference, 0.5)
        first = model.score_many(self.queries, self.released)
        second = model.score_many(self.queries[::-1], self.released[::-1])
        for name in first:
            np.testing.assert_allclose(first[name], second[name][::-1], atol=1e-12)

    def test_exact_copy_strongly_exceeds_far_query(self):
        reference = np.arange(20.0)[:, None]
        release = np.array([[40.0], [40.01], [40.02]])
        result = LocalLikelihoodInfluence(reference, 0.1).score_many(np.array([[40.0], [-40.0]]), release, neighbours=(1, 3))
        self.assertGreater(result["gen_lra_k3"][0], result["gen_lra_k3"][1] + 10)

    def test_rejects_invalid_matrices_and_parameters(self):
        for value in (np.zeros((0, 3)), np.array([[np.inf]]), np.array([1.0])):
            with self.assertRaises(ValueError):
                LocalLikelihoodInfluence(value, 0.5)
        for h in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                LocalLikelihoodInfluence(self.reference, h)
        model = LocalLikelihoodInfluence(self.reference, 0.5)
        for sizes in ((0,), (20,), (1.5,), (1, 1), (True,)):
            with self.assertRaises(ValueError):
                model.score_many(self.queries, self.released, neighbours=sizes)
        with self.assertRaises(ValueError):
            model.score_many(self.queries[:, :2], self.released)

    def test_bandwidth_finite_for_duplicate_reference(self):
        self.assertGreater(reference_bandwidth(np.zeros((10, 2))), 0)
        with self.assertRaises(ValueError):
            reference_bandwidth(np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
