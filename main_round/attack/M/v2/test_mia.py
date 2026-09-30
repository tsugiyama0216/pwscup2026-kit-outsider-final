"""MIA invariance and joint-world contract checks; no personal values printed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes, load_runtime
from M.v1 import model as baseline
from M.v1.actual import apply_private_mask
from M.v2.mia import STRENGTHS, adjust_scores, raw_scores
from M.v2.shadow_joint import build_world


class AdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.default_rng(9921).normal(size=(51, 6))
        self.raw[1] = self.raw[0]
        self.mask = np.arange(51) < 7

    def test_strength_zero_is_exact_frozen_v1_with_and_without_mask(self):
        expected = np.column_stack([baseline._rank01(self.raw[:, j]) for j in range(self.raw.shape[1])])
        self.assertTrue(np.array_equal(adjust_scores(self.raw, strength=0), expected))
        masked = np.column_stack([apply_private_mask(expected[:, j], self.mask) for j in range(expected.shape[1])])
        self.assertTrue(np.array_equal(adjust_scores(self.raw, self.mask, 0), masked))

    def test_target_and_candidate_permutation_invariance_and_input_nonmutation(self):
        row_order = np.random.default_rng(9922).permutation(len(self.raw))
        col_order = np.array([4, 1, 5, 0, 3, 2])
        before, mask_before = self.raw.copy(), self.mask.copy()
        for strength in STRENGTHS:
            expected = adjust_scores(self.raw, self.mask, strength)
            actual = adjust_scores(self.raw[row_order][:, col_order], self.mask[row_order], strength)
            self.assertTrue(np.array_equal(actual, expected[row_order][:, col_order]),
                            "Permutation changed scores; arrays suppressed")
        self.assertTrue(np.array_equal(self.raw, before) and np.array_equal(self.mask, mask_before))

    def test_ties_bounds_and_known_negatives(self):
        self.mask[:2] = False
        for strength in STRENGTHS:
            result = adjust_scores(self.raw, self.mask, strength)
            self.assertTrue(np.array_equal(result[0], result[1]), "Equal evidence received different scores")
            self.assertTrue(np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)))
            self.assertTrue(np.all(result[self.mask] == 0) and np.all(result[~self.mask] > 0))
        result = adjust_scores(np.ones((51, 6)), self.mask, 1)
        self.assertTrue(np.all(result[~self.mask] == result[~self.mask][0]), "No-evidence rows were forced apart")

    def test_full_centering_removes_shared_person_propensity(self):
        signal = np.array([[1., 2., 4.], [3., 2., 1.], [0., 0., 0.], [4., 0., 2.]])
        nuisance = np.array([10., -7., 20., 4.])[:, None]
        self.assertTrue(np.array_equal(adjust_scores(signal, strength=1),
                                       adjust_scores(signal + nuisance, strength=1)))

    def test_one_target_has_no_cross_target_adjustment_and_empty_query_is_supported(self):
        single = self.raw[:, :1]
        self.assertTrue(np.array_equal(adjust_scores(single, strength=1), adjust_scores(single, strength=0)))
        self.assertEqual(adjust_scores(np.empty((0, 3)), strength=.5).shape, (0, 3))

    def test_invalid_shapes_masks_and_values_are_rejected(self):
        invalid = [(np.ones(5), None, .5), (np.empty((5, 0)), None, .5),
                   (np.array([[np.nan]]), None, .5), (np.array([[np.inf]]), None, .5),
                   (self.raw, np.ones(50, dtype=bool), .5),
                   (self.raw, np.zeros(51, dtype=int), .5),
                   (self.raw, None, float("nan")), (self.raw, None, -.1), (self.raw, None, 1.1)]
        for raw, mask, strength in invalid:
            with self.subTest(kind="invalid contract"):
                with self.assertRaises(ValueError):
                    adjust_scores(raw, mask, strength)


class JointWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verified = verify_hashes()
        if verified["verified_files"] != 94 or verified["warnings"]:
            raise RuntimeError("Unexpected source verification")
        originals = ROOT.parent / "anonymization/team10/participant_data"
        cls.source = pd.read_csv(originals / "A_bg.csv", float_precision="round_trip")
        cls.background = pd.read_csv(originals / "B.csv", float_precision="round_trip")
        cls.schema = json.loads((originals / "schema.json").read_text(encoding="utf-8-sig"))
        cls.worlds = [build_world(cls.source, cls.background, cls.schema, split="dev", seed=2026093041,
                                 visible_rare_total=total, layout="mixed") for total in (0, 5, 11)]

    def test_exclusivity_invisible_cohorts_and_fixed_known_private(self):
        for world in self.worlds:
            self.assertEqual(world.truth.shape, (1049, 22))
            self.assertTrue(np.all(world.truth.sum(axis=1) <= 1))
            self.assertFalse(bool(world.truth[world.private_mask].any()))
            self.assertEqual(int(world.private_mask.sum()), 163)
            self.assertEqual(int(world.rare.sum()), 24)
            self.assertEqual(int(world.rare[world.private_mask].sum()), 13)
            counts = world.truth.sum(axis=0)
            self.assertTrue(np.all((counts >= 25) & (counts <= 27)))
            self.assertEqual(int(counts.sum()) + world.metadata["invisible7_shared_total"], 736)
            rare_counts = world.truth[world.rare].sum(axis=0)
            self.assertEqual(int(rare_counts.sum()), world.metadata["visible_rare_total"])
            self.assertTrue(np.all(rare_counts <= 2) and int(rare_counts.sum()) <= 11)
            self.assertGreaterEqual(int(np.count_nonzero(rare_counts == 0)), 11)

    def test_same_world_rare_counts_do_not_change_candidates_mask_or_styles(self):
        first = self.worlds[0]
        for world in self.worlds[1:]:
            self.assertTrue(world.candidates.equals(first.candidates), "Candidate pool changed across rare configurations")
            self.assertTrue(np.array_equal(world.private_mask, first.private_mask))
            self.assertTrue(np.array_equal(world.rare, first.rare))
            self.assertEqual(world.metadata["styles"], first.metadata["styles"])
            self.assertEqual(world.metadata["visible_shared_counts"], first.metadata["visible_shared_counts"])

    def test_raw_scores_match_v1_and_ignore_identifiers_or_labels(self):
        world = self.worlds[1]
        c, bg, queries = world.releases[0], world.background, world.candidates
        before = [f.copy(deep=True) for f in (c, bg, queries)]
        actual = raw_scores(c=c, background=bg, candidates=queries)
        cc, bb = baseline._canonical(c), baseline._canonical(bg)
        qq = queries.loc[:, baseline.PUBLIC_COLUMNS].copy()
        expected = baseline._local_density_score(cc, bb, qq, baseline._full_features((cc, bb, qq), bb), 7.)
        self.assertTrue(np.array_equal(actual, expected), "Full raw score differs from frozen v1")
        changed = queries.copy(deep=True)
        changed["record_id"] = np.arange(len(changed))[::-1] + 99999
        changed["unavailable_membership_truth"] = np.arange(len(changed)) % 2
        changed["is_rare"] = False
        again = raw_scores(c=c, background=bg, candidates=changed)
        self.assertTrue(np.array_equal(again, actual), "Identifiers or extra labels changed scoring")
        self.assertTrue(all(a.equals(b) for a, b in zip(before, (c, bg, queries))), "Raw scoring mutated inputs")

    def test_raw_scores_preserve_query_alignment_and_reference_permutation(self):
        world = self.worlds[2]
        order = np.random.default_rng(9923).permutation(1049)
        for c in world.releases[:2]:
            actual = raw_scores(c=c, background=world.background, candidates=world.candidates)
            changed = raw_scores(c=c.sample(frac=1, random_state=91),
                background=world.background.sample(frac=1, random_state=92),
                candidates=world.candidates.iloc[order])
            self.assertTrue(np.array_equal(changed, actual[order]), "Permutation changed raw scores")


class OfficialMetricParityTests(unittest.TestCase):
    def test_official_all_rare_degenerate_and_tied_score_equivalence(self):
        from M.v2.evaluate_mia import metric
        from pwscup2026.scoring.attack_score import _tpr_at_fpr, score_mia
        _, _, _, cfg, _ = load_runtime()
        scores = np.array([1., .9, .9, .7, .7, .7, .3, .2, .2, .1, 0., 0.])
        truth = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        cases = [
            (scores, truth, np.zeros(len(truth), dtype=bool)),
            (scores, truth, truth.copy()),
            (scores, truth, np.ones(len(truth), dtype=bool)),
            (scores, truth, np.arange(len(truth)) % 2 == 0),
            (np.full(len(truth), .5), truth, np.arange(len(truth)) % 2 == 0),
            (scores, np.zeros(len(truth), dtype=bool), np.ones(len(truth), dtype=bool)),
            (scores, np.ones(len(truth), dtype=bool), np.ones(len(truth), dtype=bool)),
        ]
        for confidence, membership, rare in cases:
            with self.subTest(kind="official ROC/rare boundary"):
                expected = score_mia(confidence, membership.astype(int), cfg, rare_mask=rare)
                actual = metric(confidence, membership, rare, float(cfg["scoring"]["low_fpr"]), _tpr_at_fpr)
                self.assertEqual(actual["mia"], expected.score)
                self.assertEqual(actual["mia_all"], expected.breakdown["tpr_at_low_fpr"])
                if actual["mia_rare"] is None:
                    self.assertTrue(np.isnan(expected.breakdown["tpr_rare"]))
                else:
                    self.assertEqual(actual["mia_rare"], expected.breakdown["tpr_rare"])


if __name__ == "__main__":
    unittest.main()
