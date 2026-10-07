"""Joint topology and public/private boundary regression tests."""
from collections import Counter
import unittest

import numpy as np
import pandas as pd

from sugiyama_v2.generate import (ACTIVE_TARGETS, FEATURES, MEMBER_QUOTAS, QI,
                                 TEAM_ID, _kernel, build_world)


def copy_release(target, background, style, seed, schema):
    return target[FEATURES].copy()


class JointGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build_world(2026100801, release_factory=copy_release)

    def test_every_cohort_has_official_size_and_birth_rare_quota(self):
        private = self.world["private"]
        pool = private["pool"].set_index("pool_id")
        self.assertEqual(len(private["cohorts"]), 30)
        for ids in private["cohorts"].values():
            self.assertEqual(len(ids), 1049)
            self.assertEqual(len(set(ids)), 1049)
            self.assertEqual(int(pool.loc[ids].is_rare.sum()), 24)
            self.assertEqual(int((pool.loc[ids].is_rare * pool.loc[ids].onset).sum()),
                             self.world["metadata"]["cohort_rare_onset"])

    def test_pair_topology_is_globally_consistent(self):
        cohorts = {team: set(ids) for team, ids in self.world["private"]["cohorts"].items()}
        pool = self.world["private"]["pool"].set_index("pool_id")
        appearances = Counter(index for ids in cohorts.values() for index in ids)
        self.assertEqual(set(appearances.values()), {1, 2})
        self.assertEqual(sum(count == 2 for count in appearances.values()), 11040)
        self.assertEqual(sum(count == 1 for count in appearances.values()), 9390)
        for left, ids in cohorts.items():
            shared = []
            shared_rare = 0
            for right, other in cohorts.items():
                if right == left:
                    continue
                overlap = ids & other
                self.assertIn(len(overlap), (25, 26, 27))
                rare_count = int(pool.loc[sorted(overlap)].is_rare.sum())
                self.assertEqual(rare_count, len(overlap) - 25)
                shared_rare += rare_count
                shared.extend(overlap)
            self.assertEqual(shared_rare, 11)
            self.assertEqual(len(shared), 736)
            self.assertEqual(len(set(shared)), 736)
            self.assertEqual(sum(appearances[index] == 1 for index in ids), 313)

    def test_aia_truth_matches_same_target_and_exact_matched_controls(self):
        private = self.world["private"]
        target_pool = set(index for ids in private["cohorts"].values() for index in ids)
        pool = private["pool"].set_index("pool_id")
        all_controls = []
        for team, truth in private["aia_truth"].items():
            ids = np.asarray(private["challenge_pool_ids"][team])
            member_ids = ids[truth.label.eq("member")]
            control_ids = ids[truth.label.eq("control")]
            self.assertEqual(len(member_ids), 163)
            self.assertEqual(len(control_ids), 163)
            self.assertTrue(set(member_ids) <= set(private["cohorts"][team]))
            self.assertFalse(set(control_ids) & target_pool)
            all_controls.extend(control_ids.tolist())
            for label in ("member", "control"):
                rows = truth.loc[truth.label.eq(label)]
                observed = Counter(zip(rows.is_rare.astype(int), rows.onset.astype(int)))
                self.assertEqual(observed, Counter(MEMBER_QUOTAS))
            def keys(index):
                rows = pool.loc[index]
                return Counter(zip(rows.is_rare, rows.onset, rows.sex, rows.prefecture,
                                   (rows.age - 40) // 5))
            self.assertEqual(keys(member_ids), keys(control_ids))
            np.testing.assert_array_equal(truth.is_rare, pool.loc[ids].is_rare)
            np.testing.assert_array_equal(truth.onset, pool.loc[ids].onset)
            np.testing.assert_array_equal(truth.time, pool.loc[ids].time)
        self.assertEqual(len(all_controls), len(set(all_controls)))

    def test_public_self_proof_identifies_exactly163_private_candidates(self):
        public, private = self.world["public"], self.world["private"]
        keys = set(map(tuple, public["self_challenge"][QI].to_numpy()))
        mask = np.asarray([tuple(row) in keys for row in public["candidates"][QI].to_numpy()])
        self.assertEqual(int(mask.sum()), 163)
        known = set(np.asarray(private["candidate_pool_ids"])[mask])
        self.assertEqual(known, set(private["self_known_private_pool_ids"]))
        for team, ids in private["cohorts"].items():
            if team != TEAM_ID:
                self.assertFalse(known & set(ids))

    def test_public_contract_contains_no_hidden_truth(self):
        public = self.world["public"]
        self.assertEqual(set(public), {"team_id", "candidates", "self_labels", "background",
                                      "self_challenge", "targets", "schema"})
        self.assertEqual(tuple(public["targets"]), ACTIVE_TARGETS)
        self.assertEqual(list(public["candidates"].columns), ["record_id"] + FEATURES)
        self.assertEqual(list(public["self_labels"].columns), ["record_id"] + FEATURES + ["is_rare"])
        pd.testing.assert_frame_equal(public["candidates"], public["self_labels"].drop(columns="is_rare"))
        self.assertEqual(list(public["background"].columns), FEATURES)
        self.assertEqual(len(public["background"]), 5143)
        self.assertEqual(list(public["self_challenge"].columns), ["challenge_row_id"] + QI)
        for target in public["targets"].values():
            self.assertEqual(set(target), {"release", "challenge"})
            self.assertEqual(list(target["release"].columns), FEATURES)
            self.assertEqual(list(target["challenge"].columns), ["challenge_row_id"] + QI)
            self.assertEqual(len(target["challenge"]), 326)

    def test_same_seed_is_identical_and_new_seed_changes_world(self):
        again = build_world(2026100801, release_factory=copy_release)
        pd.testing.assert_frame_equal(self.world["private"]["pool"], again["private"]["pool"])
        pd.testing.assert_frame_equal(self.world["public"]["background"], again["public"]["background"])
        self.assertEqual(self.world["private"]["cohorts"], again["private"]["cohorts"])
        self.assertEqual(self.world["metadata"], again["metadata"])
        changed = build_world(2026100802, release_factory=copy_release)
        self.assertFalse(self.world["public"]["candidates"].equals(changed["public"]["candidates"]))
        self.assertNotEqual(self.world["private"]["cohorts"], changed["private"]["cohorts"])

    def test_holdout_has_disjoint_empirical_donor_rows(self):
        holdout = build_world(2026100801, split="holdout", release_factory=copy_release)
        for field in ("labeled_donor_indices", "background_donor_indices"):
            self.assertFalse(set(self.world["private"][field]) & set(holdout["private"][field]))
        self.assertIn("reused", " ".join(self.world["metadata"]["limitations"]))
        self.assertGreater(self.world["metadata"]["max_labeled_donor_reuse"], 1)

    def test_no_synthetic_identity_or_qi_collisions(self):
        private = self.world["private"]
        self.assertFalse(private["pool"].pool_id.duplicated().any())
        self.assertFalse(private["pool"][QI].duplicated().any())
        self.assertFalse(set(private["pool"].pool_id) & set(private["background_identity_ids"]))
        self.assertFalse(private["pool"][FEATURES].duplicated().any())

    def test_conditional_kernel_preserves_covariate_relationships_for_controls(self):
        # Distinct matching keys have deliberately different clinical values.
        # Copying sex/pref/age onto unconditional draws would mix these groups.
        template = self.world["public"]["self_labels"].iloc[[0, 0]].copy().reset_index(drop=True)
        template["is_rare"] = 0
        template["onset"] = 0
        template["sex"] = [0, 1]
        template["prefecture"] = ["甲県", "乙県"]
        template["age"] = [40, 74]
        template["BMI"] = [20., 40.]
        background = pd.concat([template.iloc[[0]][FEATURES]] * 3, ignore_index=True)
        matching = pd.concat([template.iloc[[0]]] * 500 + [template.iloc[[1]]] * 500,
                             ignore_index=True)
        values, _ = _kernel(template, [(0, 0)] * len(matching), background,
            np.random.default_rng(411), self.world["public"]["schema"], matching=matching)
        np.testing.assert_array_equal(values.sex, matching.sex)
        np.testing.assert_array_equal(values.prefecture, matching.prefecture)
        np.testing.assert_array_equal((values.age - 40) // 5, (matching.age - 40) // 5)
        np.testing.assert_allclose(values.BMI, matching.BMI, rtol=0, atol=.001)
        self.assertGreater(np.std(values.loc[values.sex.eq(0), "BMI"]), 0)
        unsupported = matching.copy()
        unsupported.loc[0, "prefecture"] = "未対応県"
        with self.assertRaisesRegex(ValueError, "never relaxed"):
            _kernel(template, [(0, 0)] * len(matching), background,
                    np.random.default_rng(411), self.world["public"]["schema"], matching=unsupported)

    def test_invalid_split_seed_or_profile_are_rejected(self):
        for kwargs in ({"split": "train"}, {"seed": -1}, {"profile": "easyonly"}):
            args = {"seed": 42, "release_factory": copy_release}
            args.update(kwargs)
            with self.assertRaises(ValueError):
                build_world(**args)


if __name__ == "__main__":
    unittest.main()
