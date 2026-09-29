import copy
import json
import unittest

import attack_workspace as workspace

workspace.verify_hashes()
np, pd, _, CFG, _ = workspace.load_runtime()
from local_eval.data import FEATURES, QI, QUOTAS, build_case, make_release, partition_source, rare_mask
from local_eval.__main__ import predict_safely, summarize
from local_eval.methods import multiview
from local_eval.scoring import score_arrays, validate_predictions, null_diagnostics
from pwscup2026.scoring.attack_score import score_mia


class LocalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
        cls.background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
        cls.schema = json.loads((workspace.ORIGINALS / "schema.json").read_text())
        cls.case = build_case(cls.source, cls.background, cls.schema, "dev", 20260927)

    def test_person_disjoint_partition_and_reference(self):
        parts = partition_source(self.source, self.schema)
        self.assertFalse(set(parts["dev"]) & set(parts["holdout"]))
        self.assertEqual(len(parts["dev"]) + len(parts["holdout"]), len(self.source))
        self.assertEqual(len(self.source[QI].merge(self.background[QI], on=QI)), 0)

    def test_reproducible_case_and_reserved_holdout_structure(self):
        again = build_case(self.source, self.background, self.schema, "dev", 20260927)
        self.assertEqual(again.metadata, self.case.metadata)
        pd.testing.assert_frame_equal(again.challenge, self.case.challenge)
        # Structure only: do not generate predictions or scores for holdout.
        hold = build_case(self.source, self.background, self.schema, "holdout", 20261001)
        left = set().union(*(set(v) for v in self.case.private_roles.values()))
        right = set().union(*(set(v) for v in hold.private_roles.values()))
        self.assertFalse(left & right)

    def test_mia_roles_and_rare_composition_all_development_seeds(self):
        for seed in (20260927, 20260928, 20260929):
            case = build_case(self.source, self.background, self.schema, "dev", seed)
            self.assertEqual(len(case.candidates), 1049)
            self.assertEqual(int(case.mia_truth.sum()), 26)
            self.assertEqual(int(case.mia_rare.sum()), 24)
            self.assertEqual(int(case.mia_rare[case.mia_truth == 1].sum()), seed % 3)
            roles = case.private_roles
            self.assertFalse(set(roles["mia_shared"]) & set(roles["aia_member"]))
            self.assertFalse(set(roles["target"]) & (set(roles["mia_nonmember"]) | set(roles["aia_control"])))

    def test_aia_exact_pair_matching_and_quotas(self):
        case = self.case
        for side in (case.aia_member, ~case.aia_member):
            for (rare, onset), quota in QUOTAS.items():
                self.assertEqual(int((side & (case.aia_rare == rare) & (case.aia_onset == onset)).sum()), quota)
        for pair in range(163):
            indices = np.flatnonzero(case.aia_pair == pair)
            self.assertEqual(len(indices), 2)
            self.assertEqual(int(case.aia_member[indices].sum()), 1)
            rows = case.challenge.iloc[indices]
            self.assertEqual(rows.sex.nunique(), 1)
            self.assertEqual(rows.prefecture.nunique(), 1)
            self.assertEqual(((rows.age - 40) // 5).nunique(), 1)

    def test_prediction_boundary_and_mutation_isolation(self):
        case = self.case
        original = case.challenge.copy()
        release = make_release(case.target, case.background, "copy", 20260927, self.schema)

        def spy(**inputs):
            self.assertEqual(set(inputs), {"c", "background", "candidates", "challenge", "cfg"})
            self.assertEqual(list(inputs["challenge"].columns), ["challenge_row_id"] + QI)
            self.assertEqual(list(inputs["c"].columns), FEATURES)
            self.assertNotIn("is_rare", inputs["candidates"])
            inputs["challenge"].loc[:, "age"] = 0
            inputs["cfg"]["scoring"]["low_fpr"] = .9
            return {"mia": np.full(1049, .5), "aia": np.full(326, 10.)}

        predict_safely(spy, release, case, CFG)
        pd.testing.assert_frame_equal(case.challenge, original)
        self.assertEqual(CFG["scoring"]["low_fpr"], .01)

    def test_invalid_prediction_rejection(self):
        for bad in (np.full(1049, np.nan), np.full(1049, 1.01), np.zeros(1048)):
            with self.assertRaises(ValueError):
                validate_predictions({"mia": bad, "aia": np.zeros(326)}, self.case)
        with self.assertRaises(ValueError):
            validate_predictions({"mia": np.zeros(1049), "aia": np.full(326, -1)}, self.case)

    def test_official_ties_and_perfect_prediction_directions(self):
        case = self.case
        constant = score_mia(np.full(1049, .5), case.mia_truth, CFG, rare_mask=case.mia_rare)
        self.assertAlmostEqual(constant.score, .01)
        truth_time = case.aia_time.copy()
        truth_time[~case.aia_member] += 1
        result = score_arrays(case.mia_truth.astype(float), truth_time, case, CFG)
        self.assertAlmostEqual(result["mia"], 1)
        self.assertAlmostEqual(result["aia_no_phi"], 1)
        self.assertAlmostEqual(result["combined_no_phi"], 2)

    def test_independent_release_does_not_depend_on_target(self):
        first = make_release(self.case.target, self.case.background, "independent", 20260927, self.schema)
        changed = self.case.target.copy()
        changed["time"] = .01
        second = make_release(changed, self.case.background, "independent", 20260927, self.schema)
        pd.testing.assert_frame_equal(first, second)

    def test_copula_release_is_reproducible_finite_and_not_exact_copy(self):
        first = make_release(self.case.target, self.case.background, "copula", 20260927, self.schema)
        second = make_release(self.case.target, self.case.background, "copula", 20260927, self.schema)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.shape, (1049, 13))
        self.assertTrue(np.isfinite(first.drop(columns="prefecture").to_numpy(float)).all())
        self.assertFalse(((first.onset == 1) & (first.death == 1)).any())
        self.assertEqual(len(first.merge(self.case.target, on=FEATURES)), 0)

    def test_copy_leakage_sanity(self):
        from threadpoolctl import threadpool_limits
        release = make_release(self.case.target, self.case.background, "copy", 20260927, self.schema)
        with threadpool_limits(limits=1):
            mia, aia = predict_safely(multiview, release, self.case, CFG)
        self.assertAlmostEqual(score_arrays(mia, aia, self.case, CFG)["mia"], 1)

    def test_null_diagnostics_reproducible_and_not_applied_to_score(self):
        mia, aia = np.full(1049, .5), np.full(326, 10.)
        score = score_arrays(mia, aia, self.case, CFG)
        first = null_diagnostics(mia, aia, self.case, CFG, 8)
        second = null_diagnostics(mia, aia, self.case, CFG, 8)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mia_null_mean"], .01)
        self.assertEqual(score, score_arrays(mia, aia, self.case, CFG))

    def test_primary_summary_excludes_sanity_styles_and_pairs_same_cases(self):
        rows = []
        for seed in (1, 2):
            for style in ("copy", "rounded", "jitter", "bootstrap", "independent"):
                for method in ("official", "new"):
                    score = (100 if style in ("copy", "independent") else seed) + (method == "new")
                    rows.append({"seed": seed, "style": style, "method": method, "mia": 0,
                                 "aia_no_phi": 0, "combined_no_phi": score})
        summary, paired = summarize(pd.DataFrame(rows))
        new = summary.set_index("method").loc["new"]
        self.assertEqual(new["cases"], 6)
        self.assertEqual(new["paired_delta_vs_official_mean"], 1)
        self.assertEqual(new["combined_no_phi_mean"], 2.5)
        self.assertEqual(set(paired["style"]), {"rounded", "jitter", "bootstrap"})


if __name__ == "__main__":
    unittest.main()
