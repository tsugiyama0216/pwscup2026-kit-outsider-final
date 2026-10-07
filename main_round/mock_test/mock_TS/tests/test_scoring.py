"""Official arithmetic parity, floor provenance and malformed-answer rejection."""
import json
import shutil
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

from mock_TS.common import OUTPUT_ROOT, cfg, read_csv
from mock_TS.scoring import (_aia_result, _floor_inputs, _reference_time_samples,
                             FLOOR_METHOD, estimate_floor, score_predictions)
from mock_TS.report import write_report
from pwscup2026.attack.aia import aia_predict
from pwscup2026.kit.codabench.scoring_io import score_attack
from pwscup2026.scoring.attack_score import score_mia


def make_fixture(root):
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir()
    idx = np.arange(1049)
    candidates = pd.DataFrame({"record_id": idx + 2000})
    member = (idx < 2) | ((idx >= 24) & (idx < 48))
    mia_truth = pd.DataFrame({"record_id": candidates.record_id, "member": member.astype(int),
                              "is_rare": (idx < 24).astype(int)})
    qidx = np.arange(326)
    query = pd.DataFrame({"challenge_row_id": qidx + 6000, "age": 40 + qidx % 35,
                          "SBP": 120. + qidx / 10, "TG": 100. + qidx / 20,
                          "HDL": 50. + qidx / 100, "ALT": 25. + qidx / 100,
                          "sex": qidx % 2, "prefecture": "P"})
    aia_truth = pd.DataFrame({"challenge_row_id": query.challenge_row_id,
                              "label": np.where(qidx < 163, "member", "control"),
                              "time": 1. + (qidx % 9), "onset": (qidx % 4 == 0).astype(int),
                              "is_rare": ((qidx % 163) < 13).astype(int)})
    aidx = np.arange(1080)
    background = pd.DataFrame({"record_id": aidx, "age": 40 + aidx % 35,
        "SBP": 120. + aidx / 10, "TG": 100. + aidx / 20, "HDL": 50. + aidx / 100,
        "ALT": 25. + aidx / 100, "sex": aidx % 2, "prefecture": "P",
        "time": 1. + aidx % 9, "onset": (aidx % 4 == 0).astype(int)})
    for data, relative in [(candidates, "public/B.csv"), (query, "public/aia_challenge_0.csv"),
                           (background, "public/A_bg.csv"), (mia_truth, "private/mia_truth.csv"),
                           (aia_truth, "private/aia_truth.csv")]:
        data.to_csv(root / relative, index=False)
    mia = pd.DataFrame({"record_id": candidates.record_id, "0": np.where(member, .9, .1)})
    aia = pd.DataFrame({"target_k": 0, "challenge_row_id": query.challenge_row_id,
                        "time_hat": np.where(qidx < 163, aia_truth.time, 20.)})
    floor = {"phi": .2, "sd": .1, "se": .1 / np.sqrt(1000), "repetitions": 1000,
        "seed": 17, "sample_rows": 1049, "method": FLOOR_METHOD,
        "diagnostic_reduced_repetitions": False, "mock_truth_only": True,
        "official_hidden_floors_known": False, "input_sha256": _floor_inputs(root, cfg())}
    return candidates, query, mia, aia, floor


class ScoringTests(unittest.TestCase):
    def setUp(self):
        # Inherit workspace ACLs rather than Windows mkdtemp(mode=0700).
        self.root = OUTPUT_ROOT / ("test-scoring-" + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.candidates, self.query, self.mia, self.aia, self.floor = make_fixture(self.root)

    def cleanup(self):
        path = self.root.resolve()
        self.assertEqual(path.parent, OUTPUT_ROOT.resolve())
        self.assertTrue(path.name.startswith("test-scoring-"))
        shutil.rmtree(path)

    def score(self, mia=None, aia=None, floor=None):
        return score_predictions(self.root, self.mia if mia is None else mia,
                                 self.aia if aia is None else aia,
                                 self.floor if floor is None else floor,
                                 self.root / ("score-" + uuid4().hex))

    def test_official_parity_and_floor_order(self):
        scratch = self.root / "official"
        actual = score_predictions(self.root, self.mia, self.aia, self.floor, scratch)
        expected = score_attack({"F_mia.csv": self.mia, "F_aia.csv": self.aia}, scratch, cfg(), team_id=None)
        self.assertEqual(actual["official_result"], expected)
        self.assertEqual(actual["score_1"], 1.8)
        self.assertEqual(actual["score_2"], 1.)
        self.assertEqual(actual["score_4"], 1.)
        self.assertEqual(actual["aia_before_floor"], 1.)
        self.assertEqual(actual["aia_power"], .8)
        self.assertIsNone(actual["score_6"])
        self.assertAlmostEqual(actual["provided_phi10_sensitivity"]["aia_power"], .973113)

    def test_uninformative_mia_and_rare_max_use_official_arithmetic(self):
        mia = self.mia.copy()
        mia["0"] = .5
        actual = self.score(mia=mia)
        self.assertAlmostEqual(actual["mia_power"], .01)
        truth = read_csv(self.root / "private/mia_truth.csv")
        mia.loc[0, "0"] = 1.
        expected = score_mia(mia["0"].to_numpy(), truth.member.to_numpy(), cfg(),
                             rare_mask=truth.is_rare.to_numpy(dtype=bool))
        actual = self.score(mia=mia)
        self.assertEqual(actual["mia_power"], expected.score)
        self.assertGreater(actual["score_3"], actual["score_2"])

    def test_shuffled_ids_produce_identical_result(self):
        original = self.score()
        permuted = self.score(mia=self.mia.sample(frac=1, random_state=20),
                              aia=self.aia.sample(frac=1, random_state=21))
        self.assertEqual(original, permuted)

    def test_reject_malformed_mia(self):
        cases = []
        cases.append(self.mia.iloc[:-1].copy())
        cases.append(pd.concat([self.mia, self.mia.iloc[:1]], ignore_index=True))
        x = self.mia.copy(); x.loc[0, "record_id"] = 999999; cases.append(x)
        x = self.mia.astype({"record_id": float}); x.loc[0, "record_id"] = 1.5; cases.append(x)
        x = self.mia.rename(columns={"0": "1"}); cases.append(x)
        x = self.mia.copy(); x["unexpected"] = .5; cases.append(x)
        for value in [np.nan, np.inf, -np.inf, -.01, 1.01]:
            x = self.mia.copy(); x.loc[0, "0"] = value; cases.append(x)
        for x in cases:
            with self.subTest(case=str(x.iloc[0].to_dict())):
                with self.assertRaises((ValueError, TypeError)):
                    self.score(mia=x)

    def test_reject_malformed_aia(self):
        cases = [self.aia.iloc[:-1].copy(), pd.concat([self.aia, self.aia.iloc[:1]], ignore_index=True)]
        for column, value in [("target_k", 1), ("target_k", .5), ("challenge_row_id", 99999),
                              ("time_hat", np.nan), ("time_hat", np.inf), ("time_hat", -.1)]:
            x = self.aia.astype({column: float}); x.loc[0, column] = value; cases.append(x)
        for x in cases:
            with self.subTest(case=str(x.iloc[0].to_dict())):
                with self.assertRaises((ValueError, TypeError)):
                    self.score(aia=x)

    def test_floor_missing_invalid_or_bound_to_other_input_fails_closed(self):
        for floor in [None, {}, {**self.floor, "phi": np.nan}, {**self.floor, "phi": 1.1},
                      {**self.floor, "repetitions": 1}, {**self.floor, "se": .5},
                      {**self.floor, "input_sha256": {}}, {**self.floor, "mock_truth_only": False}]:
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    score_predictions(self.root, self.mia, self.aia, floor, self.root / uuid4().hex)

    def test_truth_cannot_silently_drop_queries(self):
        truth_path = self.root / "private/aia_truth.csv"
        truth = read_csv(truth_path)
        truth.iloc[:-1].to_csv(truth_path, index=False)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            self.score()

    def test_floor_matches_direct_pinned_reference(self):
        seed, repetitions = 51, 3
        actual = estimate_floor(self.root, seed, repetitions)
        background = read_csv(self.root / "public/A_bg.csv")
        truth = read_csv(self.root / "private/aia_truth.csv")
        query = read_csv(self.root / "public/aia_challenge_0.csv")
        rng = np.random.default_rng(seed)
        direct = []
        for _ in range(repetitions):
            sample = rng.choice(len(background), 1049, replace=False)
            _, predictions = aia_predict(background.iloc[sample].reset_index(drop=True), query, background, cfg())
            direct.append(_aia_result(predictions, truth, cfg()).score)
        self.assertEqual(actual["phi"], float(np.mean(direct)))
        self.assertEqual(actual["sd"], float(np.std(direct, ddof=1)))
        self.assertTrue(actual["diagnostic_reduced_repetitions"])
        self.score(floor=actual)
        with (self.root / "public/A_bg.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "different questions"):
            self.score(floor=actual)

    def test_cached_reference_ties_and_block_fallbacks(self):
        background = pd.DataFrame({"age": [50., 50., 55., 55., 60., 65., 70., 70.],
            "SBP": [120.] * 8, "TG": [90.] * 8, "HDL": [60.] * 8, "ALT": [30.] * 8,
            "sex": [0, 0, 0, 1, 1, 1, 1, 1], "prefecture": ["A", "A", "B", "A", "B", "B", "C", "C"],
            "onset": [0, 1, 0, 1, 0, 1, 0, 1], "time": [1., 5., 2., 4., 3., 6., 2., 8.]})
        query = background.iloc[[0, 1, 7]].copy().reset_index(drop=True)
        query.loc[1, "prefecture"] = "missing"
        query.loc[2, "sex"] = 9
        samples = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [0, 1, 6, 7]])
        cached = _reference_time_samples(background, query, samples, cfg())
        direct = np.stack([aia_predict(background.iloc[s].reset_index(drop=True), query, background, cfg())[1]
                           for s in samples])
        np.testing.assert_array_equal(cached, direct)
        self.assertEqual(cached[0, 0], 3.)

    def test_report_is_portable_and_escapes_untrusted_labels(self):
        row = {"case_id": "<script>alert(1)</script>", "model": "M1_v5", "method": "reference", **self.score()}
        diagnostic = {**row, "method": "background_only"}
        report = write_report(self.root / "report", [row, diagnostic], {"note": "</pre><script>bad</script>"})
        html = report.read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Cなしの診断", html)
        payload = json.loads((report.parent / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["case_means"]), 1)
        self.assertEqual(payload["case_means"][0]["case_count"], 1)
        self.assertTrue((report.parent / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
