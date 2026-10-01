"""Independent behavioral checks for AIA and regional shadow generation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import attack_workspace as workspace
from M.v2 import aia
from M.v2 import evaluate_aia as evaluation


def table(values, sexes=None, prefectures=None, times=None):
    values = np.asarray(values, dtype=float)
    n = len(values)
    frame = pd.DataFrame({name: values.copy() for name in aia.PUBLIC if name != "prefecture"})
    frame["sex"] = np.zeros(n, dtype=int) if sexes is None else sexes
    frame["prefecture"] = ["A"] * n if prefectures is None else prefectures
    frame["time"] = np.linspace(1, 9, n) if times is None else times
    frame["onset"] = np.zeros(n, dtype=int)
    frame["smoking"] = np.zeros(n, dtype=int)
    frame["death"] = np.zeros(n, dtype=int)
    return frame[aia.PUBLIC]


class AiaBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.verify_hashes()
        _, _, cls.modules, cls.cfg, _ = workspace.load_runtime()

    def setUp(self):
        self.background = table(np.linspace(-1, 1, 61), sexes=np.arange(61) % 2)
        self.release = table([.7, .05, .05, 0], sexes=[0, 0, 0, 1],
                             prefectures=["A", "B", "B", "A"], times=[9, 2, 4, 1])
        self.query = table([0, .65], times=[3, 4])[aia.QI]

    def test_reference_matches_official_and_tie_average(self):
        outputs = aia.predict_variants(self.release, self.background, self.query, self.cfg)
        _, expected = self.modules["pwscup2026.attack.aia"].aia_predict(
            aia._canonical(self.release), self.query, aia._canonical(self.background), self.cfg)
        np.testing.assert_array_equal(outputs["v1_reference"], expected)
        self.assertEqual(outputs["support8"][0], 3.)
        self.assertEqual(outputs["v1_reference"][0], 9.)
        self.assertEqual(outputs["guarded40"][0], 3.)
        self.assertEqual(outputs["soft40"][0], 3.)

    def test_sparse_routing_keeps_nearest_when_distance_penalty_small(self):
        q = self.query.iloc[[1]]
        outputs = aia.predict_variants(self.release, self.background, q, self.cfg)
        self.assertEqual(outputs["guarded40"][0], 9.)
        diagnostic = aia.diagnose(self.release, self.background, q, self.cfg)
        self.assertEqual(diagnostic["routes"]["guarded40"]["reference"], 1)
        self.assertEqual(diagnostic["routes"]["support40"]["sex_only"], 1)

    def test_rich_reference_block_is_preserved_for_every_variant(self):
        release = table(np.r_[np.linspace(.4, .8, 40), 0],
                        prefectures=["A"] * 40 + ["B"], times=np.r_[np.repeat(9., 40), 1.])
        outputs = aia.predict_variants(release, self.background, self.query, self.cfg)
        for name, values in outputs.items():
            np.testing.assert_array_equal(values, outputs["v1_reference"], err_msg=name)

    def test_row_order_indexes_hidden_columns_and_input_immutability(self):
        expected = aia.predict_variants(self.release, self.background, self.query, self.cfg)
        copies = [frame.copy(deep=True) for frame in (self.release, self.background, self.query)]
        cfg_copy = copy.deepcopy(self.cfg)
        permuted = self.release.iloc[[3, 1, 0, 2]].copy()
        permuted.index = [100, 20, 777, -3]
        permuted["record_id"] = ["z", "c", "a", "b"]
        background = self.background.iloc[::-1].copy()
        background.index = np.arange(61) * 3 + 100
        query = self.query.copy()
        query["challenge_row_id"] = [900, 100]
        query["is_member"] = [True, False]
        query["is_rare"] = [False, True]
        query["time"] = [float("nan"), -9999]
        query["onset"] = [True, False]
        actual = aia.predict_variants(permuted, background, query, self.cfg)
        for name in aia.VARIANTS:
            np.testing.assert_array_equal(actual[name], expected[name])
        for original, unchanged in zip((self.release, self.background, self.query), copies):
            pd.testing.assert_frame_equal(original, unchanged)
        self.assertEqual(self.cfg, cfg_copy)

    def test_query_permutation_is_equivariant(self):
        expected = aia.predict_variants(self.release, self.background, self.query, self.cfg)
        actual = aia.predict_variants(self.release, self.background, self.query.iloc[::-1], self.cfg)
        for name in aia.VARIANTS:
            np.testing.assert_array_equal(actual[name], expected[name][::-1])

    def test_empty_block_and_absent_sex_use_official_fallback(self):
        query = self.query.copy()
        query["prefecture"] = "absent"
        query.loc[query.index[1], "sex"] = 7
        outputs = aia.predict_variants(self.release, self.background, query, self.cfg)
        for name in aia.VARIANTS:
            np.testing.assert_array_equal(outputs[name], outputs["v1_reference"])

    def test_empty_queries_and_invalid_features(self):
        outputs = aia.predict_variants(self.release, self.background, self.query.iloc[:0], self.cfg)
        self.assertTrue(all(value.shape == (0,) for value in outputs.values()))
        for bad in (self.release.drop(columns="onset"), self.release.iloc[:0]):
            with self.assertRaises(ValueError):
                aia.predict(bad, self.background, self.query, self.cfg)
        bad_query = self.query.copy()
        bad_query.loc[0, "age"] = np.inf
        with self.assertRaises(ValueError):
            aia.predict(self.release, self.background, bad_query, self.cfg)
        with self.assertRaises(ValueError):
            aia.predict(self.release, self.background, self.query.drop(columns="sex"), self.cfg)
        with self.assertRaises(ValueError):
            aia.predict(self.release, self.background, self.query, self.cfg, variant="unknown")

    def test_background_constant_columns_are_finite(self):
        background = self.background.copy()
        background["age"] = 50
        outputs = aia.predict_variants(self.release, background, self.query, self.cfg)
        self.assertTrue(all(np.isfinite(value).all() for value in outputs.values()))


class RegionalGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace.verify_hashes()
        cls.source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
        cls.background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
        cls.schema = json.loads((workspace.ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))

    def test_regional_mapping_is_consistent_across_roles_without_label_leak(self):
        home = evaluation.region_names(self.background)["rare"]
        case = evaluation.regional_case(self.source, self.background, self.schema,
                                        "dev", evaluation.SEEDS["dev"][0], home)
        self.assertEqual(case.metadata["regional_private_rows"], 313)
        self.assertEqual(case.challenge.prefecture.nunique(), 1)
        self.assertEqual(case.metadata["regional_challenge_home_count"], 326)
        # Compare a person using every untouched attribute, independently of
        # regional_case's original-row mapping implementation.
        same_person = [column for column in evaluation.FEATURES if column != "prefecture"]
        target_map = {row[:-1]: row[-1] for row in
                      case.target[same_person + ["prefecture"]].itertuples(index=False, name=None)}
        overlap = 0
        for row in case.candidates[same_person + ["prefecture"]].itertuples(index=False, name=None):
            if row[:-1] in target_map:
                self.assertTrue(row[-1] == target_map[row[:-1]], "shared person's region differs across roles")
                overlap += 1
        self.assertEqual(overlap, 26)
        no_region_qi = [column for column in evaluation.QI if column != "prefecture"]
        qmap = {row[:-1]: row[-1] for row in
                case.target[no_region_qi + ["prefecture"]].itertuples(index=False, name=None)}
        member_count = 0
        for row in case.challenge[no_region_qi + ["prefecture"]].itertuples(index=False, name=None):
            if row[:-1] in qmap:
                self.assertTrue(row[-1] == qmap[row[:-1]], "AIA member's region differs across roles")
                member_count += 1
        self.assertEqual(member_count, 163)
        for side in (True, False):
            mask = case.aia_member == side
            self.assertEqual(int(mask.sum()), 163)
            self.assertEqual(int(case.aia_rare[mask].sum()), 13)
            self.assertEqual(int(case.aia_onset[mask].sum()), 16)
        self.assertEqual(set(case.challenge.columns), set(evaluation.QI + ["challenge_row_id"]))

    def test_regional_processing_changes_only_release_county(self):
        release = table(np.linspace(0, 1, 100), prefectures=["A"] * 80 + ["B"] * 20)
        original = release.copy(deep=True)
        for mode in evaluation.REGION_PROCESSING:
            processed = evaluation.process_region(release, self.background, mode,
                                                   evaluation.SEEDS["dev"][0], "rare", "copy")
            pd.testing.assert_frame_equal(processed.drop(columns="prefecture"), release.drop(columns="prefecture"))
            if mode == "shuffle":
                self.assertEqual(processed.prefecture.value_counts().to_dict(), release.prefecture.value_counts().to_dict())
        pd.testing.assert_frame_equal(release, original)

    def test_nonempty_output_is_not_overwritten(self):
        base = Path(__file__).resolve().parent.parent / "local-output/v2"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="aia-test-", dir=base) as temp:
            keep = Path(temp) / "keep.txt"
            keep.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                evaluation.main(["--stage", "holdout", "--output", temp])
            self.assertEqual(keep.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
