"""Meaningful protocol checks using the actual local participant inputs."""
from collections import Counter
from pathlib import Path
import json
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from mock_TS.common import FEATURES, QI, OUTPUT_ROOT, PARTICIPANT, read_csv, sha256
from mock_TS.generation import generate, _keys, _official_members, _exact_csv, MIN_QI_DISTANCE


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cls.temporary = Path(tempfile.mkdtemp(prefix="test_generation_", dir=OUTPUT_ROOT)).resolve()
        cls.suite = cls.temporary / "suite"
        cls.manifest = generate(cls.suite, count=3)
        cls.b = read_csv(PARTICIPANT / "B_self.csv")
        cls.background = read_csv(PARTICIPANT / "A_bg.csv")

    @classmethod
    def tearDownClass(cls):
        # The test owns only this fresh subtree; never delete source inputs.
        if cls.temporary.is_relative_to(OUTPUT_ROOT.resolve()) and cls.temporary.name.startswith("test_generation_"):
            shutil.rmtree(cls.temporary)

    def case(self, index=0):
        return self.suite / self.manifest["cases"][index]["path"]

    def test_positive_negative_and_rare_quotas_and_background_removal(self):
        b_keys = set(_keys(self.b, FEATURES))
        for index in range(3):
            case = self.case(index)
            candidates = read_csv(case / "public/B.csv")
            truth = read_csv(case / "private/mia_truth.csv")
            background = read_csv(case / "public/A_bg.csv")
            self.assertEqual(list(candidates.columns), ["record_id"] + FEATURES)
            self.assertEqual(len(candidates), 1049)
            self.assertEqual(int(truth.member.sum()), 26)
            self.assertEqual(int(truth.is_rare.sum()), 24)
            self.assertEqual(int(truth.loc[truth.member == 1, "is_rare"].sum()), index)
            keys = _keys(candidates, FEATURES)
            self.assertEqual([key in b_keys for key in keys], truth.member.astype(bool).tolist())
            negative_keys = {key for key, member in zip(keys, truth.member) if not member}
            self.assertFalse(negative_keys & set(_keys(background, FEATURES)))
            self.assertEqual(len(background), len(self.background) - 1023)
            sources = read_csv(case / "private/mia_provenance.csv")
            aia_sources = read_csv(case / "private/aia_provenance.csv")
            member_b_rows = set(aia_sources.loc[aia_sources.source == "B", "source_id"])
            mia_b_rows = set(sources.loc[sources.source == "B", "source_row"])
            self.assertFalse(member_b_rows & mia_b_rows)

    def test_control_uniqueness_matching_and_event_validity(self):
        originals = pd.concat([self.b[FEATURES], self.background[FEATURES]], ignore_index=True)
        original_qi, original_full = set(_keys(originals, QI)), set(_keys(originals, FEATURES))
        for index in range(3):
            case = self.case(index)
            controls = read_csv(case / "private/controls.csv")
            provenance = read_csv(case / "private/control_provenance.csv")
            truth = read_csv(case / "private/aia_truth.csv")
            questions = read_csv(case / "public/aia_challenge_0.csv")
            protocol = json.loads((case / "private/protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(len(controls), 163)
            self.assertEqual(len(questions), 326)
            self.assertEqual(list(questions.columns), ["challenge_row_id"] + QI)
            self.assertEqual(truth.label.value_counts().to_dict(), {"control": 163, "member": 163})
            for label in ("member", "control"):
                subset = truth[truth.label == label]
                self.assertEqual(int(subset.is_rare.sum()), 13)
                self.assertEqual(int(subset.onset.sum()), 16)
                self.assertEqual(int((subset.is_rare * subset.onset).sum()), 4)
            self.assertFalse(set(_keys(controls, QI)) & original_qi)
            self.assertFalse(set(_keys(controls, FEATURES)) & original_full)
            self.assertEqual(len(set(_keys(controls, QI))), 163)
            self.assertTrue(((controls.onset + controls.death) <= 1).all())
            self.assertTrue(((controls.time > 0) & (controls.time <= 10)).all())
            self.assertTrue((controls.loc[(controls.onset == 0) & (controls.death == 0), "time"] == 10).all())
            self.assertTrue((provenance.minimum_scaled_qi_distance >= MIN_QI_DISTANCE).all())
            excluded = set(protocol["mia"]["excluded_A_bg_source_rows"])
            self.assertFalse(set(provenance.donor_A_bg_row_1) & excluded)
            self.assertFalse(set(provenance.donor_A_bg_row_2) & excluded)
            matched = self.b.set_index("record_id").loc[provenance.matched_B_record_id].reset_index()
            for column in ("prefecture", "sex", "onset", "is_rare"):
                self.assertEqual(matched[column].tolist(), controls[column].tolist())
            self.assertTrue(np.array_equal((matched.age.to_numpy() - 40) // 5,
                                           (controls.age.to_numpy() - 40) // 5))

    def test_manifest_seals_files_and_canonical_release_selection(self):
        releases = self.manifest["releases"]
        self.assertEqual([r["name"] for r in releases[:2]], ["M1_v5", "S1"])
        self.assertEqual(len(releases), 12)
        self.assertEqual(len({r["name"] for r in releases}), 12)
        for relative, digest in self.manifest["files"].items():
            self.assertEqual(sha256(self.suite / relative), digest)
        for case in self.manifest["cases"]:
            public = self.suite / case["path"] / "public"
            self.assertEqual({p.name for p in public.iterdir()},
                             {"B.csv", "A_bg.csv", "aia_challenge_0.csv", "schema.json", "cfg.json"})
        self.assertFalse(any("token" in key for key in self.manifest["sources"]))

    def test_reproducible_case_and_refuse_overwrite(self):
        second = self.temporary / "repeat"
        repeat = generate(second, count=1)
        self.assertEqual(repeat["releases"], self.manifest["releases"])
        original = {p: h for p, h in self.manifest["files"].items() if p.startswith("case_001/")}
        self.assertEqual(repeat["files"], original)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            generate(second, count=1)

    def test_insufficient_or_ambiguous_official_members_fail_closed(self):
        strings = _exact_csv(PARTICIPANT / "B.csv")
        with self.assertRaisesRegex(ValueError, "163"):
            _official_members(self.b, strings, strings.iloc[:3])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _official_members(self.b, pd.concat([strings, strings.iloc[:1]], ignore_index=True), strings)


if __name__ == "__main__":
    unittest.main()
