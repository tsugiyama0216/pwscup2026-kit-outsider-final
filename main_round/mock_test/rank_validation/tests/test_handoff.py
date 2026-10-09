"""Evidence-integrity tests: no generated predictions or official score fitting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from rank_validation.handoff import load_handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.put("reference/official-kit/src/scorer.py", "SCORER_VERSION = 1\n")
        self.put("truth/case/pool.csv", "record_id,label\n1,0\n")
        self.put("config/case.json", "{}")
        self.put("predictions/mia.csv", "record_id,score\n1,0.1\n")
        self.put("predictions/aia.csv", "row_id,time\n1,0.5\n")
        self.put("evidence/saved.json", "{}")
        self.experiment = {"experiment_id": "mock_TS_test1", "role": "previously_examined",
                           "independent_new_population": False}
        self.fixture = {"fixture_id": "mock_TS_test1/Rv2/case_001/R1",
                        "experiment_id": "mock_TS_test1", "model_id": "Rv2",
                        "case_id": "case_001", "release_id": "R1",
                        "truth_dir": "truth/case", "config": "config/case.json",
                        "F_mia": "predictions/mia.csv", "F_aia": "predictions/aia.csv",
                        "expected_source": "evidence/saved.json", "attacker_team_id": None,
                        "expected_metrics": {"attack_power": 0.2, "mia_all": 0.1, "mia_rare": 0.0,
                                             "aia_all": 0.15, "aia_rare": 0.0},
                        "model_aliases_with_identical_predictions": ["Rv2f"]}
        self.json("data/experiments.json", [self.experiment])
        self.json("data/replay_fixtures.json", [self.fixture])
        self.json("data/summary.json", [])
        self.json("official/index.json", [])

    def put(self, reference, text):
        path = self.root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def json(self, reference, data):
        self.put(reference, json.dumps(data))

    def test_preserves_unknowns_aliases_and_historical_roles(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        data = load_handoff(self.root)
        row = data["records"][0]
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(row["score_1"], 0.2)
        self.assertIsNone(row["mia_power"])
        self.assertEqual(row["metadata"]["model_aliases_with_identical_predictions"], ["Rv2f"])
        self.assertEqual(row["metadata"]["experiment"]["role"], "previously_examined")
        self.assertFalse(row["metadata"]["independent_new_population"])
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_component_sources_remain_explicit(self):
        self.put("evidence/aia.json", "{}")
        self.fixture["expected_source"] = ["evidence/saved.json", "evidence/aia.json"]
        self.json("data/replay_fixtures.json", [self.fixture])
        row = load_handoff(self.root)["records"][0]
        self.assertEqual(row["source"], self.fixture["expected_source"])
        self.fixture["expected_source"].append("../outside.json")
        self.json("data/replay_fixtures.json", [self.fixture])
        with self.assertRaises(ValueError):
            load_handoff(self.root)

    def test_unknown_official_total_is_not_reconstructed(self):
        self.json("official/pl1.json", {"mia": {"tpr_at_low_fpr": 0.05, "tpr_rare": 0.2},
                                         "aia": {"r_time_all": 0.03, "r_time_rare": 0.04},
                                         "aia_per_target": {"1": 0.8, "2": 0.9}})
        self.json("official/index.json", [{"model_id": "pl1", "score_1": None,
                                           "breakdown_file": "official/pl1.json"},
                                          {"model_id": None, "result_id": "unidentified_official_01",
                                           "breakdown_file": "official/pl1.json", "must_not_assume_model": "Tv105"}])
        data = load_handoff(self.root)
        self.assertEqual(len(data["official"]), 1)
        self.assertIsNone(data["official"][0]["score_1"])
        self.assertEqual(data["official"][0]["score_2"], 0.05)
        self.assertEqual(data["unresolved"][0]["must_not_assume_model"], "Tv105")

    def test_content_fingerprint_tracks_truth_config_and_scorer(self):
        a = load_handoff(self.root)["records"][0]
        self.put("reference/official-kit/src/scorer.py", "SCORER_VERSION = 2\n")
        b = load_handoff(self.root)["records"][0]
        self.assertEqual(a["case_key"], b["case_key"])
        self.assertNotEqual(a["evaluation_sha256"], b["evaluation_sha256"])
        self.put("truth/case/pool.csv", "record_id,label\n1,1\n")
        c = load_handoff(self.root)["records"][0]
        self.assertNotEqual(b["case_key"], c["case_key"])
        self.put("config/case.json", '{"low_fpr":0.01}')
        d = load_handoff(self.root)["records"][0]
        self.assertNotEqual(c["evaluation_sha256"], d["evaluation_sha256"])

    def test_rejects_traversal_absolute_and_windows_references(self):
        for value in ("../outside.csv", "/absolute.csv", "C:/outside.csv", "truth\\case"):
            with self.subTest(value=value):
                fixture = copy.deepcopy(self.fixture)
                fixture["truth_dir"] = value
                self.json("data/replay_fixtures.json", [fixture])
                with self.assertRaises(ValueError):
                    load_handoff(self.root)

    def test_rejects_duplicate_fixture_and_scoring_observation(self):
        self.json("data/replay_fixtures.json", [self.fixture, self.fixture])
        with self.assertRaisesRegex(ValueError, "Duplicate fixture"):
            load_handoff(self.root)
        second = dict(self.fixture, fixture_id="other_fixture_id")
        self.json("data/replay_fixtures.json", [self.fixture, second])
        with self.assertRaisesRegex(ValueError, "Duplicate matched"):
            load_handoff(self.root)

    def test_rejects_nonfinite_boolean_and_conflicting_metric(self):
        for value in (float("nan"), float("inf"), True, "0.2"):
            with self.subTest(value=value):
                fixture = copy.deepcopy(self.fixture)
                fixture["expected_metrics"]["attack_power"] = value
                self.json("data/replay_fixtures.json", [fixture])
                with self.assertRaises(ValueError):
                    load_handoff(self.root)
        fixture = copy.deepcopy(self.fixture)
        fixture["expected_metrics"]["score_1"] = 0.3
        self.json("data/replay_fixtures.json", [fixture])
        with self.assertRaisesRegex(ValueError, "Conflicting metric"):
            load_handoff(self.root)

    def test_rejects_conflicting_or_duplicate_official_measurement(self):
        self.json("official/detail.json", {"scores": {"score_1": 0.5}})
        entry = {"model_id": "Rv2", "score_1": 0.6, "source": "official/detail.json"}
        self.json("official/index.json", [entry])
        with self.assertRaisesRegex(ValueError, "Conflicting official"):
            load_handoff(self.root)
        entry["score_1"] = 0.5
        self.json("official/index.json", [entry, entry])
        with self.assertRaisesRegex(ValueError, "Duplicate official"):
            load_handoff(self.root)

    def test_rejects_conflicting_nested_official_total(self):
        self.json("official/index.json", [{"model_id": "Rv2", "score_1": 0.5,
                                          "scores": {"score_1": 0.6}}])
        with self.assertRaisesRegex(ValueError, "Conflicting official"):
            load_handoff(self.root)

    def test_joint22_missing_release_has_explicit_origin(self):
        self.fixture.update(experiment_id="joint22_R_confirmation", release_id=None, attacker_team_id=10)
        self.experiment["experiment_id"] = "joint22_R_confirmation"
        self.json("data/replay_fixtures.json", [self.fixture])
        self.json("data/experiments.json", [self.experiment])
        row = load_handoff(self.root)["records"][0]
        self.assertEqual(row["release_id"], "joint22")
        self.assertIn("release_id=null", row["metadata"]["release_id_origin"])

    def test_rejects_duplicate_json_keys(self):
        self.put("data/experiments.json", '[{"experiment_id":"a","experiment_id":"b"}]')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            load_handoff(self.root)


if __name__ == "__main__":
    unittest.main()


