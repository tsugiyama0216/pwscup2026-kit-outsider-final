"""Regression tests for the preregistered v5 mean-first report gate."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from rank_validation import stable_report as s


SEEDS = (2026100961, 2026100962)


def frame(stable, lift=.1):
    return pd.DataFrame([{"target_k": k, "mia_all": stable - .05,
                          "mia_power": stable - .05 + lift, "aia_power": .05,
                          "attack_power": stable + lift, "mia_rare": None,
                          "mia_positive": 25, "mia_rare_positive": 0}
                         for k in s.TARGET_IDS]).set_index("target_k")


def cases(second=None):
    a = {"Rv2": .5, "Rv1": .4, "pl1": .3, "Tv105": .2}
    b = second or {"Rv2": .6, "Rv1": .25, "pl1": .3, "Tv105": .1}
    return [{"seed": seed, "role": "confirmation", "frames": {m: frame(v) for m, v in values.items()}}
            for seed, values in zip(SEEDS, (a, b))]


def official():
    return s._official_intervals(s._read(s.DEFAULT_OFFICIAL))


def summary(data=None):
    return s.summarize_cases(data or cases(), official(), SEEDS)


def html_fixture():
    report = summary()
    report.update(official_source=s._read(s.DEFAULT_OFFICIAL), raw_known_reference_rank_reproduced=False,
                  public_profile_coverage_insufficient=True,
                  freeze_record={"frozen_utc": "TEST ONLY"},
                  new_model_ranking_accuracy_validated=False)
    rows = []
    for panel in report["panels"]:
        panel["public_profile_coverage"] = {"unsupported_target_count": 11, "target_count": 22}
        for model, values in panel["model_metrics"].items():
            rows.append({"seed": panel["seed"], "model": model, **values})
    return report, rows


class StableSummaryTests(unittest.TestCase):
    def test_primary_mean_can_pass_when_one_case_fails(self):
        report = summary()
        self.assertTrue(report["known_reference_rank_reproduced"])
        self.assertEqual(report["fresh_confirmation_mean_rank_check"]["correct"], 6)
        self.assertEqual(report["individual_case_all_six_count"], 1)
        self.assertFalse(report["both_confirmation_cases_match_all_six"])
        self.assertEqual(report["individual_case_all_six_fraction"], .5)

    def test_mean_tie_fails_even_if_one_case_wins(self):
        report = summary(cases({"Rv2": .6, "Rv1": .2, "pl1": .3, "Tv105": .1}))
        self.assertFalse(report["known_reference_rank_reproduced"])
        pair = next(r for r in report["pair_case_sensitivity"] if {r["model_a"], r["model_b"]} == {"Rv1", "pl1"})
        self.assertEqual(pair["local_direction"], 0)
        self.assertFalse(pair["agrees"])

    def test_both_case_match_is_secondary(self):
        report = summary(cases({"Rv2": .6, "Rv1": .4, "pl1": .3, "Tv105": .1}))
        self.assertTrue(report["both_confirmation_cases_match_all_six"])
        self.assertEqual(report["individual_case_all_six_count"], 2)

    def test_pair_range_retains_reversal_not_confidence_interval(self):
        report = summary()
        pair = next(r for r in report["pair_case_sensitivity"] if {r["model_a"], r["model_b"]} == {"Rv1", "pl1"})
        self.assertLess(pair["case_difference_min"], 0)
        self.assertGreater(pair["case_difference_max"], 0)
        self.assertEqual(set(pair["case_differences"]), set(map(str, SEEDS)))
        self.assertFalse(pair["sensitivity_range_is_confidence_interval"])

    def test_same_target_is_removed_from_both_cases(self):
        data = cases({"Rv2": .6, "Rv1": .4, "pl1": .3, "Tv105": .1})
        target = s.TARGET_IDS[0]
        for case, value in zip(data, (.9, .05)):
            f = case["frames"]["Rv1"]
            f.loc[target, "mia_all"] = value
            f.loc[target, "mia_power"] = value + .1
            f.loc[target, "attack_power"] = value + .15
        report = summary(data)
        omitted = next(r for r in report["same_target_leave_one_out"] if r["omitted_target"] == target)
        self.assertAlmostEqual(omitted["model_metrics"]["Rv1"]["stable_index"], .4)
        self.assertEqual(omitted["target_count_per_case"], 21)
        self.assertEqual(len(report["same_target_leave_one_out"]), 22)
        for case in report["panels"]:
            self.assertEqual(len(case["leave_one_target_out"]), 22)
            for item in case["leave_one_target_out"]:
                self.assertEqual(item["model_metrics"]["Rv1"]["target_count"], 21)

    def test_raw_score_and_rare_lift_remain_separate(self):
        data = cases({"Rv2": .6, "Rv1": .4, "pl1": .3, "Tv105": .1})
        for case in data:
            f = case["frames"]["Tv105"]
            f["mia_power"] = .95
            f["attack_power"] = 1.
        report = summary(data)
        self.assertTrue(report["known_reference_rank_reproduced"])
        self.assertFalse(report["panels"][0]["raw_rank_check"]["all_six_correct"])
        metrics = report["confirmation_mean_metrics"]["Tv105"]
        self.assertAlmostEqual(metrics["official_attack_power"], metrics["stable_index"] + metrics["rare_lift"])
        row = report["panels"][0]["by_target"]["Tv105"][0]
        self.assertIsNone(row["mia_rare"])
        self.assertEqual(row["mia_rare_positive"], 0)

    def test_missing_duplicate_wrong_role_and_unknown_seed_fail(self):
        for data in (cases()[:1], [cases()[0], cases()[0]],
                     [dict(cases()[0], role="development"), cases()[1]],
                     [dict(cases()[0], seed=999), cases()[1]]):
            with self.assertRaises(ValueError):
                s.summarize_cases(data, official(), SEEDS)

    def test_missing_model_target_or_nonfinite_component_fail(self):
        for kind in ("model", "target", "nan"):
            data = cases()
            if kind == "model": data[0]["frames"].pop("Rv1")
            elif kind == "target": data[0]["frames"]["Rv1"] = data[0]["frames"]["Rv1"].iloc[1:]
            else: data[0]["frames"]["Rv1"].loc[s.TARGET_IDS[0], "mia_all"] = float("nan")
            with self.assertRaises(ValueError):
                summary(data)

    def test_official_duplicate_or_unresolved_pair_fails(self):
        source = s._read(s.DEFAULT_OFFICIAL)
        duplicate = copy.deepcopy(source)
        duplicate["rows"][0]["model_id"] = duplicate["rows"][1]["model_id"]
        with self.assertRaises(ValueError): s._official_intervals(duplicate)
        tied = copy.deepcopy(source)
        tied["rows"][0]["score_1_display"] = tied["rows"][1]["score_1_display"]
        with self.assertRaises(ValueError): s._official_intervals(tied)


class FrozenReportTests(unittest.TestCase):
    def test_real_protocol_freeze_contract(self):
        protocol, metric, freeze = s._verify_protocol(
            s.HERE / "generation_v5_protocol.json", s.DEFAULT_METRIC, s.DEFAULT_FREEZE)
        self.assertEqual(protocol["confirmation_seeds"], list(SEEDS))
        self.assertEqual(metric["version"], s.ranking_metric.METRIC_VERSION)
        self.assertFalse(freeze["v5_generation_started"])

    def test_changed_freeze_status_is_rejected(self):
        original = s._read
        def changed(path):
            value = original(path)
            if Path(path) == s.DEFAULT_FREEZE:
                value["new_seed_model_results_exist"] = True
            return value
        with patch.object(s, "_read", side_effect=changed), self.assertRaises(ValueError):
            s._verify_protocol(s.HERE / "generation_v5_protocol.json", s.DEFAULT_METRIC, s.DEFAULT_FREEZE)

    def test_incomplete_raw_backend_does_not_produce_primary_result(self):
        with patch.object(s.joint_report, "build_report", return_value=({"all_reserved_scenarios_present": False}, [])):
            with self.assertRaises(ValueError):
                s.build_report(["missing-a", "missing-b"])

    def test_writer_exposes_limitations_and_every_sensitivity(self):
        report, rows = html_fixture()
        parent = s.HERE.parent / "local-output" / ("unit-test-stable-report-" + uuid4().hex)
        parent.mkdir()
        destination = parent / "output"
        try:
            s.write_outputs(destination, report, rows)
            self.assertEqual(len(pd.read_csv(destination / "case_metrics.csv")), 8)
            self.assertEqual(len(pd.read_csv(destination / "target_metrics.csv")), 2*4*22)
            self.assertEqual(len(pd.read_csv(destination / "same_target_leave_one_out.csv")), 22*6)
            self.assertEqual(len(pd.read_csv(destination / "case_leave_one_out.csv")), 2*22*6)
            text = (destination / "report.html").read_text(encoding="utf-8")
            for phrase in ("未知モデル", "未検証", "不足あり", "信頼区間ではありません", "公式式の模擬総合", "希少 MIA 上乗せ", "順位比較指標", "11/22 対象", "除外後も全6組が一致"):
                self.assertIn(phrase, text)
            with self.assertRaises(ValueError):
                s.write_outputs(destination, report, rows)
        finally:
            # Only these flat generated test artifacts may be removed.
            self.assertEqual(destination.resolve().parent, parent.resolve())
            if destination.exists():
                for file in destination.iterdir():
                    self.assertTrue(file.is_file())
                    file.unlink()
                destination.rmdir()
            parent.rmdir()


class IntegrationAndCountTests(unittest.TestCase):
    def test_retained_counts_missing_null_or_inconsistent_fail(self):
        for value in (None, float("nan"), -1, 26):
            data = cases()
            data[0]["frames"]["Rv1"].loc[s.TARGET_IDS[0], "mia_rare_positive"] = value
            with self.assertRaises(ValueError): summary(data)
        data = cases()
        data[0]["frames"]["Rv1"].drop(columns="mia_positive", inplace=True)
        with self.assertRaises(ValueError): summary(data)

    def _report_fixture(self, floor_repetitions=1000):
        protocol, metric, freeze = s._verify_protocol(
            s.HERE / "generation_v5_protocol.json", s.DEFAULT_METRIC, s.DEFAULT_FREEZE)
        data = cases()
        paths = [Path("TEST_ONLY_" + str(seed)).resolve() for seed in SEEDS]
        states, raw_panels = {}, []
        signatures = {"test_fixture": "stable comparison identity"}
        coverage = {"model_ranking": {"unsupported": True}}
        for path, case in zip(paths, data):
            seed = case["seed"]
            meta = {"seed": seed, "reserved_seed_role": "confirmation",
                    "generator_source_sha256": freeze["generator_source_sha256"]}
            state = {"case_metadata": meta, "comparison_signatures": signatures,
                     "case_snapshot": {"manifest": str(seed)}, "public_sha256": {}, "truth_sha256": {},
                     "floor_repetitions": floor_repetitions}
            states[str(path)] = (state, [], case["frames"])
            raw_panels.append({"seed": seed, "path": str(path), "protocol_sha256": protocol["metric_protocol_sha256"],
                               "case_manifest_sha256": str(seed), "public_profile_coverage": coverage,
                               "public_profile_family_counts": {}, "scores": {m: float(f.attack_power.mean()) for m, f in case["frames"].items()}})
        raw = {"all_reserved_scenarios_present": True, "panels": raw_panels,
               "comparison_signatures": signatures, "known_reference_rank_reproduced": False}
        real_sha = s.sha
        def checksum(path):
            path = Path(path)
            return real_sha(path) if path.is_file() else "0" * 64
        return protocol, paths, states, raw, checksum

    def test_build_api_keeps_primary_raw_and_coverage_separate(self):
        protocol, paths, states, raw, checksum = self._report_fixture()
        with patch.object(s.joint_report, "build_report", return_value=(raw, [])), patch.object(s.joint_report, "load_panel", side_effect=lambda path: states[str(path)]), patch.object(s.joint_report, "hashes", return_value={}), patch.object(s, "sha", side_effect=checksum):
            report, rows = s.build_report(paths)
        self.assertTrue(report["known_reference_rank_reproduced"])
        self.assertFalse(report["raw_known_reference_rank_reproduced"])
        self.assertFalse(report["new_model_ranking_accuracy_validated"])
        self.assertTrue(report["public_profile_coverage_insufficient"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(report["metric_protocol_sha256"], protocol["metric_protocol_sha256"])
        self.assertNotIn(str(s.DEFAULT_METRIC.resolve()), report["source_sha256"])

    def test_reduced_floor_repetitions_in_either_case_are_rejected(self):
        for reduced in (20, 999):
            for affected_case in (0, 1):
                with self.subTest(repetitions=reduced, affected_case=affected_case):
                    _, paths, states, raw, checksum = self._report_fixture()
                    states[str(paths[affected_case])][0]["floor_repetitions"] = reduced
                    with patch.object(s.joint_report, "build_report", return_value=(raw, [])), patch.object(s.joint_report, "load_panel", side_effect=lambda path: states[str(path)]), patch.object(s.joint_report, "hashes", return_value={}), patch.object(s, "sha", side_effect=checksum):
                        with self.assertRaisesRegex(ValueError, "full 1000 floor repetitions"):
                            s.build_report(paths)


if __name__ == "__main__":
    unittest.main()

