"""Saved-case evaluation preserves inputs and verifies truth alignment before scoring."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import attack_workspace as workspace
from local_eval.__main__ import predict_safely
from local_eval.data import FEATURES, QI, build_case, make_release
from local_eval.evaluate_case import (_ensure_empty_output, _report, load_case, load_cases,
                                      resolve_output, summarize_saved)
from local_eval.generate import export_case


class SavedCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, _, cls.cfg, _ = workspace.load_runtime()
        workspace.MOCK_OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.storage = tempfile.TemporaryDirectory(prefix="saved-case-tests-", dir=workspace.MOCK_OUTPUT)
        cls.base = Path(cls.storage.name) / "original"
        export_case(split="dev", seed=20260927, styles=("rounded", "copy"), output=cls.base)
        source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
        background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
        cls.schema = json.loads((workspace.ORIGINALS / "schema.json").read_text(encoding="utf-8"))
        cls.original = build_case(source, background, cls.schema, "dev", 20260927)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.area = tempfile.TemporaryDirectory(prefix="check-", dir=self.storage.name)
        self.root = Path(self.area.name) / "case"
        shutil.copytree(self.base, self.root)

    def tearDown(self):
        self.area.cleanup()

    def rehash(self, filename):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files_sha256"][filename] = workspace.sha256(self.root / filename)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_export_reload_is_lossless_and_scores_match(self):
        from local_eval.scoring import score_arrays
        saved = load_case(self.root)
        for name in ("candidates", "challenge", "background"):
            pd.testing.assert_frame_equal(getattr(saved.case, name), getattr(self.original, name))
        for name in ("mia_truth", "mia_rare", "aia_member", "aia_rare", "aia_time", "aia_onset", "aia_pair"):
            np.testing.assert_array_equal(getattr(saved.case, name), getattr(self.original, name))
        expected = make_release(self.original.target, self.original.background, "rounded", 20260927, self.schema)
        pd.testing.assert_frame_equal(saved.releases["rounded"], expected)
        mia = np.linspace(0, 1, 1049)
        aia = np.full(326, 10.)
        self.assertEqual(score_arrays(mia, aia, saved.case, self.cfg),
                         score_arrays(mia, aia, self.original, self.cfg))
        self.assertTrue(saved.case.target.empty)
        self.assertEqual(saved.case.private_roles, {})

    def test_tampered_file_hash_is_rejected(self):
        with (self.root / "public/C_rounded.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_case(self.root)

    def test_truth_id_order_mismatch_rejected_even_with_new_hash(self):
        filename = "private/mia_truth.csv"
        frame = pd.read_csv(self.root / filename, float_precision="round_trip")
        frame.iloc[[0, 1]] = frame.iloc[[1, 0]].to_numpy()
        frame.to_csv(self.root / filename, index=False)
        self.rehash(filename)
        with self.assertRaisesRegex(ValueError, "Truth ID order mismatch"):
            load_case(self.root)

    def test_invalid_pair_rejected_even_with_new_hash(self):
        filename = "private/aia_truth.csv"
        frame = pd.read_csv(self.root / filename, float_precision="round_trip")
        frame.loc[0, "pair_id"] = 200
        frame.to_csv(self.root / filename, index=False)
        self.rehash(filename)
        with self.assertRaisesRegex(ValueError, "pair IDs"):
            load_case(self.root)

    def test_manifest_path_traversal_rejected(self):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files_sha256"]["../outside.csv"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
            load_case(self.root)

    def test_predictor_receives_only_public_tables_and_independent_copies(self):
        saved = load_case(self.root)
        original = saved.case.challenge.copy(deep=True)

        def spy(**inputs):
            self.assertEqual(set(inputs), {"c", "background", "candidates", "challenge", "cfg"})
            self.assertEqual(list(inputs["c"].columns), FEATURES)
            self.assertEqual(list(inputs["background"].columns), FEATURES)
            self.assertEqual(list(inputs["candidates"].columns), ["record_id"] + FEATURES)
            self.assertEqual(list(inputs["challenge"].columns), ["challenge_row_id"] + QI)
            inputs["challenge"]["age"] = 0
            return {"mia": np.full(1049, .5), "aia": np.full(326, 10.)}

        predict_safely(spy, saved.releases["rounded"], saved.case, self.cfg)
        pd.testing.assert_frame_equal(saved.case.challenge, original)

    def test_suite_expansion_deduplicates_and_checks_child_manifest_hash(self):
        suite_root = Path(self.area.name) / "suite"
        suite_root.mkdir()
        child = suite_root / "seed_20260927"
        shutil.copytree(self.base, child)
        suite = {"format": "pwscup2026-local-shadow-suite-v1", "split": "dev", "seeds": [20260927],
                 "cases": [{"path": child.name, "manifest_sha256": workspace.sha256(child / "manifest.json")}],
                 "official_score_estimated": False}
        suite_path = suite_root / "suite.json"
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        cases, snapshots = load_cases([suite_root, child])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].root, child.resolve())
        self.assertEqual(snapshots[0]["sha256"], workspace.sha256(suite_path))
        suite["cases"][0]["manifest_sha256"] = "0" * 64
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Suite case manifest SHA256 mismatch"):
            load_cases([suite_root])

    def test_summary_pairs_case_identity_despite_same_seed(self):
        rows = []
        for case_id, baseline, difference in (("dev_case", .2, .1), ("holdout_case", .8, -.1)):
            for style in ("rounded", "copy"):
                for method in ("official", "candidate"):
                    score = baseline + (difference if method == "candidate" else 0)
                    rows.append({"case_id": case_id, "split": case_id.split("_")[0], "seed": 1,
                                 "style": style, "method": method,
                                 "mia": score, "aia_no_phi": 0., "combined_no_phi": score})
        summary, paired = summarize_saved(pd.DataFrame(rows))
        candidate = summary.set_index("method").loc["candidate"]
        self.assertEqual(candidate["cases"], 2)
        self.assertAlmostEqual(candidate.paired_delta_vs_official_mean, 0)
        self.assertEqual(candidate.paired_wins, 1)
        self.assertEqual(candidate.paired_losses, 1)
        self.assertEqual(len(paired), 8)
        report_directory = Path(self.area.name) / "report"
        report_directory.mkdir()
        _report(report_directory, paired, summary, {"null_repetitions": 0})
        report = (report_directory / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("| dev_case | dev | 1 | candidate |", report)
        self.assertIn("| holdout_case | holdout | 1 | candidate |", report)
        sanity_summary, sanity = summarize_saved(pd.DataFrame(rows).query("style == 'copy'"))
        self.assertTrue(sanity_summary.empty)
        self.assertFalse(sanity.primary.any())
        _report(report_directory, sanity, sanity_summary, {"null_repetitions": 0})
        self.assertIn("動作確認用の方式のみ", (report_directory / "REPORT.md").read_text(encoding="utf-8"))

    def test_optional_output_and_late_nonempty_check(self):
        default = resolve_output()
        self.assertEqual(default.parent, (workspace.MOCK_OUTPUT / "evaluations").resolve())
        self.assertTrue(default.name.startswith("saved-"))
        self.assertFalse(default.exists())
        output = resolve_output(Path(self.area.name) / "late-output")
        output.mkdir()
        (output / "existing.csv").write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            _ensure_empty_output(output)
        self.assertEqual((output / "existing.csv").read_text(encoding="utf-8"), "preserve me")


if __name__ == "__main__":
    unittest.main()
