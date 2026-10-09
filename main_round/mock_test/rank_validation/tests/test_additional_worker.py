import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rank_validation import additional_worker as worker
from rank_validation import additional_report as report
from rank_validation import plugins


class AdditionalWorkerTests(unittest.TestCase):
    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            worker.adapter_for("Rv2")

    def test_frozen_source_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            worker.source_snapshot("Mv1", {"main_round/attack/Attack_M/v1/model.py": "0" * 64})

    def test_unknown_fixed_source_path_rejected(self):
        with self.assertRaises(ValueError):
            worker.source_snapshot("Rv3", {"../model.py": "0" * 64})

    def test_output_outside_local_output_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(worker, "load_public") as load:
            with self.assertRaises(SystemExit):
                worker.main(["--model", "Mv1", "--public", tmp, "--output", str(Path(tmp) / "out")])
            load.assert_not_called()

    def test_all_fixed_models_reserved_against_plugins(self):
        for name in worker.MODELS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                plugins.candidate_name(name)

    def test_extended_ranks_include_all_ten_and_preserve_ties(self):
        rows = [{"model": m, "index": 0.5} for m in report.MODELS]
        ranked = report.rank_rows(rows)
        self.assertEqual(len(ranked), 10)
        self.assertTrue(all(r["rank"] == 1 for r in ranked))
        rows[0]["index"] = 0.6
        ranked = report.rank_rows(rows)
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertTrue(all(r["rank"] == 2 for r in ranked[1:]))

    def test_failed_reference_rank_is_not_displayed_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            report.write_html(path, {"ranking": [], "known_reference_rank_reproduced": False})
            html = path.read_text(encoding="utf-8")
            self.assertIn("再現は未達成", html)
            self.assertNotIn("この評価条件で再現済み", html)

    def test_report_rejects_reference_signature_or_case_switch(self):
        import copy
        identity = {"evaluation": {"sha": "evaluation"}, "generator": {"sha": "generator"},
                    "models": {"Rv2": {"sha": "rv2"}}}
        references = {"comparison_signatures": identity, "panels": [
            {"path": "panel", "protocol_sha256": "protocol", "case_manifest_sha256": "case"}]}
        candidate = {"comparison_signatures": copy.deepcopy(identity), "inputs": [
            {"panel": "panel", "protocol_sha256": "protocol", "case_manifest_sha256": "case"}]}
        report.validate_reference_binding(references, candidate)
        switched = copy.deepcopy(candidate)
        switched["comparison_signatures"]["models"]["Rv2"]["sha"] = "other"
        with self.assertRaises(ValueError):
            report.validate_reference_binding(references, switched)
        switched = copy.deepcopy(candidate)
        switched["inputs"][0]["case_manifest_sha256"] = "other"
        with self.assertRaises(ValueError):
            report.validate_reference_binding(references, switched)

    def test_incomplete_roster_rejected(self):
        rows = [{"model": m, "index": 0.5} for m in report.MODELS[1:]]
        with self.assertRaises(ValueError):
            report.rank_rows(rows)


if __name__ == "__main__":
    unittest.main()
