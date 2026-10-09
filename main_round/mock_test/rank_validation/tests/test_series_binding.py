"""Same-panel Rv2 binding checks without attack inference or scoring truth."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rank_validation import provenance, series_binding as binding
from rank_validation.tests.test_series_provenance import derived, frozen, h


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SeriesBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "predictions" / "Rv2"
        self.baseline.mkdir(parents=True)
        (self.baseline / "F_mia.csv").write_bytes(b"record_id,0\n1,0.2\n")
        (self.baseline / "F_aia.csv").write_bytes(b"target_k,challenge_row_id,time_hat\n0,1,2.5\n")
        self.base = frozen()
        self.base.update(public_input_sha256={"B.csv": h("1"), "C_0.csv": h("2")},
                         output_sha256={n: sha(self.baseline / n) for n in binding.FILES[:2]})
        self.write_baseline()
        self.child = derived()
        self.child.update(baseline_files_unchanged=True,
                          public_input_sha256=copy.deepcopy(self.base["public_input_sha256"]),
                          baseline_model_signature=provenance.model_signature(self.base),
                          baseline_output_sha256=copy.deepcopy(self.base["output_sha256"]),
                          baseline_diagnostics_sha256=sha(self.baseline / "diagnostics.json"))

    def write_baseline(self):
        (self.baseline / "diagnostics.json").write_text(json.dumps(self.base), encoding="utf-8")

    def bind_current_diagnostic_hash(self):
        self.child["baseline_diagnostics_sha256"] = sha(self.baseline / "diagnostics.json")

    def test_exact_same_world_baseline_is_accepted(self):
        result = binding.validate_baseline_binding(self.baseline, self.child)
        self.assertEqual(result["baseline_model_signature"], provenance.model_signature(self.base))
        self.assertEqual(result["baseline_output_sha256"], self.base["output_sha256"])

    def test_different_worlds_have_equal_model_identity_but_cannot_cross_bind(self):
        old_signature = provenance.model_signature(self.child)
        changed = copy.deepcopy(self.child)
        changed["public_input_sha256"]["B.csv"] = h("3")
        changed["baseline_output_sha256"]["F_mia.csv"] = h("4")
        changed["baseline_diagnostics_sha256"] = h("5")
        self.assertEqual(old_signature, provenance.model_signature(changed))
        with self.assertRaises(ValueError):
            binding.validate_baseline_binding(self.baseline, changed)

    def test_public_world_mismatch_rejected_with_other_binding_fields_valid(self):
        self.child["public_input_sha256"]["B.csv"] = h("3")
        with self.assertRaisesRegex(ValueError, "different public worlds"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_baseline_identity_change_rejected_even_after_record_hash_is_updated(self):
        self.base["diagnostics"]["frozen_gates"]["selection_sha256"] = h("3")
        self.write_baseline(); self.bind_current_diagnostic_hash()
        with self.assertRaisesRegex(ValueError, "model identity differs"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_baseline_diagnostic_timing_change_is_still_detected(self):
        old = provenance.model_signature(self.base)
        self.base["seconds"] = 100
        self.write_baseline()
        self.assertEqual(old, provenance.model_signature(self.base))
        with self.assertRaisesRegex(ValueError, "diagnostics differ"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_real_csv_bytes_must_match_both_records(self):
        # Identical numeric values with different formatting still change bytes.
        (self.baseline / "F_mia.csv").write_bytes(b"record_id,0\n1,0.2000\n")
        with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_different_prediction_hash_records_rejected_before_using_scores(self):
        self.base["output_sha256"]["F_aia.csv"] = h("4")
        self.write_baseline(); self.bind_current_diagnostic_hash()
        with self.assertRaisesRegex(ValueError, "hash records differ"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_matching_but_false_hash_records_do_not_replace_actual_csv_hashes(self):
        self.base["output_sha256"]["F_aia.csv"] = h("4")
        self.child["baseline_output_sha256"]["F_aia.csv"] = h("4")
        self.write_baseline(); self.bind_current_diagnostic_hash()
        with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_optional_stored_baseline_signature_must_be_consistent(self):
        self.base["model_signature"] = {"signature_sha256": h("4")}
        self.write_baseline(); self.bind_current_diagnostic_hash()
        with self.assertRaisesRegex(ValueError, "stored model signature"):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_missing_public_evidence_and_preservation_flag_rejected(self):
        for name in ("public_input_sha256", "baseline_files_unchanged"):
            with self.subTest(name=name):
                value = copy.deepcopy(self.child); value.pop(name)
                with self.assertRaises(ValueError):
                    binding.validate_baseline_binding(self.baseline, value)

    def test_wrong_baseline_name_and_forbidden_reads_rejected(self):
        for change in (lambda d: d.update(model_id="Rv1"),
                       lambda d: d["diagnostics"].update(blocked_reads=["truth.csv"])):
            with self.subTest(change=change):
                base = copy.deepcopy(self.base); change(base)
                (self.baseline / "diagnostics.json").write_text(json.dumps(base), encoding="utf-8")
                self.bind_current_diagnostic_hash()
                with self.assertRaises(ValueError):
                    binding.validate_baseline_binding(self.baseline, self.child)

    def test_missing_or_extra_baseline_files_are_rejected(self):
        extra = self.baseline / "unexpected.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            binding.validate_baseline_binding(self.baseline, self.child)
        extra.unlink()
        (self.baseline / "F_aia.csv").unlink()
        with self.assertRaises(ValueError):
            binding.validate_baseline_binding(self.baseline, self.child)

    def test_no_metadata_truth_or_score_file_is_read(self):
        original = Path.read_bytes
        reads = []
        allowed = {(self.baseline / n).resolve() for n in binding.FILES}
        def restricted(path):
            reads.append(path.resolve())
            if path.resolve() not in allowed:
                raise AssertionError("Unexpected read: " + str(path))
            return original(path)
        with patch.object(Path, "read_bytes", restricted):
            binding.validate_baseline_binding(self.baseline, self.child)
        self.assertEqual(set(reads), allowed)

    def test_change_during_verification_is_detected(self):
        original = binding._sha
        changed = False
        def mutate_after_hash(path):
            nonlocal changed
            value = original(path)
            if path.name == "F_mia.csv" and not changed:
                changed = True
                path.write_bytes(path.read_bytes() + b"\n")
            return value
        with patch.object(binding, "_sha", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(ValueError, "changed during binding"):
                binding.validate_baseline_binding(self.baseline, self.child)

    def candidate_files(self):
        directory = self.root / "predictions" / "Rv6"
        directory.mkdir()
        for name in binding.FILES[:2]:
            (directory / name).write_bytes((self.baseline / name).read_bytes())
        self.child["output_sha256"] = {n: sha(directory / n) for n in binding.FILES[:2]}
        (directory / "diagnostics.json").write_text(json.dumps(self.child), encoding="utf-8")
        return directory

    def test_panel_rejects_baseline_change_before_scoring(self):
        from rank_validation import panel
        self.candidate_files()
        (self.baseline / "F_mia.csv").write_bytes(b"changed")
        state = {"public_sha256": self.child["public_input_sha256"], "config": {}}
        with patch.object(panel, "check_panel", return_value=(self.root, state, {})), \
             patch.object(panel, "evaluation_signature", return_value={}), \
             patch.object(panel, "score_world") as scorer:
            with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
                panel.score(self.root, "Rv6")
            scorer.assert_not_called()
        self.assertFalse((self.root / "scores" / "Rv6").exists())

    def test_panel_rechecks_baseline_after_scoring_before_saving(self):
        from rank_validation import panel
        import pandas as pd
        self.candidate_files()
        state = {"public_sha256": self.child["public_input_sha256"], "config": {}, "case_metadata": {}}
        def score_and_mutate(*args):
            (self.baseline / "F_aia.csv").write_bytes(b"changed during scoring")
            return {}
        with patch.object(panel, "check_panel", return_value=(self.root, state, {})), \
             patch.object(panel, "evaluation_signature", return_value={}), \
             patch.object(panel, "validate_predictions"), \
             patch.object(panel, "score_world", side_effect=score_and_mutate) as scorer, \
             patch.object(panel, "per_target_diagnostics", return_value=pd.DataFrame()):
            with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
                panel.score(self.root, "Rv6")
            scorer.assert_called_once()
        self.assertFalse((self.root / "scores" / "Rv6").exists())

    def test_series_report_rechecks_binding_after_collect(self):
        from rank_validation import series_report
        self.candidate_files()
        (self.baseline / "F_mia.csv").write_bytes(b"changed after prior scoring")
        comparison = {"inputs": [{"panel": str(self.root)}]}
        with patch.object(series_report, "validate_references", return_value=({}, None)), \
             patch.object(series_report, "collect", return_value=comparison) as collect:
            with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
                series_report.build_report([self.root])
        self.assertEqual(collect.call_count, 6)


if __name__ == "__main__":
    unittest.main()
