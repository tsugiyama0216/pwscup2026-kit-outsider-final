"""Derived series identity regressions; no model execution or scoring truth."""
import copy
import unittest

from rank_validation import provenance as p

PREFIX = "main_round/mock_test/rank_validation/"
MODEL = "main_round/anonymization/team10/attack/attackR/rv6_series/model.py"


def h(letter):
    return letter * 64


def frozen(batch=True):
    value = {
        "model_id": "Rv2", "input_files_unchanged": True,
        "adapter_sha256": h("a"), "worker_sha256": h("b"),
        "diagnostics": {"model_id": "Rv2", "blocked_reads": [],
                        "frozen_gates": {"selection_sha256": h("c"), "count": 12}},
    }
    if batch:
        value.update(shared_batch=True, batch_dependency_sha256={
            "batch_worker.py": h("b"), "model_worker.py": h("d"), "adapters.py": h("a")})
    return value


def derived(name="Rv6", batch=True):
    return {
        "model_id": name, "input_files_unchanged": True, "derived_series_batch": True,
        "adapter_sha256": h("a"), "worker_sha256": h("b"),
        "series_source_sha256": {PREFIX + "series_worker.py": h("b"),
                                 PREFIX + "adapters.py": h("a"), MODEL: h("c")},
        "series_protocol_sha256": h("d"), "series_dependency_manifest_sha256": h("e"),
        "baseline_model_signature": p.model_signature(frozen(batch)),
        "baseline_output_sha256": {"F_mia.csv": h("f"), "F_aia.csv": h("1")},
        "baseline_diagnostics_sha256": h("2"),
        "diagnostics": {"model_id": name, "blocked_reads": [],
                        "imported_source_sha256": {MODEL: h("c")},
                        "frozen_gates": {"series_verified_files": 126,
                                         "manifest_sha256": h("e"), "protocol_sha256": h("d")}},
    }


def resign(value):
    value["signature_sha256"] = p.canonical_sha256(
        {k: v for k, v in value.items() if k != "signature_sha256"})


class SeriesProvenanceTests(unittest.TestCase):
    def test_all_six_models_have_distinct_fixed_identities(self):
        signatures = [p.model_signature(derived(f"Rv{i}")) for i in range(6, 12)]
        self.assertEqual({s["kind"] for s in signatures}, {"derived_series_batch"})
        self.assertEqual(len({s["signature_sha256"] for s in signatures}), 6)

    def test_both_recorded_rv2_worker_kinds_are_accepted_but_not_equal(self):
        self.assertNotEqual(p.model_signature(derived(batch=True)), p.model_signature(derived(batch=False)))

    def test_world_baseline_bytes_and_timings_are_evidence_not_model_identity(self):
        a = derived()
        b = copy.deepcopy(a)
        b.update(baseline_output_sha256={"F_mia.csv": h("3"), "F_aia.csv": h("4")},
                 baseline_diagnostics_sha256=h("5"), seconds=999, pid=123,
                 public_input_sha256={"B.csv": h("6")}, seed=2026100962)
        b["diagnostics"].update(seconds=777, known_negative_proof={"count": 163})
        self.assertEqual(p.model_signature(a), p.model_signature(b))

    def test_numeric_source_change_changes_identity(self):
        a = derived(); b = copy.deepcopy(a)
        b["series_source_sha256"][MODEL] = h("6")
        b["diagnostics"]["imported_source_sha256"][MODEL] = h("6")
        self.assertNotEqual(p.model_signature(a), p.model_signature(b))

    def test_protocol_and_dependency_manifest_changes_change_identity(self):
        original = derived()
        for field, gate_field in (("series_protocol_sha256", "protocol_sha256"),
                                  ("series_dependency_manifest_sha256", "manifest_sha256")):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = h("6")
                changed["diagnostics"]["frozen_gates"][gate_field] = h("6")
                self.assertNotEqual(p.model_signature(original), p.model_signature(changed))

    def test_baseline_model_change_changes_derived_identity(self):
        a = derived(); b = copy.deepcopy(a)
        old = frozen(); old["diagnostics"]["frozen_gates"]["selection_sha256"] = h("6")
        b["baseline_model_signature"] = p.model_signature(old)
        self.assertNotEqual(p.model_signature(a), p.model_signature(b))

    def test_baseline_signature_cannot_be_tampered_without_resigning(self):
        value = derived()
        value["baseline_model_signature"]["frozen_gates"]["count"] = 999
        with self.assertRaises(ValueError):
            p.model_signature(value)

    def test_incomplete_but_self_signed_baseline_is_rejected(self):
        for kind in ("frozen_batch", "frozen_individual"):
            with self.subTest(kind=kind):
                value = derived()
                baseline = {"format": "panel-model-signature-v1", "model_id": "Rv2", "kind": kind}
                resign(baseline)
                value["baseline_model_signature"] = baseline
                with self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_resigned_baseline_with_broken_source_hashes_is_rejected(self):
        for change in (
            lambda b: b["inference_sources"].update(adapter_sha256="bad"),
            lambda b: b["batch_dependency_sha256"].update({"batch_worker.py": h("7")}),
            lambda b: b.update(frozen_gates={}),
        ):
            value = derived(); baseline = value["baseline_model_signature"]
            change(baseline); resign(baseline)
            with self.assertRaises(ValueError):
                p.model_signature(value)

    def test_wrong_baseline_model_or_kind_is_rejected_even_when_resigned(self):
        for field, replacement in (("model_id", "Rv1"), ("model_id", "rv2"),
                                   ("kind", "plugin"), ("kind", "derived_series_batch")):
            with self.subTest(field=field, replacement=replacement):
                value = derived(); value["baseline_model_signature"][field] = replacement
                resign(value["baseline_model_signature"])
                with self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_both_baseline_output_hashes_and_diagnostics_hash_are_required(self):
        mutations = [lambda d: d.pop("baseline_output_sha256"),
                     lambda d: d.pop("baseline_diagnostics_sha256"),
                     lambda d: d.update(baseline_diagnostics_sha256=""),
                     lambda d: d["baseline_output_sha256"].pop("F_aia.csv"),
                     lambda d: d["baseline_output_sha256"].update({"F_mia.csv": "bad"}),
                     lambda d: d["baseline_output_sha256"].update({"extra.csv": h("3")})]
        for change in mutations:
            value = derived(); change(value)
            with self.assertRaises(ValueError):
                p.model_signature(value)

    def test_plugin_cannot_impersonate_any_series_model_in_any_case(self):
        for name in [f"Rv{i}" for i in range(6, 12)] + ["rv6", "RV11", "rV2"]:
            with self.subTest(name=name):
                value = derived(name); value.pop("derived_series_batch")
                value["diagnostics"]["trusted_plugin"] = True
                value["plugin_runtime_sha256"] = h("c")
                value["diagnostics"].update(plugin_sha256=h("d"), fitted_assets_sha256={})
                with self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_mixed_identity_kinds_are_rejected(self):
        for field in ("shared_batch", "trusted_plugin"):
            with self.subTest(field=field):
                value = derived()
                if field == "trusted_plugin": value["diagnostics"][field] = True
                else: value[field] = True
                with self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_unknown_or_case_changed_series_name_is_rejected(self):
        for name in ("Rv5", "Rv12", "rv6", "Candidate", "Rv2"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.model_signature(derived(name))

    def test_worker_source_and_gate_evidence_must_agree(self):
        changes = [lambda d: d.update(worker_sha256=h("6")),
                   lambda d: d.update(adapter_sha256=h("6")),
                   lambda d: d["diagnostics"]["frozen_gates"].update(protocol_sha256=h("6")),
                   lambda d: d["diagnostics"]["frozen_gates"].update(manifest_sha256=h("6")),
                   lambda d: d["diagnostics"]["frozen_gates"].update(series_verified_files=0),
                   lambda d: d["diagnostics"]["frozen_gates"].update(series_verified_files=True)]
        for change in changes:
            value = derived(); change(value)
            with self.assertRaises(ValueError):
                p.model_signature(value)

    def test_source_paths_and_dependency_capture_cannot_be_missing_or_escape(self):
        for field in ("series_source_sha256", "imported_source_sha256"):
            value = derived()
            location = value if field == "series_source_sha256" else value["diagnostics"]
            location.pop(field)
            with self.assertRaises(ValueError):
                p.model_signature(value)
        value = derived(); value["series_source_sha256"]["../other/model.py"] = h("6")
        with self.assertRaises(ValueError):
            p.model_signature(value)

    def test_preservation_and_no_forbidden_reads_are_required(self):
        for change in (lambda d: d.update(input_files_unchanged=False),
                       lambda d: d["diagnostics"].update(blocked_reads=["scoring_truth/roles.json"])):
            value = derived(); change(value)
            with self.assertRaises(ValueError):
                p.model_signature(value)

    def test_returned_identity_is_detached_from_mutable_worker_evidence(self):
        value = derived(); identity = p.model_signature(value)
        previous = copy.deepcopy(identity)
        value["baseline_model_signature"]["frozen_gates"]["count"] = 999
        value["diagnostics"]["frozen_gates"]["series_verified_files"] = 999
        self.assertEqual(identity, previous)
        self.assertEqual(identity["signature_sha256"], p.canonical_sha256(
            {k: v for k, v in identity.items() if k != "signature_sha256"}))


if __name__ == "__main__":
    unittest.main()
