"""Additional frozen identities, with unchanged historical signature regressions."""
import copy
import json
from pathlib import Path
import unittest

from rank_validation import provenance as p
from rank_validation.tests.test_provenance import plugin
from rank_validation.tests.test_series_provenance import derived, frozen

PREFIX = "main_round/mock_test/rank_validation/"
MAIN = {
    "Rv3": "main_round/anonymization/team10/atack/Rv3/model.py",
    "Mv1": "main_round/attack/Attack_M/v1/model.py",
    "Mv2": "main_round/attack/Attack_M/v2/model.py",
}
ASSET = "main_round/attack/Attack_M/v1/fit.json"
OLD_SIGNATURES = {
    "Rv1": "662832de3f7edf42cb4afa0a429e458bb6e86ced2c14e57209d8c39d6041c8e1",
    "Rv2": "ae3426e0ae686df98fb3f5a323a5071a2af651fe3f68900a6448ca1ad7f581a0",
    "pl1": "4df1466f60f62e7964dd3a7ca4f54600fa924d9662e8419714d7dcdb5b1a41ac",
    "Tv105": "4f9cb157180b032e1b07021616fa8f517ccdec07c4599d2f5bd0ef31e02999ad",
    "Rv1_batch": "80857f34468548efeb4fe93bd36b9ff23da0bb768c1aa735a58b1b35ced7918a",
    "Rv2_batch": "85c0289f7628f49c89c7a945f9218033f4029e13ba30565749cb21c2805d15de",
    "Tv105_batch": "22d25cadbe7084528cf4e5a4e4ec4380f7a1c63ddc5281c0f693ba2ed0a6a825",
    "Rv6": "da0c8358898ee5d20dcf46a3013df31b3f72c57d247b7f5647f324e7fb3a976e",
    "Rv7": "d6fead5324cd9cf65a88b50524fa009c4e1ce1c4c23a0c5d096216fa6570e578",
    "Rv8": "a9aaf450ee22c3f2767964044462b3a60231c0e704d7cb788c49c24b759fccc6",
    "Rv9": "2582545fc53682a2c44ee95748ad02c2024c42a0bdab4e11142d661d24c04475",
    "Rv10": "b270c867b273b281475a3a5b23567c4d51dc2c3f257daa1f7fcdb06f37c4da85",
    "Rv11": "dc3114be217193960149f666c80e9e6e1dfa4f16acbb564678a3c2e3ca3d6129",
    "plugin": "0323221b8b1c72d2f3206543dae34f7a2106274b58a4db5c94670104e6fa25dd",
}


def h(letter):
    return letter * 64


def additional(name="Mv1"):
    adapter = "additional_r.py" if name == "Rv3" else "additional_m.py"
    fixed = {MAIN[name]: h("c"), ASSET: h("d")}
    return {
        "model_id": name, "input_files_unchanged": True, "additional_frozen": True,
        "adapter_sha256": h("a"), "worker_sha256": h("b"),
        "additional_source_sha256": {PREFIX + adapter: h("a"),
                                     PREFIX + "additional_worker.py": h("b"), **fixed},
        "diagnostics": {"model_id": name, "blocked_reads": [],
                        "imported_source_sha256": {MAIN[name]: h("c"), PREFIX + adapter: h("a")},
                        "imported_sources_unchanged": True,
                        "frozen_gates": {"verified_files": len(fixed), "files_sha256": dict(fixed)}},
    }


def recorded_diagnostic(identity):
    """Reconstruct only the documented recorded worker fields, without reading current sources."""
    kind = identity["kind"]
    d = {"model_id": identity["model_id"], "input_files_unchanged": True,
         **identity["inference_sources"],
         "diagnostics": {"model_id": identity["model_id"], "blocked_reads": [],
                         "frozen_gates": copy.deepcopy(identity["frozen_gates"])}}
    if identity["imported_dependency_capture"] == "recorded":
        d["diagnostics"]["imported_source_sha256"] = copy.deepcopy(identity["imported_source_sha256"])
    if kind == "frozen_batch":
        d.update(shared_batch=True, batch_dependency_sha256=copy.deepcopy(identity["batch_dependency_sha256"]))
    elif kind == "derived_series_batch":
        d["derived_series_batch"] = True
        for field in ("series_source_sha256", "series_protocol_sha256", "series_dependency_manifest_sha256",
                      "baseline_model_signature"):
            d[field] = copy.deepcopy(identity[field])
        # Per-world prediction hashes do not contribute to the model identity.
        d.update(baseline_output_sha256={"F_mia.csv": h("e"), "F_aia.csv": h("f")},
                 baseline_diagnostics_sha256=h("1"))
    elif kind != "frozen_individual":
        raise AssertionError("Unexpected historical identity kind")
    return d


def saved_signatures(value):
    if isinstance(value, dict):
        if value.get("format") == "panel-model-signature-v1":
            yield value
        for item in value.values():
            yield from saved_signatures(item)
    elif isinstance(value, list):
        for item in value:
            yield from saved_signatures(item)


class AdditionalProvenanceTests(unittest.TestCase):
    def test_three_named_frozen_models_have_distinct_identities(self):
        identities = [p.model_signature(additional(name)) for name in MAIN]
        self.assertEqual({d["kind"] for d in identities}, {"frozen_additional"})
        self.assertEqual(len({d["signature_sha256"] for d in identities}), 3)

    def test_old_four_six_and_plugin_synthetic_signatures_are_byte_stable(self):
        values = {}
        for name in ("Rv1", "Rv2", "pl1", "Tv105"):
            d = frozen(False); d["model_id"] = name; d["diagnostics"]["model_id"] = name
            values[name] = p.model_signature(d)["signature_sha256"]
        for name in ("Rv1", "Rv2", "Tv105"):
            d = frozen(True); d["model_id"] = name; d["diagnostics"]["model_id"] = name
            values[name + "_batch"] = p.model_signature(d)["signature_sha256"]
        for i in range(6, 12):
            values[f"Rv{i}"] = p.model_signature(derived(f"Rv{i}"))["signature_sha256"]
        values["plugin"] = p.model_signature(plugin())["signature_sha256"]
        self.assertEqual(values, OLD_SIGNATURES)

    def test_actual_archived_four_and_six_model_signatures_remain_identical(self):
        root = Path(__file__).resolve().parents[1] / "evidence"
        paths = [root / "publication/v5-final/result.json", root / "rv6_series_20261009/result.json"]
        identities = {}
        for path in paths:
            for identity in saved_signatures(json.loads(path.read_text(encoding="utf-8"))):
                identities[identity["signature_sha256"]] = identity
        self.assertEqual({d["model_id"] for d in identities.values()},
                         {"Rv1", "Rv2", "pl1", "Tv105", *[f"Rv{i}" for i in range(6, 12)]})
        for identity in identities.values():
            with self.subTest(model=identity["model_id"], kind=identity["kind"]):
                self.assertEqual(p.model_signature(recorded_diagnostic(identity)), identity)

    def test_world_inputs_routes_timings_do_not_change_identity(self):
        a = additional(); b = copy.deepcopy(a)
        b.update(pid=12, seconds=99, public_input_sha256={"B.csv": h("2")},
                 output_sha256={"F_mia.csv": h("3"), "F_aia.csv": h("4")})
        b["diagnostics"].update(numeric_diagnostics={"route": "different"})
        self.assertEqual(p.model_signature(a), p.model_signature(b))

    def test_each_frozen_dependency_and_import_change_changes_identity(self):
        original = additional()
        changed = copy.deepcopy(original)
        for where in (changed["additional_source_sha256"], changed["diagnostics"]["frozen_gates"]["files_sha256"]):
            where[ASSET] = h("e")
        self.assertNotEqual(p.model_signature(original), p.model_signature(changed))
        changed = copy.deepcopy(original)
        changed["diagnostics"]["imported_source_sha256"]["main_round/attack/extra_helper.py"] = h("e")
        self.assertNotEqual(p.model_signature(original), p.model_signature(changed))

    def test_worker_and_chosen_adapter_hashes_must_match(self):
        for name in MAIN:
            for change in (lambda d: d.update(worker_sha256=h("e")),
                           lambda d: d.update(adapter_sha256=h("e")),
                           lambda d: d["additional_source_sha256"].pop(PREFIX + "additional_worker.py"),
                           lambda d: d["additional_source_sha256"].pop(PREFIX + ("additional_r.py" if name == "Rv3" else "additional_m.py"))):
                value = additional(name); change(value)
                with self.subTest(model=name), self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_gate_needs_complete_hash_map_and_named_canonical_source(self):
        changes = [lambda d: d["diagnostics"].update(frozen_gates={}),
                   lambda d: d["diagnostics"]["frozen_gates"].pop("files_sha256"),
                   lambda d: d["diagnostics"]["frozen_gates"].update(files_sha256={}),
                   lambda d: d["diagnostics"]["frozen_gates"]["files_sha256"].pop(MAIN["Mv1"]),
                   lambda d: d["additional_source_sha256"].pop(ASSET),
                   lambda d: d["additional_source_sha256"].update({ASSET: h("e")}),
                   lambda d: d["diagnostics"]["frozen_gates"].update(count=float("nan"))]
        for change in changes:
            value = additional(); change(value)
            with self.assertRaises(ValueError): p.model_signature(value)

    def test_imports_are_required_preserved_and_consistent_with_fixed_sources(self):
        changes = [lambda d: d["diagnostics"].pop("imported_source_sha256"),
                   lambda d: d["diagnostics"].update(imported_source_sha256={}),
                   lambda d: d["diagnostics"].pop("imported_sources_unchanged"),
                   lambda d: d["diagnostics"].update(imported_sources_unchanged=False),
                   lambda d: d["diagnostics"]["imported_source_sha256"].update({MAIN["Mv1"]: h("e")})]
        for change in changes:
            value = additional(); change(value)
            with self.assertRaises(ValueError): p.model_signature(value)

    def test_all_source_maps_reject_unsafe_paths_or_invalid_digests(self):
        for label in ("additional", "gate", "imports"):
            for name, digest in (("../other.py", h("e")), ("C:/other.py", h("e")),
                                 ("model/token.txt", h("e")), ("model.py", "bad")):
                value = additional()
                where = {"additional": value["additional_source_sha256"],
                         "gate": value["diagnostics"]["frozen_gates"]["files_sha256"],
                         "imports": value["diagnostics"]["imported_source_sha256"]}[label]
                where[name] = digest
                with self.subTest(label=label, path=name), self.assertRaises(ValueError):
                    p.model_signature(value)

    def test_additional_kind_cannot_mix_with_any_previous_kind(self):
        for field in ("shared_batch", "derived_series_batch", "trusted_plugin"):
            value = additional()
            (value["diagnostics"] if field == "trusted_plugin" else value)[field] = True
            with self.assertRaises(ValueError): p.model_signature(value)

    def test_unknown_or_case_changed_model_is_rejected(self):
        for name in ("rv3", "MV1", "Mv3", "Rv1", "Rv6", "Candidate"):
            value = additional(); value["model_id"] = name; value["diagnostics"]["model_id"] = name
            with self.subTest(name=name), self.assertRaises(ValueError): p.model_signature(value)

    def test_plugin_cannot_impersonate_new_frozen_names_in_any_case(self):
        for name in ("Rv3", "rv3", "MV1", "mV1", "Mv2", "mv2"):
            value = plugin(); value["model_id"] = name; value["diagnostics"]["model_id"] = name
            with self.subTest(name=name), self.assertRaises(ValueError): p.model_signature(value)

    def test_input_preservation_identity_and_forbidden_reads_are_checked(self):
        for change in (lambda d: d.update(input_files_unchanged=False),
                       lambda d: d["diagnostics"].update(model_id="Mv2"),
                       lambda d: d["diagnostics"].update(blocked_reads=["scoring_truth/roles.json"])):
            value = additional(); change(value)
            with self.assertRaises(ValueError): p.model_signature(value)

    def test_result_is_detached_and_contains_complete_gate_map(self):
        value = additional(); identity = p.model_signature(value); old = copy.deepcopy(identity)
        value["diagnostics"]["frozen_gates"]["files_sha256"][ASSET] = h("e")
        value["additional_source_sha256"][ASSET] = h("e")
        self.assertEqual(identity, old)
        self.assertEqual(identity["frozen_gates"]["files_sha256"][ASSET], h("d"))
        self.assertEqual(identity["signature_sha256"], p.canonical_sha256(
            {key: item for key, item in identity.items() if key != "signature_sha256"}))


if __name__ == "__main__":
    unittest.main()
