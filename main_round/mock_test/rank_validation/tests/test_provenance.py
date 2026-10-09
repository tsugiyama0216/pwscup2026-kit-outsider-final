"""Provenance regression tests without model inference or target truth."""
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch
from rank_validation import provenance as p

def h(c): return c*64
def frozen(batch=False):
    d={"model_id":"Rv2","input_files_unchanged":True,"adapter_sha256":h("a"),"worker_sha256":h("b"),
       "diagnostics":{"model_id":"Rv2","frozen_gates":{"selection_sha256":h("c"),"count":12},"blocked_reads":[]}}
    if batch:
        d.update(shared_batch=True,batch_dependency_sha256={"batch_worker.py":h("b"),"model_worker.py":h("d"),"adapters.py":h("a")})
    return d
def plugin():
    return {"model_id":"Candidate","input_files_unchanged":True,"adapter_sha256":h("a"),"worker_sha256":h("b"),"plugin_runtime_sha256":h("c"),
      "diagnostics":{"model_id":"Candidate","trusted_plugin":True,"plugin_sha256":h("d"),"fitted_assets_sha256":{"model/fit.json":h("e")},
                     "imported_source_sha256":{"model/helper.py":h("f")},"blocked_reads":[]}}

class ProvenanceTests(unittest.TestCase):
    def test_canonical_order_but_not_values(self):
        self.assertEqual(p.canonical_sha256({"a":1,"b":2}),p.canonical_sha256({"b":2,"a":1}))
        self.assertNotEqual(p.canonical_sha256({"a":1}),p.canonical_sha256({"a":2}))
    def test_nonfinite_or_key_coercion_rejected(self):
        for v in ({"a":float("nan")},{"a":[float("inf")]},{1:"ambiguous"},{"a":object()}):
            with self.assertRaises(ValueError):p.canonical_sha256(v)
    def test_dynamic_fields_are_not_model_identity(self):
        a=frozen(); b=copy.deepcopy(a)
        b.update(pid=1,seconds=123,public_input_sha256={"B.csv":h("e")})
        b["diagnostics"].update(routes={"0":"different"},seconds=456)
        self.assertEqual(p.model_signature(a),p.model_signature(b))
    def test_every_plugin_dependency_changes_identity(self):
        original=plugin(); expected=p.model_signature(original)
        mutations=[lambda d:d.update(worker_sha256=h("f")),lambda d:d["diagnostics"].update(plugin_sha256=h("e")),
          lambda d:d["diagnostics"]["fitted_assets_sha256"].update({"model/fit.json":h("f")}),
          lambda d:d["diagnostics"]["imported_source_sha256"].update({"model/helper.py":h("e")})]
        for change in mutations:
            d=copy.deepcopy(original);change(d)
            self.assertNotEqual(expected,p.model_signature(d))
    def test_plugin_without_import_capture_fails(self):
        d=plugin();d["diagnostics"].pop("imported_source_sha256")
        with self.assertRaises(ValueError):p.model_signature(d)
    def test_frozen_and_batch_dependencies_change_identity(self):
        d=frozen(True);old=p.model_signature(d);d["batch_dependency_sha256"]["model_worker.py"]=h("e")
        self.assertNotEqual(old,p.model_signature(d))
        d=frozen();old=p.model_signature(d);d["diagnostics"]["frozen_gates"]["selection_sha256"]=h("d")
        self.assertNotEqual(old,p.model_signature(d))
    def test_inconsistent_batch_rejected(self):
        d=frozen(True);d["batch_dependency_sha256"]["batch_worker.py"]=h("e")
        with self.assertRaises(ValueError):p.model_signature(d)
    def test_identity_preservation_hash_and_read_checks(self):
        for change in [lambda d:d["diagnostics"].update(model_id="Rv1"),lambda d:d.update(input_files_unchanged=False),
           lambda d:d.update(worker_sha256="unknown"),lambda d:d["diagnostics"].update(blocked_reads=["truth.csv"]),
           lambda d:d["diagnostics"].update(frozen_gates={})]:
            d=frozen();change(d)
            with self.assertRaises(ValueError):p.model_signature(d)
    def test_source_paths_safe(self):
        for name in ("../fit.json","C:/fit.json","/fit.json","model/../fit.json","token.txt"):
            d=plugin();d["diagnostics"]["fitted_assets_sha256"]={name:h("e")}
            with self.assertRaises(ValueError):p.model_signature(d)
    def test_signature_detached_from_input(self):
        d=frozen();r=p.model_signature(d);d["diagnostics"]["frozen_gates"]["count"]=99
        self.assertEqual(r["frozen_gates"]["count"],12)
    def test_cfg_and_runtime_are_evaluation_identity(self):
        with patch.object(p,"_module_source",return_value={"path":"source.py","sha256":h("a")}),patch.object(p,"_sha",return_value=h("a")):
            with patch.object(p,"_runtime_versions",return_value={"numpy":"2"}):
                a=p.evaluation_signature({"a":1,"b":2})
                self.assertEqual(a,p.evaluation_signature({"b":2,"a":1}))
                self.assertNotEqual(a,p.evaluation_signature({"a":2,"b":2}))
                with self.assertRaises(ValueError):p.evaluation_signature({"a":float("nan")})
            with patch.object(p,"_runtime_versions",return_value={"numpy":"3"}):
                self.assertNotEqual(a,p.evaluation_signature({"a":1,"b":2}))
    def test_actual_import_must_be_main_checkout(self):
        m=types.SimpleNamespace(__file__=str(Path(__file__).resolve()))
        with patch.object(p.importlib,"import_module",return_value=m),self.assertRaises(ValueError):
            p._module_source("pwscup2026.scoring.attack_score",p.OFFICIAL_PACKAGE)
    def test_runtime_pin_mismatch_fails(self):
        with patch.object(p.importlib.metadata,"version",return_value="wrong"),self.assertRaises(ValueError):
            p._runtime_versions()
    def test_all_generator_sources_and_not_seed(self):
        d={key:h("a") for key in p.GENERATOR_KEYS};old=p.generator_signature(d)
        self.assertEqual(old,p.generator_signature({**d,"seed":1}))
        for key in p.GENERATOR_KEYS:self.assertNotEqual(old,p.generator_signature({**d,key:h("b")}))
        d.pop(p.GENERATOR_KEYS[0])
        with self.assertRaises(ValueError):p.generator_signature(d)

if __name__=="__main__":unittest.main()
