"""M adapter boundaries and exact frozen mask/joint operations; no full inference."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rank_validation import additional_m as m
from rank_validation.tests.test_adapters import public_world


class AdditionalMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = m._modules("Mv2")

    def setUp(self):
        self.public = public_world()
        self.mask = np.zeros(1049, dtype=bool)
        self.mask[:163] = True
        self.actual = SimpleNamespace(
            derive_private_mask=Mock(return_value=(self.mask.copy(), {"source": "supplied public tables"})),
            apply_private_mask=self.modules["M.v1.actual"].apply_private_mask)

    def run_predict(self, name, selected, modules, public=None):
        with redirect_stdout(io.StringIO()):
            return m._predict(name, self.public if public is None else public, selected, modules)

    def test_prepare_frozen_gates_and_source_closure_without_prediction(self):
        for name, expected in (("Mv1", {"variant": "full_reference_aia"}),
                               ("Mv2", {"mia_strength": 1., "aia_variant": "guarded40"})):
            with self.subTest(model=name):
                predict, gate, sources = m.prepare(name)
                first = gate()
                self.assertTrue(callable(predict))
                self.assertEqual(first, gate())
                self.assertEqual(first["selection"], expected)
                self.assertEqual(first["M_v1_preservation"]["verified_files"], 65)
                self.assertEqual(first["files_sha256"], sources)
                version = "v1" if name == "Mv1" else "v2"
                self.assertIn(f"main_round/attack/Attack_M/{version}/model.py", sources)
                self.assertIn(f"main_round/attack/M/{version}/model.py", sources)
                self.assertFalse(any(Path(p).name.lower() == "token.txt" for p in sources))
                if name == "Mv2": self.assertEqual(first["v2_dependency_files"], 31)

    def test_unknown_model_is_rejected_before_loading(self):
        with patch.object(m, "_modules", side_effect=AssertionError("must not import")):
            with self.assertRaises(ValueError): m.prepare("M1")

    def test_wrong_compatibility_import_origin_is_rejected(self):
        with patch.object(m.adapters, "_bootstrap"), \
             patch.object(m.importlib, "import_module", return_value=SimpleNamespace(__file__=__file__)):
            with self.assertRaisesRegex(ImportError, "origin"):
                m._modules("Mv1")

    def test_canonical_and_compatibility_source_mismatch_is_rejected(self):
        original = m.adapters.sha
        target = m.CANONICAL / "v1/model.py"
        def changed(path):
            return "0" * 64 if Path(path) == target else original(path)
        with patch.object(m.adapters, "sha", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Canonical and compatibility"):
                m._source_hashes(self.modules)

    def test_selection_and_final_check_cannot_be_replaced(self):
        original = m._json
        def bad_variant(path):
            value = original(path)
            if Path(path) == m.CANONICAL / "v1/selection.json":
                value["selected_variant"] = "official"
            return value
        with patch.object(m, "_json", side_effect=bad_variant), self.assertRaisesRegex(ValueError, "v1 selection"):
            m._gate("Mv1", self.modules)
        def unfinished(path):
            value = original(path)
            if Path(path) == m.CANONICAL / "local-output/v2/final_check.json": value["completed"] = False
            return value
        with patch.object(m, "_json", side_effect=unfinished), self.assertRaisesRegex(ValueError, "final check"):
            m._gate("Mv2", self.modules)

    def test_m_v1_calls_selected_variant_and_exact_affine_private_mask(self):
        raw = np.linspace(0., 1., 1049)
        raw[200:203] = .5
        numeric = Mock(return_value={"mia": raw.copy(), "aia": np.linspace(0., 10., 326)})
        modules = {"M.v1.actual": self.actual, "M.v1.model": SimpleNamespace(predict=numeric)}
        result = self.run_predict("Mv1", {"variant": "full_reference_aia"}, modules)
        self.assertEqual(numeric.call_count, 22)
        self.assertEqual(self.actual.derive_private_mask.call_count, 1)
        expected = self.actual.apply_private_mask(raw, self.mask)
        for k in m.adapters.TARGETS:
            np.testing.assert_array_equal(result["F_mia"][str(k)], expected)
        self.assertNotEqual(float(expected[200]), float(raw[200]))
        self.assertEqual(expected[200], expected[201])
        for call in numeric.call_args_list:
            self.assertEqual(set(call.kwargs), {"c", "background", "candidates", "challenge", "cfg", "variant"})
            self.assertEqual(call.kwargs["variant"], "full_reference_aia")
        self.assertEqual(result["F_aia"].shape, (7172, 3))

    def test_m_v2_one_joint_call_preserves_cross_target_median_and_single_mask(self):
        numeric = self.modules["M.v2.model"]
        mia = self.modules["M.v2.mia"]
        aia = self.modules["M.v2.aia"]
        rng = np.random.default_rng(7341)
        matrix = rng.integers(-4, 5, size=(1049, 22)).astype(float)
        # Strong per-person offsets make a target-wise adapter observably wrong.
        matrix += np.arange(1049)[:, None] / 10.
        expected = mia.adjust_scores(matrix, private_mask=self.mask, strength=1.)
        targetwise = mia.adjust_scores(matrix, private_mask=self.mask, strength=0.)
        self.assertFalse(np.array_equal(expected, targetwise))
        modules = {"M.v1.actual": self.actual, "M.v2.model": numeric}
        with patch.object(mia, "raw_scores", side_effect=[matrix[:, j] for j in range(22)]) as raw, \
             patch.object(aia, "predict", return_value=np.linspace(0., 10., 326)) as attribute, \
             patch.object(numeric, "predict_all", wraps=numeric.predict_all) as joint:
            result = self.run_predict("Mv2", {"mia_strength": 1., "aia_variant": "guarded40"}, modules)
        joint.assert_called_once()
        self.assertEqual(raw.call_count, 22)
        self.assertEqual(attribute.call_count, 22)
        self.assertEqual(set(joint.call_args.kwargs["targets"]), set(m.adapters.TARGETS))
        np.testing.assert_array_equal(result["F_mia"].drop(columns="record_id"), expected)
        for call in attribute.call_args_list: self.assertEqual(call.kwargs["variant"], "guarded40")

    def test_joint_model_wrong_order_is_rejected(self):
        numeric = Mock(return_value={"target_order": list(reversed(m.adapters.TARGETS))})
        modules = {"M.v1.actual": self.actual, "M.v2.model": SimpleNamespace(predict_all=numeric)}
        with self.assertRaisesRegex(ValueError, "joint target order"):
            self.run_predict("Mv2", {"mia_strength": 1., "aia_variant": "guarded40"}, modules)

    def test_hidden_arguments_partial_world_and_invalid_ids_are_rejected(self):
        for kind in ("hidden", "partial", "id"):
            with self.subTest(kind=kind):
                public = copy.deepcopy(self.public)
                if kind == "hidden": public["truth_path"] = "never read"
                elif kind == "partial": public["targets"].pop(0)
                else: public["candidates"].loc[1, "record_id"] = 0
                with self.assertRaises(ValueError):
                    self.run_predict("Mv1", {"variant": "full_reference_aia"}, {}, public)

    def test_bad_public_mask_proof_is_not_replaced_with_all_false(self):
        self.actual.derive_private_mask.side_effect = ValueError("public proof does not hold")
        with self.assertRaisesRegex(ValueError, "public proof"):
            self.run_predict("Mv1", {"variant": "full_reference_aia"}, {"M.v1.actual": self.actual})

    def test_original_public_tables_are_not_mutated(self):
        before = self.public["background"].copy(deep=True)
        def numeric(**kwargs):
            kwargs["background"].iloc[0, 0] = -9999
            return {"mia": np.full(1049, .5), "aia": np.ones(326)}
        modules = {"M.v1.actual": self.actual, "M.v1.model": SimpleNamespace(predict=numeric)}
        self.run_predict("Mv1", {"variant": "full_reference_aia"}, modules)
        pd.testing.assert_frame_equal(self.public["background"], before)


if __name__ == "__main__":
    unittest.main()
