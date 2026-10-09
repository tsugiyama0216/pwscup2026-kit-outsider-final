"""Boundary, original gate and real selected-API checks for frozen Rv3."""
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rank_validation import additional_r as rv3
from rank_validation.adapters import sha
from rank_validation.plugins import read_guard
from rank_validation.tests.test_adapters import public_world


@contextmanager
def bound_model(value):
    missing = object()
    previous = sys.modules.get("model", missing)
    sys.modules["model"] = value
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous


class AdditionalRTests(unittest.TestCase):
    def test_unknown_model_is_rejected_before_any_file_read(self):
        with patch.object(rv3, "_bytes", side_effect=AssertionError("unexpected read")):
            with self.assertRaisesRegex(ValueError, "only Rv3"):
                rv3.prepare("Rv2")

    def test_original_hash_gates_and_source_closure(self):
        predict, gate, sources = rv3.prepare("Rv3")
        self.assertTrue(callable(predict))
        result = gate()
        self.assertEqual(result["selection_verified_files"], 5)
        self.assertEqual(result["production_verified_files"], 7)
        self.assertEqual(result["dependency_verified_files"], 48)
        self.assertEqual(result["selection"], {"mia": "full5", "aia": "forest_contrast5_1.0"})
        self.assertGreaterEqual(len(sources), 54)
        for relative, expected in sources.items():
            self.assertEqual(sha(rv3.ROOT / relative), expected)

    def test_changed_frozen_source_is_rejected_without_execution(self):
        with self.assertRaisesRegex(ValueError, "dependency mismatch"):
            rv3._load_source("bad", rv3.BASE / "selected.py", "0" * 64)

    def test_sources_are_audited_and_generic_model_is_restored(self):
        sentinel = types.ModuleType("different_model")
        with bound_model(sentinel):
            with read_guard(rv3.__file__) as audit:
                selected, mask = rv3._load_components()
            self.assertIs(sys.modules["model"], sentinel)
        self.assertTrue(callable(selected))
        self.assertTrue(callable(mask))
        self.assertEqual(audit["blocked"], [])
        for path in (rv3.BASE / "model.py", rv3.BASE / "selected.py", rv3.MASK):
            self.assertEqual(audit["code_hashes"][path], sha(path))

    def test_failed_selected_load_restores_generic_model(self):
        sentinel = types.ModuleType("different_model")
        hashes = {**rv3.NUMERIC_SHA256, "selected.py": "0" * 64}
        with bound_model(sentinel), patch.object(rv3, "NUMERIC_SHA256", hashes):
            with self.assertRaisesRegex(ValueError, "dependency mismatch"):
                rv3._load_components()
            self.assertIs(sys.modules["model"], sentinel)

    def test_full_world_ids_mask_and_public_only_call(self):
        tables = public_world()
        tables["candidates"].index = np.arange(5000, 6049)
        tables["self_labels"].index = np.arange(5000, 6049)
        before = tables["candidates"].copy(deep=True)
        mask = np.zeros(1049, dtype=bool)
        mask[::7] = True
        calls = []
        original = np.linspace(.1, .9, 1049)
        def selected(selection, **kwargs):
            calls.append((selection, set(kwargs)))
            return {"mia": original, "aia": np.full(326, 11.)}
        with patch.object(rv3, "_load_components", return_value=(selected, lambda *args: (mask, {"fixture": True}))), redirect_stdout(StringIO()):
            result = rv3._predict(tables)
        self.assertEqual(len(calls), 22)
        for selection, keys in calls:
            self.assertEqual(selection, rv3.SELECTION)
            self.assertEqual(keys, {"c", "background", "candidates", "challenge", "cfg"})
        self.assertEqual(result["F_mia"].record_id.tolist(), tables["candidates"].record_id.tolist())
        values = result["F_mia"].iloc[:, 1:].to_numpy()
        self.assertTrue(np.all(values[mask] == 0))
        np.testing.assert_array_equal(values[~mask, 0], original[~mask])
        self.assertTrue(np.all(original > 0))  # Masking never mutates model-returned arrays.
        for k, rows in result["F_aia"].groupby("target_k"):
            self.assertEqual(rows.challenge_row_id.tolist(), tables["targets"][k]["challenge"].challenge_row_id.tolist())
        pd.testing.assert_frame_equal(tables["candidates"], before)

    def test_real_selected_numeric_api_uses_only_supplied_small_tables(self):
        tables = public_world()
        public = {"c": tables["targets"][0]["release"].iloc[:30].copy(),
                  "background": tables["background"].iloc[10:50].copy(),
                  "candidates": tables["candidates"].iloc[:8].copy(),
                  "challenge": tables["targets"][0]["challenge"].iloc[:4].copy(), "cfg": {}}
        with read_guard(rv3.__file__) as audit:
            selected, _ = rv3._load_components()
            answer = selected(dict(rv3.SELECTION), **public)
        self.assertEqual(audit["blocked"], [])
        self.assertEqual(answer["mia"].shape, (8,))
        self.assertEqual(answer["aia"].shape, (4,))
        self.assertTrue(np.isfinite(answer["mia"]).all())
        self.assertTrue(np.isfinite(answer["aia"]).all())


if __name__ == "__main__":
    unittest.main()