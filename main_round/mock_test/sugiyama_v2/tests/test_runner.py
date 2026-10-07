"""Saved-world integration and trusted-plugin boundary regression checks."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import textwrap
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

import sugiyama_v2  # Initialize canonical workspace paths for standalone test discovery.
import attack_workspace as workspace
from pwscup2026.kit.codabench.scoring_io import score_attack
from sugiyama_v2.generate import build_world
from sugiyama_v2.io import export_world, load_world, snapshot
from sugiyama_v2.runner import evaluate, load_plugin


PUBLIC_KEYS = {"candidates", "self_labels", "background", "self_challenge", "targets", "schema", "cfg"}
CHEAP_PLUGIN = '''
import numpy as np
import pandas as pd

MODEL_NAME = "constant-public-boundary-test"

def predict_world(**inputs):
    expected = {"candidates", "self_labels", "background", "self_challenge", "targets", "schema", "cfg"}
    assert set(inputs) == expected, "Hidden or missing plugin arguments"
    assert all(set(item) == {"release", "challenge"} for item in inputs["targets"].values())
    assert "is_rare" not in inputs["candidates"]
    assert "is_rare" in inputs["self_labels"]
    keys = sorted(inputs["targets"], reverse=True)
    mia = pd.DataFrame({"record_id": inputs["candidates"].record_id.to_numpy(copy=True),
                        **{str(k): np.full(1049, .5) for k in keys}})
    rows = []
    for target in keys:
        query = inputs["targets"][target]["challenge"]
        rows.append(pd.DataFrame({"target_k": target,
            "challenge_row_id": query.challenge_row_id.to_numpy(copy=True), "time_hat": 2.}))
    aia = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=12).reset_index(drop=True)
    # Mutating every argument must not change the evaluator's data or truth.
    inputs["candidates"].loc[:, "record_id"] = -1
    inputs["self_labels"].loc[:, "is_rare"] = 7
    inputs["background"].loc[:, "time"] = 100.
    inputs["self_challenge"].loc[:, "challenge_row_id"] = -1
    inputs["targets"][keys[0]]["challenge"].loc[:, "challenge_row_id"] = -1
    inputs["targets"][keys[0]]["release"].loc[:, "time"] = 100.
    inputs["schema"]["age_range"][0] = -1
    inputs["cfg"]["scoring"]["low_fpr"] = .99
    return {"F_mia": mia, "F_aia": aia,
            "diagnostics": {"received_keys": sorted(inputs), "mutated_own_arguments": True}}
'''


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inherited workspace ACLs are required under the restricted Windows
        # desktop token; tempfile's private-mode directory can be inaccessible.
        cls.scratch_parent = (workspace.MOCK_OUTPUT / "sugiyama-runner-tests").resolve()
        cls.scratch_parent.mkdir(parents=True, exist_ok=True)
        cls.root = cls.scratch_parent / ("run-" + uuid4().hex)
        cls.root.mkdir()
        cls.addClassCleanup(cls.clean_scratch)
        cls.case = cls.root / "case"
        world = build_world(2026100803, release_factory=lambda target, background, style, seed, schema: target.copy(deep=True))
        cls.world = world
        export_world(world, cls.case)
        cls.before = snapshot(cls.case)[1]

    @classmethod
    def clean_scratch(cls):
        final = cls.root.resolve()
        if final.parent != cls.scratch_parent or not final.name.startswith("run-"):
            raise RuntimeError("Runner-test cleanup escaped its dedicated workspace directory")
        if final.exists():
            shutil.rmtree(final)

    def plugin(self, name, source):
        path = self.root / (name + "-" + uuid4().hex + ".py")
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def test_generate_save_evaluate_public_only_and_official_score_parity(self):
        plugin = self.plugin("constant_mutating", CHEAP_PLUGIN)
        destination = self.root / "evaluation"
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(self.case, plugin, destination, floor_seed=812, floor_repetitions=2)
        public, private, manifest, after = load_world(self.case)
        self.assertEqual(after, self.before)
        self.assertTrue(result["completed"])
        self.assertTrue(result["case_files_unchanged"])
        self.assertEqual(result["model"], "constant-public-boundary-test")
        self.assertEqual(set(result["plugin_diagnostics"]["received_keys"]), PUBLIC_KEYS)
        self.assertTrue(result["plugin_diagnostics"]["mutated_own_arguments"])
        self.assertEqual(result["config"]["scoring"]["low_fpr"], .01)
        self.assertTrue(public["candidates"].record_id.ge(0).all())
        self.assertTrue(public["self_labels"].is_rare.isin([0, 1]).all())
        self.assertEqual(public["schema"]["age_range"], [40, 74])
        self.assertFalse(result["official_score_prediction"])
        self.assertEqual(result["floor_estimate"]["repetitions"], 2)
        self.assertTrue(result["floor_estimate"]["diagnostic_reduced_repetitions"])
        self.assertEqual(result["validation"], {"mia_rows": 1049, "aia_rows": 7172, "target_count": 22})
        truth_names = ["pool_ids_10.csv", "case_metadata.json"]
        truth_names += [f"pool_ids_{target}.csv" for target in public["targets"]]
        truth_names += [f"aia_truth_{target}.csv" for target in public["targets"]]
        for name in truth_names:
            self.assertEqual((private / name).read_bytes(), (destination / "scoring_truth" / name).read_bytes())
        frames = {name: pd.read_csv(destination / name, float_precision="round_trip")
                  for name in ("F_mia.csv", "F_aia.csv")}
        expected = score_attack(frames, destination / "scoring_truth", result["config"], team_id=10)
        fields = ("attack_power", "mia_all", "mia_rare", "aia_all", "aia_rare")
        for number, field in enumerate(fields, 1):
            self.assertEqual(result["leaderboard"][f"score_{number}"], expected[field])
        self.assertEqual(result["mia_power"], expected["mia_power"])
        self.assertEqual(result["aia_power"], expected["aia_power"])
        self.assertEqual(len(pd.read_csv(destination / "by_target.csv")), 22)
        stored = json.loads((destination / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["leaderboard"], result["leaderboard"])
        self.assertEqual(stored["case_manifest_sha256"], self.before["manifest"])
        with self.assertRaisesRegex(ValueError, "Preserve previous output"):
            evaluate(self.case, plugin, destination, floor_repetitions=2)
        self.assertEqual(snapshot(self.case)[1], self.before)

    def test_saved_case_and_nested_output_are_never_overwritten(self):
        plugin = self.plugin("constant", CHEAP_PLUGIN)
        with self.assertRaisesRegex(ValueError, "Preserve previous output"):
            export_world(self.world, self.case)
        with self.assertRaises(ValueError):
            evaluate(self.case, plugin, self.case, floor_repetitions=2)
        nested = self.case / "evaluation"
        with self.assertRaisesRegex(ValueError, "separate"):
            evaluate(self.case, plugin, nested, floor_repetitions=2)
        self.assertFalse(nested.exists())
        self.assertEqual(snapshot(self.case)[1], self.before)

    def test_invalid_prediction_fails_before_scoring_and_leaves_case_unchanged(self):
        invalid = CHEAP_PLUGIN + '''
original_predict = predict_world
def predict_world(**inputs):
    result = original_predict(**inputs)
    result["F_mia"].iloc[0, 1] = np.nan
    return result
'''
        plugin = self.plugin("invalid", invalid)
        destination = self.root / "invalid-evaluation"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "finite"):
            evaluate(self.case, plugin, destination, floor_repetitions=2)
        self.assertFalse((destination / "result.json").exists())
        self.assertFalse((destination / "scoring_truth").exists())
        self.assertEqual(snapshot(self.case)[1], self.before)

    def test_load_plugin_preserves_original_errors_and_cleans_partial_module(self):
        for source, error, message in (
            ("raise RuntimeError('plugin sentinel')", RuntimeError, "plugin sentinel"),
            ("def broken(:\n pass", SyntaxError, "invalid syntax"),
            ("VALUE = 1", ValueError, "predict_world"),
        ):
            with self.subTest(error=error.__name__):
                path = self.plugin("bad-import", source)
                before = {name for name in sys.modules if name.startswith("sugiyama_plugin_")}
                with self.assertRaisesRegex(error, message):
                    load_plugin(path)
                after = {name for name in sys.modules if name.startswith("sugiyama_plugin_")}
                self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
