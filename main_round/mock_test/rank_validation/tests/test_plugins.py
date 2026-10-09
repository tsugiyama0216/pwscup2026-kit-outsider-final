"""Public-only plugin contract, output validation and read audit."""
from pathlib import Path
import unittest
import importlib
import importlib.util
import py_compile
import sys
import hashlib
import os
import subprocess
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from rank_validation import adapters, plugins
from rank_validation.tests.test_adapters import public_world

GOOD = '''
import numpy as np
import pandas as pd

def predict_world(*, candidates, self_labels, background, self_challenge, targets, schema, cfg):
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy(),
        **{str(k): np.full(len(candidates), .5) for k in sorted(targets)}})
    aia = pd.concat([pd.DataFrame({"target_k": k,
        "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(), "time_hat": 11.})
        for k, item in sorted(targets.items())], ignore_index=True)
    candidates.loc[0, "SBP"] = -98765
    return {"F_mia": mia, "F_aia": aia, "diagnostics": {"custom": "kept"}}
'''


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.scratch = adapters.MOCK / "local-output" / ("plugin-test-" + uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.plugin = self.scratch / "candidate.py"
        self.helper_name = "_rank_helper_" + uuid4().hex
        self.helper = self.scratch / (self.helper_name + ".py")

    def tearDown(self):
        sys.modules.pop(self.helper_name, None)
        self.assertTrue(self.scratch.resolve().is_relative_to((adapters.MOCK / "local-output").resolve()))
        for path in self.scratch.iterdir():
            path.unlink()
        self.scratch.rmdir()

    def write(self, source):
        self.plugin.write_text(source, encoding="utf-8")
        return self.plugin

    def test_safe_names_and_reserved_frozen_names(self):
        self.assertEqual(plugins.candidate_name("candidate_v2-test.3"), "candidate_v2-test.3")
        for value in (None, "", "../escape", "a..b", "bad name", "a.", "CON", "con.x", "Rv2", "tv105"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plugins.candidate_name(value)

    def test_public_contract_validation_and_copy_isolation(self):
        public = public_world()
        before = public["candidates"].copy(deep=True)
        result = plugins.predict_plugin(self.write(GOOD), "candidate_test", **public)
        pd.testing.assert_frame_equal(public["candidates"], before)
        self.assertEqual(result["diagnostics"]["validation"]["target_count"], 22)
        self.assertEqual(result["diagnostics"]["plugin_diagnostics"], {"custom": "kept"})
        self.assertEqual(result["diagnostics"]["plugin_sha256"], adapters.sha(self.plugin))
        self.assertEqual(result["diagnostics"]["blocked_reads"], [])

    def test_missing_function_rejected(self):
        with self.assertRaisesRegex(ValueError, "expose predict_world"):
            plugins.predict_plugin(self.write("x=1"), "candidate_test", **public_world())

    def test_malformed_outputs_rejected(self):
        for source in ("def predict_world(**kwargs): return {}", GOOD.replace('"time_hat": 11.', '"time_hat": float("nan")')):
            with self.subTest(source=source[:40]), self.assertRaises(ValueError):
                plugins.predict_plugin(self.write(source), "candidate_test", **public_world())

    def test_cached_prediction_reads_rejected_even_if_exception_suppressed(self):
        forbidden = adapters.ATTACK / "Attack_R/v1/submission/F_aia.csv"
        source = GOOD + '\nfrom pathlib import Path\ntry:\n    Path(' + repr(str(forbidden)) + ').read_bytes()\nexcept RuntimeError:\n    pass\n'
        with self.assertRaisesRegex(RuntimeError, "read forbidden"):
            plugins.predict_plugin(self.write(source), "candidate_test", **public_world())

    def test_csv_cannot_be_declared_fitted_asset(self):
        source = GOOD + '\nFITTED_ASSETS={"main_round/attack/Attack_R/v1/submission/F_aia.csv": "' + "0"*64 + '"}\n'
        with self.assertRaisesRegex(ValueError, "Invalid declared"):
            plugins.predict_plugin(self.write(source), "candidate_test", **public_world())

    def helper_plugin(self, source="VALUE = .25\n"):
        self.helper.write_text(source, encoding="utf-8")
        return self.write("import " + self.helper_name + " as helper\n" + GOOD.replace(".5)", "helper.VALUE)"))

    def test_imported_helper_hash_changes_across_runs(self):
        plugin = self.helper_plugin()
        with patch.object(sys, "path", [str(self.scratch), *sys.path]):
            first = plugins.predict_plugin(plugin, "candidate_test", **public_world())
            sys.modules.pop(self.helper_name, None)
            self.helper.write_text("VALUE = .75\n", encoding="utf-8")
            importlib.invalidate_caches()
            second = plugins.predict_plugin(plugin, "candidate_test", **public_world())
        relative = self.helper.relative_to(adapters.ROOT).as_posix()
        self.assertNotEqual(first["diagnostics"]["imported_source_sha256"][relative],
                            second["diagnostics"]["imported_source_sha256"][relative])
        self.assertEqual(second["diagnostics"]["imported_source_sha256"][relative], adapters.sha(self.helper))
        self.assertTrue(second["diagnostics"]["imported_sources_unchanged"])

    def test_preloaded_cached_import_is_recorded(self):
        plugin = self.helper_plugin()
        with patch.object(sys, "path", [str(self.scratch), *sys.path]):
            importlib.import_module(self.helper_name)
            result = plugins.predict_plugin(plugin, "candidate_test", **public_world())
        relative = self.helper.relative_to(adapters.ROOT).as_posix()
        self.assertEqual(result["diagnostics"]["imported_source_sha256"][relative], adapters.sha(self.helper))

    def test_pyc_and_corresponding_source_are_recorded(self):
        self.helper.write_text("VALUE = .25\n", encoding="utf-8")
        compiled = self.helper.with_suffix(".pyc")
        py_compile.compile(str(self.helper), cfile=str(compiled), doraise=True)
        prefix = ("import importlib.util, sys\n"
                  "spec = importlib.util.spec_from_file_location(" + repr(self.helper_name) + ", " + repr(str(compiled)) + ")\n"
                  "helper = importlib.util.module_from_spec(spec)\n"
                  "sys.modules[" + repr(self.helper_name) + "] = helper\n"
                  "spec.loader.exec_module(helper)\n")
        result = plugins.predict_plugin(self.write(prefix + GOOD), "candidate_test", **public_world())
        for path in (compiled, self.helper):
            relative = path.relative_to(adapters.ROOT).as_posix()
            self.assertEqual(result["diagnostics"]["imported_source_sha256"][relative], adapters.sha(path))

    def test_helper_write_during_inference_is_rejected_without_changing_bytes(self):
        self.helper_plugin()
        before = self.helper.read_bytes()
        code = self.plugin.read_text().replace("    mia =", "    from pathlib import Path\n    Path(" + repr(str(self.helper)) + ").write_text('VALUE = .75')\n    mia =")
        with patch.object(sys, "path", [str(self.scratch), *sys.path]):
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                plugins.predict_plugin(self.write(code), "candidate_test", **public_world())
        self.assertEqual(self.helper.read_bytes(), before)

    def test_external_helper_change_during_inference_is_detected_afterward(self):
        self.helper_plugin()
        mutation = "from pathlib import Path; Path(" + repr(str(self.helper)) + ").write_text('VALUE = .75')"
        injected = "    import subprocess, sys\n    subprocess.run([sys.executable, '-B', '-c', " + repr(mutation) + "], check=True)\n    mia ="
        code = self.plugin.read_text().replace("    mia =", injected)
        with patch.object(sys, "path", [str(self.scratch), *sys.path]):
            with self.assertRaisesRegex(RuntimeError, "Python dependency changed"):
                plugins.predict_plugin(self.write(code), "candidate_test", **public_world())

    def test_cli_requires_exclusive_model_or_plugin_and_name(self):
        from rank_validation.model_worker import main
        for args in (["--model", "Rv1", "--plugin", "x.py"], ["--plugin", "x.py"], ["--model", "Rv1", "--name", "ignored"]):
            with self.subTest(args=args), patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
                main([*args, "--public", "unused", "--output", "unused"])
            self.assertEqual(caught.exception.code, 2)


class FittedAssetBoundaryTests(unittest.TestCase):
    """Only generated fake fixtures are used; no real roles/truth are read."""
    def setUp(self):
        self.scratch = adapters.MOCK / "local-output" / ("plugin-assets-test-" + uuid4().hex)
        self.repo = self.scratch / "repo"
        self.repo.mkdir(parents=True)
        self.plugin = self.repo / "candidate.py"
        self.plugin.write_text(GOOD, encoding="utf-8")
        self.links = []
        self.scope = patch.multiple(plugins, ROOT=self.repo, ATTACK=self.repo / "main_round/attack",
                                    MOCK=self.repo / "main_round/mock_test")
        self.scope.start()

    def tearDown(self):
        self.scope.stop()
        self.assertTrue(self.scratch.resolve().is_relative_to((adapters.MOCK / "local-output").resolve()))
        for link in self.links:
            if link.is_symlink():
                link.unlink()
            elif getattr(link, "is_junction", lambda: False)():
                link.rmdir()
        for path in sorted(self.scratch.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            self.assertTrue(path.resolve().is_relative_to(self.scratch.resolve()))
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                path.rmdir()
        self.scratch.rmdir()

    def fixture(self, relative, payload=b'{"fixture": "fake test data only"}'):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def declaration(self, relative, digest, *, read_path=None):
        source = GOOD + "\nFITTED_ASSETS={" + repr(relative) + ":" + repr(digest) + "}\n"
        if read_path is not None:
            source = source.replace("    mia =", "    from pathlib import Path\n    assert Path(" + repr(str(read_path)) + ").read_bytes()\n    mia =")
        self.plugin.write_text(source, encoding="utf-8")
        return self.plugin

    def test_scoring_truth_roles_declaration_rejected_before_hash_or_read(self):
        relative = "main_round/mock_test/local-output/panels/fake/scoring_truth/roles.json"
        path, digest = self.fixture(relative)
        original_sha = plugins.sha
        def refuse_fixture_read(candidate):
            if Path(candidate).resolve() == path.resolve():
                raise AssertionError("The fake truth file must not even be hashed")
            return original_sha(candidate)
        with patch.object(plugins, "sha", side_effect=refuse_fixture_read):
            with self.assertRaisesRegex(ValueError, "Invalid declared fitted asset"):
                plugins.predict_plugin(self.declaration(relative, digest, read_path=path), "candidate_test", **public_world())

    def test_all_mock_materials_and_model_evaluation_paths_are_ineligible(self):
        names = ("main_round/mock_test/local-output/public/coefficients.json",
                 "main_round/mock_test/candidates/model.joblib",
                 "main_round/mock_test/official-kit/roles.json",
                 "main_round/attack/attackT/Candidate/scoring_truth/weights.json",
                 "main_round/attack/attackT/Candidate/SCORING_TRUTH/weights.json",
                 "main_round/attack/attackT/Candidate/private/weights.json",
                 "main_round/attack/attackT/Candidate/ground_truth/weights.json",
                 "main_round/attack/attackT/Candidate/evidence/weights.joblib",
                 "main_round/attack/attackT/Candidate/participant_data/weights.json",
                 "main_round/attack/attackT/Candidate/targets/weights.json",
                 "main_round/attack/attackT/Candidate/public/weights.json",
                 "main_round/attack/attackT/Candidate/mock_TS_evaluation_20261009/weights.json",
                 "main_round/attack/attackT/Candidate/panels/weights.json",
                 "main_round/attack/attackT/Candidate/results/weights.json",
                 "main_round/attack/attackT/Candidate/roles.json",
                 "main_round/attack/attackT/Candidate/ROLES.JSON",
                 "main_round/attack/attackT/Candidate/case_metadata.json",
                 "main_round/attack/attackT/Candidate/panel.json",
                 "main_round/attack/attackT/Candidate/manifest.json",
                 "main_round/attack/reference_baseline/params.json",
                 "main_round/anonymization/team10/participant_data/params.json")
        with patch.object(plugins, "sha", side_effect=AssertionError("No evaluation file may be hashed")):
            for relative in names:
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Invalid declared"):
                    plugins._validated_fitted_asset(relative, "0" * 64)

    def test_traversal_absolute_and_drive_relative_declarations_rejected(self):
        for relative in ("main_round/attack/attackT/Candidate/../Other/params.json",
                         "../external/params.json", str(self.repo / "params.json"),
                         "C:params.json", "main_round/attack/attackT/Candidate/params.json:stream"):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Invalid declared"):
                plugins._validated_fitted_asset(relative, "0" * 64)

    def test_model_looking_links_cannot_resolve_to_evaluation_or_outside_root(self):
        for i, target in enumerate((self.repo / "main_round/mock_test/local-output/fake/scoring_truth", self.scratch / "outside_repo")):
            target.mkdir(parents=True)
            (target / "coefficients.json").write_bytes(b'{"fake":true}')
            link = self.repo / f"main_round/attack/attackT/Candidate/linked_{i}"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    raise
                created = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
                if created.returncode:
                    self.skipTest("Temporary directory links unavailable on this host")
            self.links.append(link)
            relative = (link / "coefficients.json").relative_to(self.repo).as_posix()
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "resolved location"):
                plugins._validated_fitted_asset(relative, "0" * 64)

    def test_evaluation_rejection_precedes_asset_allowlist(self):
        path, _ = self.fixture("main_round/mock_test/local-output/fake/scoring_truth/roles.json")
        with plugins.read_guard(self.plugin) as state:
            state["assets"].add(path.resolve())
            with self.assertRaisesRegex(RuntimeError, "evaluation material"):
                path.read_bytes()
        self.assertEqual(state["blocked"], [str(path.resolve())])

    def test_undeclared_model_data_cannot_be_read_regardless_of_extension(self):
        for suffix in (".json", ".yaml", ".bin", ".dat", ".txt"):
            path, _ = self.fixture("main_round/attack/attack_TS/Candidate/params" + suffix)
            with self.subTest(suffix=suffix), plugins.read_guard(self.plugin):
                with self.assertRaisesRegex(RuntimeError, "undeclared repository data"):
                    path.read_bytes()

    def test_fitted_asset_hash_mismatch_and_verified_read(self):
        for relative in ("main_round/attack/attack_TS/Candidate/params.json",
                         "main_round/attack/Attak_T/local-output/attack-research/round03/router.joblib"):
            path, digest = self.fixture(relative)
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    plugins.predict_plugin(self.declaration(relative, "0" * 64), "candidate_test", **public_world())
                result = plugins.predict_plugin(self.declaration(relative, digest, read_path=path), "candidate_test", **public_world())
                self.assertEqual(result["diagnostics"]["fitted_assets_sha256"][relative], digest)
                self.assertEqual(path, plugins._validated_fitted_asset(relative, digest))


if __name__ == "__main__":
    unittest.main()