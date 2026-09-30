"""Check shared/TS import routing and frozen M behavior in fresh processes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


MOCK_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = MOCK_ROOT.parent / "attack"
TS_ROOT = SHARED_ROOT / "attack_TS"


class WorkspaceCompatibilityTests(unittest.TestCase):
    def run_python(self, body, *, cwd):
        setup = "import sys\nfrom pathlib import Path\n"
        for name, path in (("mock_root", MOCK_ROOT), ("ts_root", TS_ROOT),
                           ("shared_root", SHARED_ROOT)):
            setup += f"{name} = Path({str(path)!r})\n"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", setup + textwrap.dedent(body)],
            cwd=cwd, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_common_and_ts_entry_points_use_shared_evaluator(self):
        for cwd, entry in ((MOCK_ROOT, "local_eval"), (TS_ROOT, "attack_research.features")):
            with self.subTest(entry=entry):
                result = self.run_python(f"""
                    import importlib, json
                    importlib.import_module({entry!r})
                    import attack_workspace as workspace
                    import local_eval
                    from local_eval import generate, evaluate_case
                    assert Path(workspace.__file__).resolve() == shared_root / 'attack_workspace.py'
                    assert workspace.ROOT == shared_root
                    assert workspace.TS_ROOT == ts_root
                    assert workspace.TS_OUTPUT == ts_root / 'local-output'
                    assert workspace.MOCK_ROOT == mock_root
                    assert workspace.MOCK_OUTPUT == mock_root / 'local-output'
                    assert workspace.TARGETS == shared_root / 'targets'
                    assert Path(local_eval.__file__).resolve() == mock_root / 'local_eval/__init__.py'
                    assert Path(generate.__file__).resolve() == mock_root / 'local_eval/generate.py'
                    assert Path(evaluate_case.__file__).resolve() == mock_root / 'local_eval/evaluate_case.py'
                    print(json.dumps({{'initialized': True}}))
                """, cwd=cwd)
                self.assertTrue(result["initialized"])

    def test_shared_precedence_and_repeated_initialization(self):
        result = self.run_python("""
            import json
            sys.path[:0] = [str(shared_root), str(ts_root)]
            import attack_research
            import local_eval
            import local_eval.tests
            assert Path(local_eval.__file__).resolve() == mock_root / 'local_eval/__init__.py'
            assert local_eval.tests.__file__ is None
            from attack_TS._workspace_compat import initialize
            from mock_workspace import initialize as initialize_shared
            workspace = initialize()
            before = list(sys.path)
            assert initialize() is workspace
            assert initialize_shared() is workspace
            assert sys.path == before
            assert sys.path[:3] == [str(mock_root), str(ts_root), str(shared_root)]
            for root in (mock_root, ts_root, shared_root):
                assert sys.path.count(str(root)) == 1
            print(json.dumps({'precedence_and_idempotence': True}))
        """, cwd=TS_ROOT)
        self.assertTrue(result["precedence_and_idempotence"])

    def test_fresh_m_process_retains_frozen_runtime_and_hash_gates(self):
        result = self.run_python("""
            import json
            import attack_workspace as workspace
            import local_eval
            from local_eval import data, scoring
            from M.v2.artifacts import load_selection, check_v1_preserved
            assert Path(workspace.__file__).resolve() == shared_root / 'attack_workspace.py'
            assert Path(local_eval.__file__).resolve() == shared_root / 'local_eval/__init__.py'
            assert Path(data.__file__).resolve() == shared_root / 'local_eval/data.py'
            assert Path(scoring.__file__).resolve() == shared_root / 'local_eval/scoring.py'
            assert not hasattr(workspace, 'MOCK_ROOT')
            selection = load_selection()
            preserved = check_v1_preserved()
            assert selection['selected_before_holdout']
            assert preserved['all_bytes_unchanged']
            print(json.dumps({'frozen_dependencies': len(selection['dependency_sha256']),
                              'preserved_files': preserved['verified_files']}))
        """, cwd=SHARED_ROOT)
        self.assertGreater(result["frozen_dependencies"], 0)
        self.assertGreater(result["preserved_files"], 0)

    def test_preloaded_frozen_package_is_rejected_without_replacing_modules(self):
        result = self.run_python("""
            import json
            import local_eval
            from local_eval import data
            original_package, original_data = local_eval, data
            sys.path.insert(0, str(ts_root))
            try:
                import attack_research
            except ImportError as error:
                assert 'fresh Python process' in str(error)
            else:
                raise AssertionError('Frozen package was silently reused or replaced')
            assert sys.modules['local_eval'] is original_package
            assert sys.modules['local_eval.data'] is original_data
            assert Path(data.__file__).resolve() == shared_root / 'local_eval/data.py'
            from M.v2.artifacts import load_selection
            load_selection()
            print(json.dumps({'cached_frozen_modules_preserved': True}))
        """, cwd=SHARED_ROOT)
        self.assertTrue(result["cached_frozen_modules_preserved"])

    def test_wrong_checkout_workspace_is_rejected(self):
        result = self.run_python("""
            import json, types
            sys.path.insert(0, str(shared_root))
            from attack_TS._workspace_compat import initialize
            sys.modules['attack_workspace'] = types.SimpleNamespace(
                __file__=str(ts_root / 'wrong_checkout/attack_workspace.py'))
            try:
                initialize()
            except ImportError:
                print(json.dumps({'wrong_checkout_rejected': True}))
            else:
                raise AssertionError('Unexpected workspace was accepted')
        """, cwd=TS_ROOT)
        self.assertTrue(result["wrong_checkout_rejected"])


if __name__ == '__main__':
    unittest.main()
