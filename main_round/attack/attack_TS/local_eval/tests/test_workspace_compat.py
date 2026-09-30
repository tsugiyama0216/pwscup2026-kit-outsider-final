"""Exercise TS import routing in clean processes alongside frozen M files."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


TS_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = TS_ROOT.parent


class WorkspaceCompatibilityTests(unittest.TestCase):
    def run_python(self, body):
        setup = "import sys\nfrom pathlib import Path\n"
        setup += f"ts_root = Path({str(TS_ROOT)!r})\n"
        setup += f"shared_root = Path({str(SHARED_ROOT)!r})\n"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", setup + textwrap.dedent(body)],
            cwd=TS_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_both_ts_entry_packages_initialize_shared_workspace(self):
        for entry in ("attack_research", "local_eval"):
            with self.subTest(entry=entry):
                result = self.run_python(f"""
                    import importlib, json
                    importlib.import_module({entry!r})
                    import attack_workspace as workspace
                    import local_eval
                    from local_eval import generate
                    assert Path(workspace.__file__).resolve() == shared_root / 'attack_workspace.py'
                    assert workspace.ROOT == shared_root
                    assert workspace.TS_ROOT == ts_root
                    assert workspace.TS_OUTPUT == ts_root / 'local-output'
                    assert workspace.TARGETS == shared_root / 'targets'
                    assert Path(local_eval.__file__).resolve() == ts_root / 'local_eval/__init__.py'
                    assert Path(generate.__file__).resolve() == ts_root / 'local_eval/generate.py'
                    print(json.dumps({{'initialized': True}}))
                """)
                self.assertTrue(result["initialized"])

    def test_ts_research_overrides_shared_package_search_precedence(self):
        result = self.run_python("""
            import json
            sys.path[:0] = [str(shared_root), str(ts_root)]
            import attack_research
            import local_eval
            assert Path(local_eval.__file__).resolve() == ts_root / 'local_eval/__init__.py'
            from attack_TS._workspace_compat import initialize
            workspace = initialize()
            before = list(sys.path)
            assert initialize() is workspace
            assert sys.path == before
            assert sys.path[0] == str(ts_root)
            assert sys.path[1] == str(shared_root)
            assert sys.path.count(str(ts_root)) == 1
            assert sys.path.count(str(shared_root)) == 1
            print(json.dumps({'ts_precedence_and_idempotence': True}))
        """)
        self.assertTrue(result["ts_precedence_and_idempotence"])

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
        """)
        self.assertTrue(result["wrong_checkout_rejected"])


if __name__ == '__main__':
    unittest.main()
