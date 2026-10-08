"""Run all new attack tests from any working directory, without prelim truths."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("explore_common.test_core"))
    for relative in ("explore_3_gen_lra/tests/test_model.py", "explore_4_predictor/tests/test_plugin.py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(relative.replace("/", "_").replace(".py", ""), HERE / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
