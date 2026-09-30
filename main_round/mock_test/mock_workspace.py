"""Initialize shared mock-test paths without editing M's frozen runtime files."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def initialize():
    """Use the shared evaluator and attach paths to the unchanged attack runtime."""
    mock_root = Path(__file__).resolve().parent
    shared_root = mock_root.parent / "attack"
    ts_root = shared_root / "attack_TS"
    package_root = mock_root / "local_eval"
    # Changing sys.path cannot safely replace modules already cached by Python.
    # M's legacy evaluator snapshot must keep its identity in an existing process.
    for name, module in tuple(sys.modules.items()):
        if name == "local_eval" or name.startswith("local_eval."):
            origin = getattr(module, "__file__", None)
            if origin is None:
                # tests/ is an ordinary Python namespace package in this toolkit.
                locations = tuple(getattr(module, "__path__", ()))
                correct_package = bool(locations) and all(
                    Path(path).resolve().is_relative_to(package_root) for path in locations
                )
            else:
                correct_package = Path(origin).resolve().is_relative_to(package_root)
            if not correct_package:
                raise ImportError(
                    "local_eval is already loaded from a frozen or different workspace; "
                    "start a fresh Python process from main_round/mock_test, "
                    "or import attack_research before local_eval for TS work"
                )
    for root in (shared_root, ts_root, mock_root):
        value = str(root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    workspace = importlib.import_module("attack_workspace")
    origin = getattr(workspace, "__file__", None)
    if origin is None or Path(origin).resolve() != shared_root / "attack_workspace.py":
        raise ImportError("Mock tests must use attack_workspace.py from their shared sibling")
    workspace.TS_ROOT = ts_root
    workspace.TS_OUTPUT = ts_root / "local-output"
    workspace.MOCK_ROOT = mock_root
    workspace.MOCK_OUTPUT = mock_root / "local-output"
    return workspace
