"""Initialize TS through the shared mock toolkit, preserving M's frozen files."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def initialize():
    """Resolve the complete shared evaluator before M's legacy snapshot."""
    mock_root = Path(__file__).resolve().parents[2] / "mock_test"
    value = str(mock_root)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
    bootstrap = importlib.import_module("mock_workspace")
    if Path(bootstrap.__file__).resolve() != mock_root / "mock_workspace.py":
        raise ImportError("TS must use its sibling mock-test workspace bootstrap")
    return bootstrap.initialize()
