"""Initialize TS paths without changing M's byte-frozen shared workspace."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def initialize():
    """Keep TS packages first and attach TS output paths to the shared runtime."""
    ts_root = Path(__file__).resolve().parent
    shared_root = ts_root.parent
    # M retains three frozen local_eval compatibility files in the shared root.
    # TS must resolve its complete local_eval package before those files.
    for root in (shared_root, ts_root):
        value = str(root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    workspace = importlib.import_module("attack_workspace")
    if Path(workspace.__file__).resolve() != shared_root / "attack_workspace.py":
        raise ImportError("TS must use attack_workspace.py from its shared parent")
    workspace.TS_ROOT = ts_root
    workspace.TS_OUTPUT = ts_root / "local-output"
    return workspace
