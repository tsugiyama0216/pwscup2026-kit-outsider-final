"""Local attack comparison with known shadow labels, never official scoring."""

# Direct module commands run from mock_test and share the frozen attack runtime.
from pathlib import Path as _Path
import sys as _sys

_mock_root = _Path(__file__).resolve().parents[1]
if str(_mock_root) not in _sys.path:
    _sys.path.insert(0, str(_mock_root))

import mock_workspace as _bootstrap

if _Path(_bootstrap.__file__).resolve() != _mock_root / "mock_workspace.py":
    raise ImportError("Mock tests must use their own mock_workspace.py bootstrap")
_bootstrap.initialize()
