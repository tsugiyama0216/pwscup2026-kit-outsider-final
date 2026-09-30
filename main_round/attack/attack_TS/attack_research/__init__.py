"""Contest-only attack research on a fixed, label-separated shadow protocol."""

# Direct module commands run from attack_TS and share the parent workspace.
from pathlib import Path as _Path
import sys as _sys

_shared_root = str(_Path(__file__).resolve().parents[2])
if _shared_root not in _sys.path:
    _sys.path.insert(0, _shared_root)
_mock_root = str(_Path(__file__).resolve().parents[3] / "mock_test")
if _mock_root not in _sys.path:
    _sys.path.insert(0, _mock_root)
