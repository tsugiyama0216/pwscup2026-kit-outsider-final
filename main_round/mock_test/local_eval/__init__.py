"""Local attack comparison with known shadow labels, never official scoring."""

# Direct module commands run from mock_test and share the pinned attack runtime.
from pathlib import Path as _Path
import sys as _sys

_shared_root = str(_Path(__file__).resolve().parents[2] / "attack")
if _shared_root not in _sys.path:
    _sys.path.insert(0, _shared_root)
