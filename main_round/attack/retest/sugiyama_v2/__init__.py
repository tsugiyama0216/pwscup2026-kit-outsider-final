"""Saved retest evaluation; reuse the repository's frozen runtime bootstrap."""
from pathlib import Path as _Path
import sys as _sys

_retest_root = _Path(__file__).resolve().parents[1]
_main_round = _Path(__file__).resolve().parents[3]
_mock_root = _main_round / "mock_test"
if not (_mock_root / "mock_workspace.py").is_file():
    raise ImportError("retest must remain under main_round/attack")
_sys.path.insert(0, str(_mock_root))
import local_eval  # Preserve the existing frozen workspace bootstrap unchanged.
if _Path(local_eval.__file__).resolve() != _mock_root / "local_eval/__init__.py":
    raise ImportError("retest requires the canonical shared local_eval bootstrap")
# Keep this saved evaluator ahead of its original implementation on sys.path.
while str(_retest_root) in _sys.path:
    _sys.path.remove(str(_retest_root))
_sys.path.insert(0, str(_retest_root))
