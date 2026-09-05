"""Run the preserved preliminary workspace suite from the Git repository root.

Historical commands also work directly from ``prelim/workspace``. This bridge
does not import preliminary attack code into the future main-round workspace.
"""
from pathlib import Path
import sys
import unittest


def load_tests(loader, tests, pattern):
    workspace = Path(__file__).resolve().parent / "prelim" / "workspace"
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    # A fresh loader avoids changing the caller's root-discovery state.
    return unittest.TestLoader().discover(
        start_dir=str(workspace), pattern=pattern or "test*.py",
        top_level_dir=str(workspace),
    )
