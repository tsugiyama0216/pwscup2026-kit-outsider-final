"""Small loader for a research script living outside an importable package."""

from __future__ import annotations

import runpy
from pathlib import Path


def load_zero_helpers(root: Path) -> dict[str, object]:
    path = root / "local-output" / "zero_target_attack" / "evaluate_zero_variants.py"
    namespace = runpy.run_path(str(path), run_name="zero_variant_helpers")
    return {"scenario_releases": namespace["scenario_releases"]}
