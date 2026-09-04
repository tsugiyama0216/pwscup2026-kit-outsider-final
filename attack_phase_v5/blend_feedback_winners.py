"""Build the final AIA candidate from target-wise live feedback.

Each target block is copied intact from the source with the highest official
``aia_per_target`` value.  No row-level truth is inferred and no score is
fabricated; the script only automates the block-wise comparison exposed by the
competition's detailed results.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import OFFICIAL_COLUMNS  # noqa: E402
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    EXPECTED_AIA_ROWS_PER_TARGET,
    load_public_targets,
    sha256_file,
    validate_aia,
    validate_mia,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


def _string_score_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    output: dict[str, float] = {}
    for key, raw_score in value.items():
        try:
            target_id = str(int(key))
            score = float(raw_score)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        output[target_id] = score
    return output


def load_aia_per_target(path: Path) -> dict[str, float]:
    """Read either a direct map or common nested detailed-result shapes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[Any] = [data]
    if isinstance(data, dict):
        candidates.extend(
            (
                data.get("aia_per_target"),
                data.get("_breakdown", {}).get("aia_per_target")
                if isinstance(data.get("_breakdown"), dict)
                else None,
                data.get("breakdown", {}).get("aia_per_target")
                if isinstance(data.get("breakdown"), dict)
                else None,
            )
        )
    for candidate in candidates:
        parsed = _string_score_map(candidate)
        if parsed is not None:
            return parsed
    raise ValueError(f"aia_per_target map was not found in {path}")


def blend_target_winners(
    sources: dict[str, pd.DataFrame],
    scores: dict[str, dict[str, float]],
    *,
    target_ids: Iterable[str] = EXPECTED_TARGET_IDS,
    rows_per_target: int = EXPECTED_AIA_ROWS_PER_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy each whole target block from the best-scoring source."""
    if not sources or set(sources) != set(scores):
        raise ValueError("source and score labels must be equal and non-empty")
    ordered_labels = list(sources)
    target_ids = tuple(str(int(value)) for value in target_ids)
    normalized: dict[str, pd.DataFrame] = {}
    expected_keys: dict[str, set[int]] | None = None
    for label, frame in sources.items():
        if list(frame.columns) != list(OFFICIAL_COLUMNS):
            raise ValueError(f"{label} has invalid AIA columns")
        if frame.duplicated(["target_k", "challenge_row_id"]).any():
            raise ValueError(f"{label} has duplicate AIA keys")
        work = frame.copy()
        work["target_k"] = work["target_k"].astype(int)
        keys: dict[str, set[int]] = {}
        for target_id in target_ids:
            block = work.loc[work["target_k"].eq(int(target_id))]
            if len(block) != rows_per_target:
                raise ValueError(
                    f"{label} target {target_id} has {len(block)} rows; "
                    f"expected {rows_per_target}"
                )
            keys[target_id] = set(block["challenge_row_id"].astype(int))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"{label} challenge keys differ from the first source")
        normalized[label] = work

    missing = {
        label: sorted(set(target_ids) - set(score_map))
        for label, score_map in scores.items()
        if set(target_ids) - set(score_map)
    }
    if missing:
        raise ValueError(f"per-target scores are incomplete: {missing}")

    parts: list[pd.DataFrame] = []
    winners: dict[str, dict[str, Any]] = {}
    counts = {label: 0 for label in ordered_labels}
    for target_id in target_ids:
        winner = max(
            ordered_labels,
            key=lambda label: (scores[label][target_id], -ordered_labels.index(label)),
        )
        block = normalized[winner].loc[
            normalized[winner]["target_k"].eq(int(target_id)), OFFICIAL_COLUMNS
        ]
        parts.append(block.copy())
        counts[winner] += 1
        winners[target_id] = {
            "winner": winner,
            "score": scores[winner][target_id],
            "all_scores": {
                label: scores[label][target_id] for label in ordered_labels
            },
        }
    output = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["target_k", "challenge_row_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return output, {"winners": winners, "winner_counts": counts}


def _parse_mapping(values: list[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in output:
            raise ValueError(f"invalid or duplicate {option} label: {label!r}")
        output[label] = Path(raw_path).resolve()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公式aia_per_target実測値からtarget block winnerを合成します。"
    )
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--scores", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--mia", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-v5" / "final_blend.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = _parse_mapping(args.source, "--source")
    score_paths = _parse_mapping(args.scores, "--scores")
    if set(source_paths) != set(score_paths):
        raise ValueError("--source and --scores labels differ")
    sources = {label: pd.read_csv(path) for label, path in source_paths.items()}
    scores = {
        label: load_aia_per_target(score_paths[label]) for label in source_paths
    }
    blended, diagnostics = blend_target_winners(sources, scores)

    _, challenges = load_public_targets(args.targets.resolve())
    mia = pd.read_csv(args.mia.resolve())
    validate_mia(mia, mia)
    validate_aia(blended, challenges)
    official_dist = official_validate.load_dist(args.targets.resolve())
    mia_result = official_validate.validate_mia_submission(mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(blended, {}, official_dist)
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            f"official logical validator failed: MIA={mia_result.errors}, "
            f"AIA={aia_result.errors}"
        )

    candidate_dir = args.candidate_dir.resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in candidate_dir.iterdir()
        if path.name not in {"F_mia.csv", "F_aia.csv"}
    )
    if unexpected:
        raise ValueError(f"candidate directory has extra files: {unexpected}")
    shutil.copyfile(args.mia.resolve(), candidate_dir / "F_mia.csv")
    blended.to_csv(candidate_dir / "F_aia.csv", index=False)
    diagnostics.update(
        {
            "source_paths": {label: str(path) for label, path in source_paths.items()},
            "score_paths": {label: str(path) for label, path in score_paths.items()},
            "candidate_hashes": {
                "F_mia.csv": sha256_file(candidate_dir / "F_mia.csv"),
                "F_aia.csv": sha256_file(candidate_dir / "F_aia.csv"),
            },
            "official_validator": "PASS",
        }
    )
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: block-wise feedback winner candidate generated")
    print(f"winner counts: {diagnostics['winner_counts']}")
    print(f"candidate: {candidate_dir}")


if __name__ == "__main__":
    main()
