"""Rebuild normal-format production MIA scores from the actual team-24 cohort.

Research-only: reads B/B_self/A_bg and public targets, writes only F_mia
variants and diagnostics below local-output.  It never opens token.txt, never
modifies an existing candidate, and never creates an archive.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
for source in (ROOT, ROOT / "starter" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_mia_independent,
)
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v3 import generate_submission_v3 as v3  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    load_public_targets,
    validate_mia,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


ACTUAL = Path(
    r"C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
)
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "attack-v8" / "actual-mia"
TEAM = ROOT / "submissions" / "prelim" / "team24"
EXPECTED_HASHES = {
    "B.csv": "7A7AA39E16FA5C76EE9C8DC009928373AEC6CF47E7B1BD10DC4E6899B5D2AB3A",
    "B_self.csv": "297354A3C909FBA6BF4D33053CF91EFA9139892BF5269050A5607B86E0F33E35",
    "A_bg.csv": "F4817FBA6C1434D6B74221C2B5ABE90E4A53E80D8FA17E12602D43ADAE0A3126",
    "schema.json": "D0698EB89CDE0901B28DDAF72CB57270D744BDB72B45F8315F75B3199F024128",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    observed = {name: sha256(ACTUAL / name) for name in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise RuntimeError(f"actual input hash mismatch: {observed}")
    candidates = pd.read_csv(ACTUAL / "B.csv").reset_index(drop=True)
    b_self = pd.read_csv(ACTUAL / "B_self.csv")
    aligned_self = b_self.set_index("record_id").loc[candidates["record_id"]].reset_index()
    if not candidates.equals(aligned_self[candidates.columns]):
        raise RuntimeError("B and B_self are not aligned")
    background = pd.read_csv(ACTUAL / "A_bg.csv").reset_index(drop=True)
    releases, _ = load_public_targets(TARGETS)
    self_challenge = pd.read_csv(TARGETS / "aia_challenge_24.csv")
    known_private, private_diagnostics = v3.discover_known_private_records(
        candidates, self_challenge
    )
    rare = aligned_self["is_rare"].astype(bool).to_numpy()
    if int(rare.sum()) != 24 or int((rare & known_private).sum()) != 13:
        raise RuntimeError("actual rare/private cardinality differs")
    return (
        candidates,
        aligned_self,
        background,
        releases,
        known_private,
        rare,
        {"hashes": observed, "private": private_diagnostics},
    )


def rank_overlap(left: pd.DataFrame, right: pd.DataFrame, width: int) -> float:
    overlaps = []
    for target in left.columns[1:]:
        left_top = set(left.nlargest(width, target)["record_id"].astype(int))
        right_top = set(right.nlargest(width, target)["record_id"].astype(int))
        overlaps.append(len(left_top & right_top) / width)
    return float(np.mean(overlaps))


def comparison(frame: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    baseline = reference.iloc[:, 1:].to_numpy(dtype=float)
    correlations = [
        float(spearmanr(values[:, index], baseline[:, index]).statistic)
        for index in range(values.shape[1])
    ]
    return {
        "changed_cells": int(np.count_nonzero(values != baseline)),
        "changed_rows": int(np.count_nonzero(np.any(values != baseline, axis=1))),
        "mean_spearman": float(np.mean(correlations)),
        "minimum_spearman": float(np.min(correlations)),
        "mean_top10_overlap": rank_overlap(frame, reference, 10),
        "mean_top16_overlap": rank_overlap(frame, reference, 16),
        "mean_top37_overlap": rank_overlap(frame, reference, 37),
    }


def build_family(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    releases: dict[str, pd.DataFrame],
    known_private: np.ndarray,
    rare: np.ndarray,
    *,
    include_domias: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    main, variants, diagnostics = v4.build_mia_v4(
        candidates,
        background,
        releases,
        known_private,
        minimum=25,
        maximum=27,
        total=508,
        transport_epsilon=0.35,
        include_domias=include_domias,
        density_weight=v2.DEFAULT_MIA_DENSITY_WEIGHT,
        rare_mask=rare,
        rare_weight=v2.DEFAULT_MIA_RARE_WEIGHT,
    )
    daybreak, _, daybreak_diagnostics = build_mia_independent(
        candidates, background, releases, known_private
    )
    output = {
        "v4_main": main,
        "v3_reference": variants["v3_private_demotion_transport_reference"],
        "conformal": variants["v4_private_conformal_margin_0p50"],
        "daybreak_independent": daybreak,
    }
    return output, {"v4": diagnostics, "daybreak": daybreak_diagnostics}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (
        candidates,
        _b_self,
        actual_background,
        releases,
        known_private,
        rare,
        input_diagnostics,
    ) = load_inputs()
    old_v4 = pd.read_csv(TEAM / "attack-candidate-v4" / "F_mia.csv")
    old_v4 = old_v4.set_index("record_id").loc[candidates["record_id"]].reset_index()
    build_specs = {
        "actual_bg": (actual_background, False),
        "actual_bg_domias": (actual_background, True),
        "retained_toy_bg_control": (
            pd.read_csv(ROOT / "participant_data" / "A_bg.csv"),
            False,
        ),
    }
    rows: list[dict[str, Any]] = []
    build_diagnostics: dict[str, Any] = {}
    for background_label, (background, include_domias) in build_specs.items():
        family, diagnostics = build_family(
            candidates,
            background,
            releases,
            known_private,
            rare,
            include_domias=include_domias,
        )
        build_diagnostics[background_label] = diagnostics
        for variant, frame in family.items():
            validate_mia(frame, old_v4)
            result = official_validate.validate_mia_submission(
                frame, {}, official_validate.load_dist(TARGETS)
            )
            if not result.ok:
                raise RuntimeError(f"official validator failed: {result.errors}")
            path = OUT / f"F_mia_{background_label}_{variant}.csv"
            frame.to_csv(path, index=False)
            # Some builders retain sub-CSV-precision floating point differences
            # in memory.  The submitted artifact is the serialized CSV, so use
            # a re-read frame for changed-cell claims and retain the in-memory
            # comparison only as an explicitly labelled diagnostic.
            serialized_frame = pd.read_csv(path)
            serialized_comparison = comparison(serialized_frame, old_v4)
            in_memory_comparison = comparison(frame, old_v4)
            rows.append(
                {
                    "background": background_label,
                    "variant": variant,
                    "sha256": sha256(path),
                    "equals_historical_v4_hash": sha256(path)
                    == "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
                    **serialized_comparison,
                    "pre_serialization_changed_cells": in_memory_comparison[
                        "changed_cells"
                    ],
                    "pre_serialization_changed_rows": in_memory_comparison[
                        "changed_rows"
                    ],
                }
            )
    comparison_table = pd.DataFrame(rows)
    comparison_table.to_csv(OUT / "rebuild_comparison.csv", index=False)
    report = {
        "actual_inputs": input_diagnostics,
        "actual_private_rows": int(known_private.sum()),
        "actual_rare_rows": int(rare.sum()),
        "actual_private_rare_rows": int((known_private & rare).sum()),
        "actual_eligible_rare_rows": int((~known_private & rare).sum()),
        "historical_v4_sha256": sha256(
            TEAM / "attack-candidate-v4" / "F_mia.csv"
        ),
        "comparisons": comparison_table.to_dict(orient="records"),
        "build_diagnostics": build_diagnostics,
        "safety": {
            "token_read": False,
            "archive_created": False,
            "production_changed": False,
            "submission_performed": False,
        },
    }
    (OUT / "rebuild_report.json").write_text(
        json.dumps(v2.clean_json_v2(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
