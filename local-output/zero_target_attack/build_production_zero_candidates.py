"""Build research-only zero-target candidate blocks under local-output."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_phase_v5.aia_structured_ensemble import (
    build_aia_v5_variants,
    filter_linkable_releases,
)


TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "zero_target_attack" / "production_candidates"


def load_research_module():
    path = ROOT / "local-output" / "zero_target_attack" / "evaluate_zero_variants.py"
    spec = importlib.util.spec_from_file_location("zero_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load research evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_releases = {
        path.stem.split("_")[1]: pd.read_csv(path)
        for path in DATA.glob("C_*.csv")
    }
    filtered, filter_diagnostics = filter_linkable_releases(all_releases)
    releases = {str(k): filtered[str(k)] for k in TARGETS}
    all_challenges = {
        path.stem.split("_")[-1]: pd.read_csv(path)
        for path in DATA.glob("aia_challenge_*.csv")
    }
    challenges = {
        str(k): pd.read_csv(DATA / f"aia_challenge_{k}.csv") for k in TARGETS
    }
    v4 = pd.read_csv(
        ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v4" / "F_aia.csv"
    )
    independent = pd.read_csv(
        ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v5-probe-independent"
        / "F_aia.csv"
    )
    attack_target_ids = {str(value) for value in v4["target_k"].astype(int).unique()}
    build_releases = {
        key: value for key, value in all_releases.items() if key in attack_target_ids
    }
    build_challenges = {
        key: value for key, value in all_challenges.items() if key in attack_target_ids
    }
    reconstructed_structured, _ = build_aia_v5_variants(
        v4,
        independent,
        build_releases,
        build_challenges,
        member_count=163,
        abstain_time=11.0,
        horizon_time=10.0,
    )
    original_current = reconstructed_structured["consensus_v4_copula__horizon"]
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    evaluator = load_research_module()
    variants, details = evaluator.build_variants(
        background,
        releases,
        challenges,
        v4,
        independent,
        {"consensus_v4_copula__horizon": original_current},
        abstain=11.0,
        horizon=10.0,
    )
    keep = (
        "current_horizon",
        "current_copula_censor_time",
        "current_changed_only_copula",
        "global_density_all5_horizon",
        "global_density_ensemble_horizon",
        "global_density_stable3_horizon",
        "global_density_consensus_horizon",
        "global_density_consensus_censor_time",
        "sex_consensus_horizon",
        "sex_pure_horizon",
        "sex_bgcdf_horizon",
        "sex_consensus_censor_time",
    )
    for name in keep:
        variants[name].to_csv(OUT / f"{name}.csv", index=False)

    base = variants["current_horizon"].set_index(["target_k", "challenge_row_id"])
    comparison: list[dict[str, object]] = []
    for name in keep[1:]:
        other = variants[name].set_index(["target_k", "challenge_row_id"])
        for target in TARGETS:
            left = base.loc[target, "time_hat"].to_numpy(dtype=float)
            right = other.loc[target, "time_hat"].to_numpy(dtype=float)
            comparison.append(
                {
                    "target": target,
                    "variant": name,
                    "changed_rows": int(np.count_nonzero(~np.isclose(left, right))),
                    "active_changes": int(
                        np.count_nonzero(np.isclose(left, 11.0) != np.isclose(right, 11.0))
                    ),
                    "event_predictions": int(np.count_nonzero((right < 9.5) & ~np.isclose(right, 11.0))),
                }
            )
    pd.DataFrame(comparison).to_csv(OUT / "candidate_differences.csv", index=False)
    (OUT / "diagnostics.json").write_text(
        json.dumps(
            {"filter": filter_diagnostics, "build": details},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        pd.DataFrame(comparison)
        .query("variant in ['global_density_all5_horizon','global_density_stable3_horizon','global_density_consensus_horizon','sex_consensus_horizon','sex_consensus_censor_time']")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
