"""Research-only evaluation of copula-space AIA time relinking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_SRC = REPO_ROOT / "starter" / "src"
ANALYSIS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, STARTER_SRC, ANALYSIS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_copula_warp as warp  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    QI_COLUMNS,
    _marginal_midrank_matrix,
    build_aia_v5_variants,
)
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import score_aia_power  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


def copula_links(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    selected: np.ndarray | None,
    assignment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    release_x = _marginal_midrank_matrix(release, QI_COLUMNS)
    challenge_x = _marginal_midrank_matrix(challenge, QI_COLUMNS)
    release_sex = release["sex"].to_numpy()
    release_pref = release["prefecture"].astype(str).to_numpy()
    challenge_sex = challenge["sex"].to_numpy()
    challenge_pref = challenge["prefecture"].astype(str).to_numpy()
    linked = np.full(len(challenge), -1, dtype=int)
    distance = np.full(len(challenge), np.inf, dtype=float)
    eligible = np.ones(len(challenge), dtype=bool) if selected is None else selected
    blocks = pd.DataFrame(
        {"sex": challenge_sex, "prefecture": challenge_pref}
    ).groupby(["sex", "prefecture"], sort=False).indices
    for raw_key, raw_rows in blocks.items():
        sex, prefecture = raw_key
        rows = np.asarray(raw_rows, dtype=int)
        rows = rows[eligible[rows]]
        if not len(rows):
            continue
        pool = np.flatnonzero((release_sex == sex) & (release_pref == str(prefecture)))
        if len(pool) < len(rows):
            pool = np.flatnonzero(release_sex == sex)
        if len(pool) < len(rows):
            pool = np.arange(len(release), dtype=int)
        costs = cdist(challenge_x[rows], release_x[pool], metric="euclidean")
        if assignment:
            row_index, column_index = linear_sum_assignment(costs)
            chosen_rows = rows[row_index]
            chosen_columns = pool[column_index]
            linked[chosen_rows] = chosen_columns
            distance[chosen_rows] = costs[row_index, column_index]
        else:
            column_index = np.argmin(costs, axis=1)
            linked[rows] = pool[column_index]
            distance[rows] = costs[np.arange(len(rows)), column_index]
    return linked, distance


def linked_frame(
    template: pd.DataFrame,
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    v4_frame: pd.DataFrame,
    mode: str,
    abstain: float,
    horizon: float,
) -> pd.DataFrame:
    block = template.sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
    v4_block = v4_frame.sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
    ordered_challenge = challenge.sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        block["challenge_row_id"].to_numpy(),
        ordered_challenge["challenge_row_id"].to_numpy(),
    ):
        raise ValueError("challenge alignment failure")
    selected = ~np.isclose(block["time_hat"].to_numpy(dtype=float), abstain)
    active_v4 = ~np.isclose(v4_block["time_hat"].to_numpy(dtype=float), abstain)
    if mode == "nearest_all":
        linked, _ = copula_links(release, ordered_challenge, selected=None, assignment=False)
    elif mode == "assignment_all":
        linked, _ = copula_links(release, ordered_challenge, selected=None, assignment=True)
    elif mode in {"nearest_selected", "nearest_new"}:
        linked, _ = copula_links(
            release, ordered_challenge, selected=selected, assignment=False
        )
    elif mode in {"assignment_selected", "assignment_new"}:
        linked, _ = copula_links(
            release, ordered_challenge, selected=selected, assignment=True
        )
    else:
        raise ValueError(mode)
    predicted = np.full(len(block), abstain, dtype=float)
    valid = selected & (linked >= 0)
    predicted[valid] = release["time"].to_numpy(dtype=float)[linked[valid]]
    if mode.endswith("_new"):
        keep = selected & active_v4
        predicted[keep] = v4_block.loc[keep, "time_hat"].to_numpy(dtype=float)
    predicted[selected & ~np.isfinite(predicted)] = horizon
    out = block.copy()
    out["time_hat"] = predicted
    return out


def main() -> None:
    dist = REPO_ROOT / "participant_data"
    target_dir = REPO_ROOT / "team-shared" / "prelim" / "attack-targets"
    out_dir = REPO_ROOT / "local-output" / "attack-shadow-evaluation-v6-copula-link"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    noise_scales = background[list(warp.VISIBLE_CONTINUOUS)].std(ddof=0).to_numpy(float)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target_id: pd.read_csv(target_dir / f"C_{target_id}.csv")
        for target_id in warp.PRODUCTION_IDS
    }
    spec = shadow.ShadowSpec(
        n_targets=20,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=508,
        minimum_shared=25,
        maximum_shared=26,
    )
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    abstain = v2.safe_abstain_time(horizon, tau)
    modes = (
        "nearest_all",
        "nearest_selected",
        "nearest_new",
        "assignment_all",
        "assignment_selected",
        "assignment_new",
    )
    rows: list[dict[str, object]] = []
    for seed in (20260909, 20260910, 20260911, 20260912):
        print(f"seed={seed}", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        ref_dir = out_dir / "reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)
        for noise_fraction in (0.0, 0.03, 0.10):
            scenario = f"public_marginal_warp_noise_{noise_fraction:.2f}"
            releases: dict[str, pd.DataFrame] = {}
            for index, (target_id, source) in enumerate(world.aia_sources.items()):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, index, int(target_id), 707, int(noise_fraction * 100)]
                    )
                )
                releases[target_id] = warp.rank_warp(
                    source,
                    templates[warp.PRODUCTION_IDS[index]],
                    noise_fraction=noise_fraction,
                    noise_scales=noise_scales,
                    rng=rng,
                    cfg=cfg,
                )
            baseline, _, _ = v4.build_aia_v4(
                world.attack_background,
                releases,
                world.challenges,
                main_fraction=0.5,
                variant_fractions=[0.5],
                abstain_time=abstain,
                horizon_time=horizon,
                link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
                baseline_abstain_time=shadow.choose_low_hit_time(
                    world.attack_background, horizon=horizon, tau=tau
                ),
                anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
                anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
                integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
            )
            independent, _, _ = build_aia_independent(
                world.attack_background,
                releases,
                world.challenges,
                member_count=spec.aia_members,
                horizon=horizon,
                abstain_time=abstain,
            )
            structured, _ = build_aia_v5_variants(
                baseline,
                independent,
                releases,
                world.challenges,
                member_count=spec.aia_members,
                abstain_time=abstain,
                horizon_time=horizon,
            )
            template = structured["consensus_v4_copula__horizon"]
            frames = {"v4_reference": baseline, "structured_horizon": template}
            for mode in modes:
                parts = []
                for target_id in sorted(releases, key=int):
                    target_template = template.loc[
                        template["target_k"].astype(int).eq(int(target_id))
                    ]
                    target_v4 = baseline.loc[
                        baseline["target_k"].astype(int).eq(int(target_id))
                    ]
                    parts.append(
                        linked_frame(
                            target_template,
                            releases[target_id],
                            world.challenges[target_id],
                            v4_frame=target_v4,
                            mode=mode,
                            abstain=abstain,
                            horizon=horizon,
                        )
                    )
                frames[f"copula_{mode}"] = pd.concat(parts, ignore_index=True)
            for name, frame in frames.items():
                result = score_aia_power(frame, ref_dir, cfg)
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "variant": name,
                        "power": float(result["power"]),
                        "all": float(result["all"]),
                        "rare": float(result["rare"]),
                        "p_m": float(result["p_m"]),
                        "p_c": float(result["p_c"]),
                        "per_target": json.dumps(result["per_target"], sort_keys=True),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "component_scores.csv", index=False)
    pivot = result.pivot_table(index=["seed", "scenario"], columns="variant", values="power")
    delta = pivot.sub(pivot["v4_reference"], axis=0)
    summary = delta.groupby("scenario").agg(["mean", "min", "max"])
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                scenario: {
                    variant: {
                        stat: float(summary.loc[scenario, (variant, stat)])
                        for stat in ("mean", "min", "max")
                    }
                    for variant in pivot.columns
                }
                for scenario in summary.index
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.round(6).to_string(), flush=True)


if __name__ == "__main__":
    main()
