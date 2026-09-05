"""Research-only AIA decoder and matched-pair sensitivity study.

This file writes only beneath ``local-output`` and never builds a submission.
It conditions the comparison on shadow target blocks where the corresponding
production probe direction scores exactly zero, matching the official feedback
available for targets 6, 7, 8, 9, 17, 18, and 19.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
    stratified_half_selection,
)
from attack_phase_v5.evaluate_shadow_v5 import (  # noqa: E402
    exactify_visible_control_matching,
)
from attack_phase_v6.aia_global_density import (  # noqa: E402
    global_density_ratio_score,
    sex_copula_proximity_score,
)
from local_output_import import load_zero_helpers  # type: ignore  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.scoring.attack_score import score_aia  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


TARGETS = (6, 7, 8, 9, 17, 18, 19)
CURRENT_TARGETS = {6, 8, 17}
HORIZONS = (9.498631074607, 9.5, 9.75, 10.0)
ABSTAIN = 11.0
OUT = ROOT / "local-output" / "attack-v6" / "novel-aia"
VISIBLE_NUMERIC = ("age", "SBP", "TG", "HDL", "ALT")
VISIBLE_CATEGORICAL = ("sex", "prefecture")


def group_indices(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[np.ndarray]:
    return [
        np.asarray(rows, dtype=int)
        for rows in frame.groupby(list(columns), sort=True, dropna=False).indices.values()
    ]


def age_band(values: pd.Series) -> np.ndarray:
    return np.floor((values.to_numpy(dtype=float) - 40.0) / 5.0).astype(int)


def pair_project(challenge: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    """Project a selector to one row per adjacent-age inferred matched pair."""
    work = challenge.copy()
    work["_band"] = age_band(work["age"])
    ids = work["challenge_row_id"].to_numpy(dtype=int)
    ages = work["age"].to_numpy(dtype=float)
    selected = np.asarray(selected, dtype=bool)
    output = np.zeros(len(work), dtype=bool)
    for rows in group_indices(work, ("sex", "prefecture", "_band")):
        if len(rows) % 2:
            raise ValueError("published matched strata must have even row counts")
        order = rows[np.lexsort((ids[rows], ages[rows]))]
        for offset in range(0, len(order), 2):
            pair = order[offset : offset + 2]
            if selected[pair].sum() == 1:
                winner = pair[selected[pair]][0]
            else:
                winner = pair[np.argmin(ids[pair])]
            output[winner] = True
    if output.sum() != len(output) // 2:
        raise RuntimeError("pair projection did not select exactly half")
    return output


def fit_rare_model(
    background: pd.DataFrame, detector: dict[str, object], *, seed: int
):
    """Fit a public-QI rare proxy without using any challenge truth."""
    labels = shadow.public_rare_mask(background, detector).astype(int)
    transformer = ColumnTransformer(
        [
            ("numeric", RobustScaler(), list(VISIBLE_NUMERIC)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(VISIBLE_CATEGORICAL),
            ),
        ]
    )
    model = make_pipeline(
        transformer,
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    )
    model.fit(background[[*VISIBLE_NUMERIC, *VISIBLE_CATEGORICAL]], labels)
    return model


def rare_quota_project(
    selected: np.ndarray,
    final_score: np.ndarray,
    rare_probability: np.ndarray,
    challenge_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the published 13/13 split inside the 26 most rare-like rows."""
    selected = np.asarray(selected, dtype=bool)
    rare_order = np.lexsort((challenge_ids, -rare_probability))
    rare_guess = np.zeros(len(selected), dtype=bool)
    rare_guess[rare_order[:26]] = True
    rows = np.flatnonzero(rare_guess)
    member_order = rows[np.lexsort((challenge_ids[rows], -final_score[rows]))]
    output = selected.copy()
    output[rows] = False
    output[member_order[:13]] = True
    return output, rare_guess


def target_result(
    challenge: pd.DataFrame,
    truth: pd.DataFrame,
    time_hat: np.ndarray,
    cfg: dict,
):
    indexed = truth.set_index("challenge_row_id").loc[
        challenge["challenge_row_id"].to_numpy(dtype=int)
    ]
    member = indexed["label"].eq("member").to_numpy()
    control = ~member
    rare = indexed["is_rare"].astype(bool).to_numpy()
    return score_aia(
        None,
        time_hat[member],
        indexed.loc[member, "onset"].to_numpy(dtype=int),
        indexed.loc[member, "time"].to_numpy(dtype=float),
        None,
        time_hat[control],
        indexed.loc[control, "onset"].to_numpy(dtype=int),
        indexed.loc[control, "time"].to_numpy(dtype=float),
        cfg,
        member_rare_mask=rare[member],
        control_rare_mask=rare[control],
    )


def target_power(
    challenge: pd.DataFrame,
    truth: pd.DataFrame,
    time_hat: np.ndarray,
    cfg: dict,
) -> float:
    return float(target_result(challenge, truth, time_hat, cfg).score)


def decoded(selected: np.ndarray, active_time: float) -> np.ndarray:
    output = np.full(len(selected), ABSTAIN, dtype=float)
    output[selected] = active_time
    return output


def production_selector(
    template_target: int,
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    active_v4: np.ndarray,
    active_independent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probe selection, final complement selection, and linkable release."""
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    # A single-target mapping is sufficient here: filtering was already applied
    # jointly by the caller, and this function receives that target's result.
    if template_target in CURRENT_TARGETS:
        prox = proximity_score(release, challenge)
        cop = copula_proximity_score(release, challenge)
        score = 1.10 * active_v4 + 0.90 * active_independent + 0.20 * prox + 0.20 * cop
    elif template_target == 7:
        score = global_density_ratio_score(release, challenge, background)
    elif template_target == 9:
        score = global_density_ratio_score(
            release, challenge, background, columns=("age", "SBP", "TG")
        )
    elif template_target == 18:
        prox = sex_copula_proximity_score(release, challenge)
        score = 1.10 * active_v4 + 0.90 * active_independent + 0.55 * prox
    elif template_target == 19:
        ordinary = global_density_ratio_score(
            release, challenge, background, columns=("age", "SBP", "TG")
        )
        score = -ordinary
    else:
        raise ValueError(template_target)
    probe, _ = stratified_half_selection(challenge, score, member_count=163)
    final = ~probe
    # For target 19, probe was the inverted density direction, so complementing
    # correctly recovers the ordinary direction just as the final ZIP does.
    return probe, final, score


def run_case(
    *,
    world: shadow.ShadowWorld,
    releases: dict[str, pd.DataFrame],
    background: pd.DataFrame,
    cfg: dict,
    template_target: int,
    seed: int,
    scenario: str,
    rare_model,
) -> list[dict[str, object]]:
    baseline, _, _ = v4.build_aia_v4(
        background,
        releases,
        world.challenges,
        main_fraction=0.5,
        variant_fractions=[0.5],
        abstain_time=ABSTAIN,
        horizon_time=10.0,
        link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
        baseline_abstain_time=shadow.choose_low_hit_time(background, horizon=10.0, tau=0.5),
        anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    independent, _, _ = build_aia_independent(
        background,
        releases,
        world.challenges,
        member_count=163,
        horizon=10.0,
        abstain_time=ABSTAIN,
    )
    linkable, _ = filter_linkable_releases(releases)
    rows: list[dict[str, object]] = []
    for target_id in sorted(world.challenges, key=int):
        challenge = world.challenges[target_id].reset_index(drop=True)
        truth = world.aia_truth[target_id].reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        base = baseline.loc[baseline["target_k"].astype(int).eq(int(target_id))].set_index(
            "challenge_row_id"
        )
        ind = independent.loc[
            independent["target_k"].astype(int).eq(int(target_id))
        ].set_index("challenge_row_id")
        active_v4 = ~np.isclose(base.loc[ids, "time_hat"].to_numpy(dtype=float), ABSTAIN)
        active_ind = ~np.isclose(ind.loc[ids, "time_hat"].to_numpy(dtype=float), ABSTAIN)
        probe, final, probe_score = production_selector(
            template_target,
            linkable[target_id],
            challenge,
            background,
            active_v4,
            active_ind,
        )
        probe_power = target_power(challenge, truth, decoded(probe, 10.0), cfg)
        final_pair = pair_project(challenge, final)
        rare_probability = rare_model.predict_proba(
            challenge[[*VISIBLE_NUMERIC, *VISIBLE_CATEGORICAL]]
        )[:, 1]
        final_rare, rare_guess = rare_quota_project(
            final,
            -np.asarray(probe_score, dtype=float),
            rare_probability,
            ids,
        )
        indexed_truth = truth.set_index("challenge_row_id").loc[ids]
        true_rare = indexed_truth["is_rare"].astype(bool).to_numpy()
        rare_tp = int(np.count_nonzero(rare_guess & true_rare))
        linked_time, linked_gate, _ = copula_event_links(
            linkable[target_id], challenge, margin_quantile=0.75, horizon_time=10.0
        )
        for projection_name, selection in (
            ("stratum_half", final),
            ("age_pair", final_pair),
            ("rare26_quota", final_rare),
        ):
            for horizon in HORIZONS:
                basic = decoded(selection, horizon)
                gated = basic.copy()
                use_link = selection & linked_gate
                gated[use_link] = linked_time[use_link]
                for decoder_name, values in (("constant", basic), ("event_q75", gated)):
                    result = target_result(challenge, truth, values, cfg)
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "template_target": template_target,
                            "shadow_target": int(target_id),
                            "probe_power": probe_power,
                            "probe_zero": probe_power <= 1e-15,
                            "projection": projection_name,
                            "active_time": horizon,
                            "decoder": decoder_name,
                            "final_power": float(result.score),
                            "final_all": float(result.breakdown["r_time_all"]),
                            "final_rare": float(result.breakdown["r_time_rare"]),
                            "final_p_m": float(result.breakdown["succ_time_member"]),
                            "final_p_c": float(result.breakdown["succ_time_control"]),
                            "final_p_m_rare": float(
                                result.breakdown["succ_time_member_rare"]
                            ),
                            "final_p_c_rare": float(
                                result.breakdown["succ_time_control_rare"]
                            ),
                            "selection_changes_from_plain_complement": int(
                                np.count_nonzero(selection != final)
                            ),
                            "event_overrides": int(np.count_nonzero(use_link)),
                            "rare_guess_tp": rare_tp,
                            "rare_guess_precision": rare_tp / 26.0,
                            "rare_guess_recall": rare_tp / int(true_rare.sum()),
                        }
                    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background_full = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target: pd.read_csv(target_dir / f"C_{target}.csv") for target in TARGETS
    }
    noise_scales = background_full.loc[:, ["age", "SBP", "TG", "HDL", "ALT"]].std(
        ddof=0
    ).to_numpy(dtype=float)
    helpers = load_zero_helpers(ROOT)
    scenario_releases = helpers["scenario_releases"]
    spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )
    all_rows: list[dict[str, object]] = []
    for seed in (20261021, 20261022):
        world = shadow.build_shadow_world(
            candidates, b_self, background_full, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        background = world.attack_background
        rare_model = fit_rare_model(background, detector, seed=seed)
        for template_target in TARGETS:
            for base_scenario in ("warp", "warp_noise_0p03", "warp_time_shuffled"):
                label = f"{base_scenario}_C{template_target}"
                print(f"seed={seed} {label}", flush=True)
                releases = scenario_releases(
                    world,
                    base_scenario,
                    seed=seed,
                    cfg=cfg,
                    template=templates[template_target],
                    noise_scales=noise_scales,
                )
                all_rows.extend(
                    run_case(
                        world=world,
                        releases=releases,
                        background=background,
                        cfg=cfg,
                        template_target=template_target,
                        seed=seed,
                        scenario=label,
                        rare_model=rare_model,
                    )
                )
    result = pd.DataFrame(all_rows)
    result.to_csv(OUT / "horizon_pairing_scores.csv", index=False)
    zero = result.loc[result["probe_zero"]].copy()
    summary = (
        zero.groupby(
            ["template_target", "projection", "active_time", "decoder"], as_index=False
        )
        .agg(
            cases=("final_power", "size"),
            power_mean=("final_power", "mean"),
            power_median=("final_power", "median"),
            power_min=("final_power", "min"),
            power_max=("final_power", "max"),
            pair_changes_mean=("selection_changes_from_plain_complement", "mean"),
            event_overrides_mean=("event_overrides", "mean"),
        )
        .sort_values(["template_target", "power_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT / "horizon_pairing_summary.csv", index=False)
    report = {
        "scope": "authorized local shadow evaluation only",
        "conditioning": "only shadow target blocks where the production-probe direction scored zero",
        "rows": len(result),
        "conditioned_rows": len(zero),
        "conditioned_target_cases": int(
            zero[["seed", "scenario", "shadow_target"]].drop_duplicates().shape[0]
        ),
        "best_by_template": summary.groupby("template_target", as_index=False)
        .head(1)
        .to_dict(orient="records"),
    }
    (OUT / "horizon_pairing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.groupby("template_target").head(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
