"""Research-only target-wise AIA evaluation for positive production targets.

This script deliberately lives under ``local-output``.  It evaluates fixed,
predeclared selectors and copula-space time decoders in truth-known shadow
worlds; it never writes a submission candidate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
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
    PROXIMITY_VIEWS,
    _marginal_midrank_matrix,
    copula_proximity_score,
    proximity_score,
    stratified_half_selection,
)
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_aia  # noqa: E402


PRODUCTION_IDS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
POSITIVE_IDS = frozenset(("3", "4", "5", "10", "12", "13", "22", "23", "26", "29"))
SHADOW_TO_PRODUCTION = {
    str(index + 1): target_id for index, target_id in enumerate(PRODUCTION_IDS)
}
SELECTED_SHADOW_IDS = tuple(
    key for key, value in SHADOW_TO_PRODUCTION.items() if value in POSITIVE_IDS
)
VISIBLE = ("age", "SBP", "TG", "HDL", "ALT")
ABSTAIN = 11.0
HORIZON = 10.0

# Fixed before looking at the new results.  Scale is material: expert indicator
# gaps determine preference classes, while proximity breaks ties within them.
SELECTORS = {
    "v4_raw": (1.00, 0.00, 0.30, 0.00),
    "v4_cop": (1.00, 0.00, 0.00, 0.30),
    "v4_mix": (1.00, 0.00, 0.15, 0.15),
    "ind_raw": (0.00, 1.00, 0.30, 0.00),
    "ind_cop": (0.00, 1.00, 0.00, 0.30),
    "cons_v4_raw": (1.10, 0.90, 0.35, 0.00),
    "cons_v4_mix": (1.10, 0.90, 0.20, 0.20),
    "cons_v4_cop": (1.10, 0.90, 0.00, 0.40),
    "cons_eq_mix": (1.00, 1.00, 0.20, 0.20),
    "cons_eq_cop": (1.00, 1.00, 0.00, 0.40),
    "cons_ind_mix": (0.90, 1.10, 0.20, 0.20),
    "soft_cop": (0.60, 0.60, 0.00, 1.00),
    "cop_only": (0.00, 0.00, 0.00, 1.00),
}


def rank_warp(
    source: pd.DataFrame,
    template: pd.DataFrame,
    *,
    noise_fraction: float,
    noise_scales: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
) -> pd.DataFrame:
    out = source.copy()
    if noise_fraction:
        noise = rng.normal(
            0.0,
            noise_fraction * noise_scales,
            size=(len(out), len(VISIBLE)),
        )
        for index, column in enumerate(VISIBLE):
            out[column] = out[column].to_numpy(dtype=float) + noise[:, index]
        out["time"] = out["time"].to_numpy(dtype=float) + rng.normal(
            0.0, noise_fraction * HORIZON, size=len(out)
        )

    quantile = (np.arange(len(out), dtype=float) + 0.5) / len(out)
    for column in VISIBLE:
        values = out[column].to_numpy(dtype=float)
        order = np.argsort(values, kind="mergesort")
        warped_sorted = np.quantile(
            template[column].to_numpy(dtype=float), quantile, method="linear"
        )
        warped = np.empty(len(out), dtype=float)
        warped[order] = warped_sorted
        out[column] = warped
    out = shadow.coerce_release(out, cfg)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def aligned_times(frame: pd.DataFrame, target_id: str, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(int(target_id))].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def _block_pools(reference: pd.DataFrame, query: pd.DataFrame):
    ref_sex = reference["sex"].to_numpy()
    ref_pref = reference["prefecture"].astype(str).to_numpy()
    query_groups = query.groupby(["sex", "prefecture"], sort=False, dropna=False).indices
    all_rows = np.arange(len(reference), dtype=int)
    for raw_key, raw_rows in query_groups.items():
        sex, pref = raw_key if isinstance(raw_key, tuple) else (raw_key, "")
        query_rows = np.asarray(raw_rows, dtype=int)
        pool = np.flatnonzero((ref_sex == sex) & (ref_pref == str(pref)))
        if not len(pool):
            pool = np.flatnonzero(ref_sex == sex)
        if not len(pool):
            pool = all_rows
        yield query_rows, pool


def copula_links(released: pd.DataFrame, challenge: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build reusable nearest, mutual, consensus and unique copula links."""
    full_r = _marginal_midrank_matrix(released, VISIBLE)
    full_q = _marginal_midrank_matrix(challenge, VISIBLE)
    view_r = [_marginal_midrank_matrix(released, cols) for cols in PROXIMITY_VIEWS]
    view_q = [_marginal_midrank_matrix(challenge, cols) for cols in PROXIMITY_VIEWS]

    nearest = np.empty(len(challenge), dtype=int)
    second_margin = np.empty(len(challenge), dtype=float)
    mutual = np.zeros(len(challenge), dtype=bool)
    consensus = np.empty(len(challenge), dtype=int)
    agreement = np.empty(len(challenge), dtype=float)
    unique = np.empty(len(challenge), dtype=int)

    for query_rows, pool in _block_pools(released, challenge):
        pair = cdist(full_q[query_rows], full_r[pool], metric="euclidean")
        order = np.argsort(pair, axis=1, kind="mergesort")
        nearest_local = order[:, 0]
        nearest[query_rows] = pool[nearest_local]
        # Some rare public blocks contain fewer released rows than challenge
        # rows.  Seed the unique decoder with nearest links, then overwrite all
        # rows covered by the rectangular assignment.
        unique[query_rows] = pool[nearest_local]
        if pair.shape[1] > 1:
            second_margin[query_rows] = (
                np.take_along_axis(pair, order[:, 1:2], axis=1)[:, 0]
                - pair[np.arange(len(query_rows)), nearest_local]
            )
        else:
            second_margin[query_rows] = np.inf

        reverse = np.argmin(pair, axis=0)
        mutual[query_rows] = reverse[nearest_local] == np.arange(len(query_rows))

        view_choice = np.empty((len(query_rows), len(PROXIMITY_VIEWS)), dtype=int)
        for view_index in range(len(PROXIMITY_VIEWS)):
            view_pair = cdist(
                view_q[view_index][query_rows],
                view_r[view_index][pool],
                metric="euclidean",
            )
            view_choice[:, view_index] = pool[np.argmin(view_pair, axis=1)]
        for local_row, choices in enumerate(view_choice):
            counts = Counter(int(value) for value in choices)
            best_count = max(counts.values())
            finalists = [value for value, count in counts.items() if count == best_count]
            global_query = int(query_rows[local_row])
            if len(finalists) == 1:
                chosen = finalists[0]
            else:
                chosen = min(
                    finalists,
                    key=lambda value: float(
                        np.linalg.norm(full_q[global_query] - full_r[value])
                    ),
                )
            consensus[global_query] = chosen
            agreement[global_query] = best_count / len(PROXIMITY_VIEWS)

        local_q, local_r = linear_sum_assignment(pair)
        unique[query_rows[local_q]] = pool[local_r]

    margin_rank = pd.Series(second_margin).rank(method="average", pct=True).to_numpy()
    return {
        "nearest": nearest,
        "consensus": consensus,
        "unique": unique,
        "mutual": mutual,
        "agreement": agreement,
        "margin_rank": margin_rank,
    }


def prediction_from_link(released: pd.DataFrame, linked: np.ndarray) -> np.ndarray:
    release_time = released["time"].to_numpy(dtype=float)[linked]
    release_event = (
        released["onset"].to_numpy(dtype=int)[linked]
        + released["death"].to_numpy(dtype=int)[linked]
    ) > 0
    return np.where(release_event, release_time, HORIZON)


def decoded_variants(
    selected: np.ndarray,
    v4_time: np.ndarray,
    ind_time: np.ndarray,
    released: pd.DataFrame,
    links: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    active_v4 = ~np.isclose(v4_time, ABSTAIN)
    active_ind = ~np.isclose(ind_time, ABSTAIN)
    v4_only = np.where(active_v4, v4_time, HORIZON)
    ind_fb = np.where(active_ind, ind_time, np.where(active_v4, v4_time, HORIZON))
    v4_fb = np.where(active_v4, v4_time, np.where(active_ind, ind_time, HORIZON))
    linked_predictions = {
        name: prediction_from_link(released, links[name])
        for name in ("nearest", "consensus", "unique")
    }

    chosen: dict[str, np.ndarray] = {
        "v4_only": v4_only,
        "ind_fb": ind_fb,
        "v4_fb": v4_fb,
        "horizon": np.full(len(selected), HORIZON),
    }
    for link_name, copula_time in linked_predictions.items():
        chosen[f"cop_{link_name}_all"] = copula_time
        chosen[f"v4_new_{link_name}"] = np.where(active_v4, v4_time, copula_time)
        chosen[f"ind_new_{link_name}"] = np.where(active_ind, ind_time, copula_time)

    # High-precision event overrides.  Non-event links already equal the horizon,
    # so gates only decide whether to trust an event time.
    nearest_time = linked_predictions["nearest"]
    release_event = nearest_time < HORIZON - 1e-9
    gates = {
        "mutual": links["mutual"],
        "agree": links["agreement"] >= (2.0 / 3.0),
        "strict": links["mutual"] & (links["agreement"] >= (2.0 / 3.0)),
        "margin": links["mutual"] & (links["margin_rank"] >= 0.50),
    }
    for gate_name, gate in gates.items():
        override = release_event & gate
        chosen[f"v4_event_{gate_name}"] = np.where(override, nearest_time, v4_only)
        chosen[f"ind_event_{gate_name}"] = np.where(override, nearest_time, ind_fb)

    outputs: dict[str, np.ndarray] = {}
    for name, values in chosen.items():
        output = np.full(len(selected), ABSTAIN, dtype=float)
        output[selected] = values[selected]
        outputs[name] = output
    return outputs


def score_times(
    time_hat: np.ndarray,
    challenge_ids: np.ndarray,
    truth: pd.DataFrame,
    cfg: dict,
) -> dict[str, float]:
    indexed = truth.set_index(truth["challenge_row_id"].astype(int)).loc[challenge_ids]
    member = indexed["label"].to_numpy() == "member"
    control = ~member
    onset = indexed["onset"].to_numpy(dtype=int)
    time = indexed["time"].to_numpy(dtype=float)
    rare = indexed["is_rare"].to_numpy(dtype=bool)
    result = score_aia(
        None,
        time_hat[member],
        onset[member],
        time[member],
        None,
        time_hat[control],
        onset[control],
        time[control],
        cfg,
        member_rare_mask=rare[member],
        control_rare_mask=rare[control],
    )
    return {
        "score": float(result.score),
        "all": float(result.breakdown["r_time_all"]),
        "rare": float(result.breakdown["r_time_rare"]),
        "p_m": float(result.breakdown["succ_time_member"]),
        "p_c": float(result.breakdown["succ_time_control"]),
    }


def main() -> None:
    output_dir = ROOT / "local-output" / "positive-target-copula-link"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(ROOT / "participant_data" / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(ROOT / "participant_data" / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(ROOT / "participant_data")
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target_id: pd.read_csv(
            ROOT / "team-shared" / "prelim" / "attack-targets" / f"C_{target_id}.csv"
        )
        for target_id in POSITIVE_IDS
    }
    noise_scales = background.loc[:, VISIBLE].std(ddof=0).to_numpy(dtype=float)
    spec = shadow.ShadowSpec(
        n_targets=20,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=508,
        minimum_shared=25,
        maximum_shared=26,
    )
    rows: list[dict] = []
    seeds = tuple(range(20260909, 20260917))
    scenarios = ("raw", "noise_0p03", "noise_0p10", "warp", "warp_noise_0p03")

    for seed_index, seed in enumerate(seeds):
        print(f"seed={seed}", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        source_subset = {key: world.aia_sources[key] for key in SELECTED_SHADOW_IDS}
        challenge_subset = {key: world.challenges[key] for key in SELECTED_SHADOW_IDS}
        truth_subset = {key: world.aia_truth[key] for key in SELECTED_SHADOW_IDS}

        for scenario_index, scenario in enumerate(scenarios):
            print(f"  scenario={scenario}", flush=True)
            releases: dict[str, pd.DataFrame] = {}
            for key, source in source_subset.items():
                production_id = SHADOW_TO_PRODUCTION[key]
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, seed_index, scenario_index, int(key), 706]
                    )
                )
                if scenario.startswith("warp"):
                    releases[key] = rank_warp(
                        source,
                        templates[production_id],
                        noise_fraction=0.03 if "noise" in scenario else 0.0,
                        noise_scales=noise_scales,
                        rng=rng,
                        cfg=cfg,
                    )
                else:
                    releases[key] = shadow.scenario_release(
                        source, world.attack_background, scenario, cfg, rng
                    )

            baseline, _, _ = v4.build_aia_v4(
                world.attack_background,
                releases,
                challenge_subset,
                main_fraction=0.5,
                variant_fractions=[0.5],
                abstain_time=ABSTAIN,
                horizon_time=HORIZON,
                link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
                baseline_abstain_time=shadow.choose_low_hit_time(
                    world.attack_background,
                    horizon=HORIZON,
                    tau=float(cfg["aia"]["tau_time"]),
                ),
                anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
                anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
                integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
            )
            independent, _, _ = build_aia_independent(
                world.attack_background,
                releases,
                challenge_subset,
                member_count=spec.aia_members,
                horizon=HORIZON,
                abstain_time=ABSTAIN,
            )

            for key in SELECTED_SHADOW_IDS:
                production_id = SHADOW_TO_PRODUCTION[key]
                challenge = challenge_subset[key].reset_index(drop=True)
                released = releases[key].reset_index(drop=True)
                ids = challenge["challenge_row_id"].to_numpy(dtype=int)
                v4_time = aligned_times(baseline, key, ids)
                ind_time = aligned_times(independent, key, ids)
                active_v4 = ~np.isclose(v4_time, ABSTAIN)
                active_ind = ~np.isclose(ind_time, ABSTAIN)
                raw = proximity_score(released, challenge)
                copula = copula_proximity_score(released, challenge)
                links = copula_links(released, challenge)

                base_times = {"v4_reference": v4_time, "ind_reference": ind_time}
                for variant_name, time_hat in base_times.items():
                    score = score_times(time_hat, ids, truth_subset[key], cfg)
                    rows.append(
                        {
                            "seed": seed,
                            "split": "discovery" if seed <= 20260912 else "validation",
                            "scenario": scenario,
                            "target": production_id,
                            "selector": "reference",
                            "decoder": variant_name,
                            **score,
                        }
                    )

                for selector_name, weights in SELECTORS.items():
                    wv4, wind, wraw, wcop = weights
                    selection_score = (
                        wv4 * active_v4
                        + wind * active_ind
                        + wraw * raw
                        + wcop * copula
                    )
                    selected, _ = stratified_half_selection(
                        challenge, selection_score, member_count=spec.aia_members
                    )
                    variants = decoded_variants(
                        selected, v4_time, ind_time, released, links
                    )
                    for decoder_name, time_hat in variants.items():
                        score = score_times(time_hat, ids, truth_subset[key], cfg)
                        rows.append(
                            {
                                "seed": seed,
                                "split": "discovery" if seed <= 20260912 else "validation",
                                "scenario": scenario,
                                "target": production_id,
                                "selector": selector_name,
                                "decoder": decoder_name,
                                **score,
                            }
                        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "component_scores.csv", index=False)

    reference = result[result["selector"].eq("reference")].pivot_table(
        index=["seed", "split", "scenario", "target"],
        columns="decoder",
        values="score",
    )
    reference["expert_winner"] = reference.max(axis=1)
    variants = result[~result["selector"].eq("reference")].copy()
    variants = variants.merge(
        reference[["v4_reference", "ind_reference", "expert_winner"]].reset_index(),
        on=["seed", "split", "scenario", "target"],
        how="left",
        validate="many_to_one",
    )
    variants["delta_v4"] = variants["score"] - variants["v4_reference"]
    variants["delta_winner"] = variants["score"] - variants["expert_winner"]
    variants["variant"] = variants["selector"] + "__" + variants["decoder"]
    variants.to_csv(output_dir / "variant_deltas.csv", index=False)

    summary = (
        variants.groupby(["split", "scenario", "target", "variant"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_delta_v4=("delta_v4", "mean"),
            mean_delta_winner=("delta_winner", "mean"),
            min_delta_winner=("delta_winner", "min"),
            wins_winner=("delta_winner", lambda value: int((value > 1e-12).sum())),
            losses_winner=("delta_winner", lambda value: int((value < -1e-12).sum())),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    metadata = {
        "purpose": "truth-known target-wise method comparison; not an LB prediction",
        "seeds": list(seeds),
        "discovery_seeds": list(seeds[:4]),
        "validation_seeds": list(seeds[4:]),
        "scenarios": list(scenarios),
        "positive_production_targets": sorted(POSITIVE_IDS, key=int),
        "selectors": SELECTORS,
        "submission_performed": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"rows={len(result)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
