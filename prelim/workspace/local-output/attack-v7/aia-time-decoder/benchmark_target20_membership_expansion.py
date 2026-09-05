"""Compare fixed-mask and expanded-mask target-20 copula overlays."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOVEL = ROOT / "local-output" / "attack-v6" / "novel-aia"
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, NOVEL, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import evaluate_positive_shift as positive  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    filter_linkable_releases,
)
from attack_phase_v5.evaluate_shadow_v5 import (  # noqa: E402
    exactify_visible_control_matching,
)
from local_output_import import load_zero_helpers  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


def load_time_benchmark() -> Any:
    spec = importlib.util.spec_from_file_location(
        "aia_time_positive_benchmark", HERE / "benchmark_positive_decoders.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load positive time benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGET = 20
SEEDS = {
    "discovery": (20270101, 20270102),
    "validation": (20270111, 20270112),
}


def main() -> None:
    bench = load_time_benchmark()
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background_full = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    template = pd.read_csv(target_dir / f"C_{TARGET}.csv")
    noise_scales = background_full.loc[:, list(bench.core.VISIBLE_NUMERIC)].std(
        ddof=0
    ).to_numpy(dtype=float)
    helper = load_zero_helpers(ROOT)["scenario_releases"]
    shadow_spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )

    rows: list[dict[str, Any]] = []
    for split, seeds in SEEDS.items():
        for seed in seeds:
            world = shadow.build_shadow_world(
                candidates, b_self, background_full, detector, shadow_spec, seed
            )
            exactify_visible_control_matching(world)
            shadow.validate_world(world, shadow_spec)
            for scenario in bench.SCENARIOS:
                print(f"{split} seed={seed} {scenario}", flush=True)
                releases = bench.make_releases(
                    helper,
                    world,
                    scenario,
                    seed=seed,
                    cfg=cfg,
                    template=template,
                    noise_scales=noise_scales,
                )
                frames = positive.build_base_frames(
                    world, releases, world.attack_background, cfg, TARGET
                )
                linkable, _ = filter_linkable_releases(releases)
                for target_id, base_values in frames.items():
                    challenge = world.challenges[target_id].reset_index(drop=True)
                    truth = world.aia_truth[target_id].reset_index(drop=True)
                    # Match the production v6 horizon-edge candidate exactly.
                    baseline = np.asarray(base_values, dtype=float).copy()
                    baseline[np.isclose(baseline, bench.HORIZON)] = 9.5
                    linked_time, linked_gate, _ = copula_event_links(
                        linkable[target_id],
                        challenge,
                        margin_quantile=0.50,
                        horizon_time=bench.HORIZON,
                    )
                    active = ~np.isclose(base_values, bench.ABSTAIN)
                    variants = {"edge": baseline}
                    fixed = baseline.copy()
                    fixed[active & linked_gate] = linked_time[active & linked_gate]
                    variants["fixed_mask"] = fixed
                    expanded = baseline.copy()
                    expanded[linked_gate] = linked_time[linked_gate]
                    variants["expanded_mask"] = expanded
                    for decoder, values in variants.items():
                        rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "scenario": scenario,
                                "shadow_target": int(target_id),
                                "decoder": decoder,
                                "baseline_active": int(active.sum()),
                                "trusted_links": int(linked_gate.sum()),
                                "applied_links": int(
                                    ((active & linked_gate) if decoder == "fixed_mask" else linked_gate).sum()
                                ) if decoder != "edge" else 0,
                                "added_active": int((~active & linked_gate).sum())
                                if decoder == "expanded_mask"
                                else 0,
                                **bench.target_score(challenge, truth, values, cfg),
                            }
                        )

    scores = pd.DataFrame(rows)
    key = ["split", "seed", "scenario", "shadow_target"]
    base = scores.loc[scores["decoder"].eq("edge"), key + ["power", "rare"]]
    base = base.rename(columns={"power": "base_power", "rare": "base_rare"})
    paired = scores.merge(base, on=key, validate="many_to_one")
    paired["delta"] = paired["power"] - paired["base_power"]
    paired["delta_rare"] = paired["rare"] - paired["base_rare"]
    paired.to_csv(HERE / "target20_membership_expansion_scores.csv", index=False)

    grouped = []
    for stress, selected in (
        (False, paired.loc[paired["scenario"].isin(bench.PRESERVING_SCENARIOS)]),
        (True, paired.loc[paired["scenario"].eq("warp_time_shuffled")]),
    ):
        summary = (
            selected.groupby(["split", "decoder"], as_index=False)
            .agg(
                cases=("delta", "size"),
                delta_mean=("delta", "mean"),
                delta_p10=("delta", lambda x: float(np.quantile(x, 0.10))),
                delta_min=("delta", "min"),
                negative_rate=("delta", lambda x: float(np.mean(x < -1e-15))),
                delta_rare_mean=("delta_rare", "mean"),
                added_active_mean=("added_active", "mean"),
            )
        )
        summary.insert(2, "stress_time_shuffled", stress)
        grouped.append(summary)
    summary = pd.concat(grouped, ignore_index=True)
    summary.to_csv(HERE / "target20_membership_expansion_summary.csv", index=False)
    report = {
        "scope": "target 20; production winner mask vs expansion by trusted links",
        "membership_fixed": "replace time only where baseline is active and link is trusted",
        "membership_changed": "also activate trusted links that baseline abstained on",
        "summary": summary.to_dict(orient="records"),
        "production_expansion_would_add": 4,
        "production_fixed_changes": 7,
        "production_total_trusted_links": 11,
    }
    (HERE / "target20_membership_expansion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
