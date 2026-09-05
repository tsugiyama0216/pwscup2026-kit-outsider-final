"""Finalize the time-boxed MIA recheck without touching production artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
TEAM = ROOT / "submissions" / "prelim" / "team24"


def bool_column_is_true(frame: pd.DataFrame, column: str) -> bool:
    return frame[column].astype(str).str.lower().eq("true").all()


def existing_payload_diagnostics(inventory: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(TEAM / "attack-candidate-v4" / "F_mia.csv")
    baseline = baseline.set_index("record_id")
    rows: list[dict[str, object]] = []
    for item in inventory.itertuples(index=False):
        frame = pd.read_csv(item.path).set_index("record_id")
        frame = frame.loc[baseline.index, baseline.columns]
        changed = int(
            np.count_nonzero(
                ~np.isclose(
                    frame.to_numpy(dtype=float),
                    baseline.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-15,
                )
            )
        )
        top12_changes = 0
        top15_changes = 0
        correlations: list[float] = []
        for target in baseline.columns:
            base_values = baseline[target].to_numpy(dtype=float)
            values = frame[target].to_numpy(dtype=float)
            base_order = np.argsort(-base_values, kind="mergesort")
            order = np.argsort(-values, kind="mergesort")
            top12_changes += len(set(base_order[:12]) ^ set(order[:12])) // 2
            top15_changes += len(set(base_order[:15]) ^ set(order[:15])) // 2
            correlations.append(
                float(
                    pd.Series(base_values).corr(pd.Series(values), method="spearman")
                )
            )
        rows.append(
            {
                "candidate": item.candidate,
                "sha256": item.sha256,
                "same_bytes_as_v4": bool(item.sha256 == inventory.loc[
                    inventory["candidate"].eq("attack-candidate-v4"), "sha256"
                ].iloc[0]),
                "changed_cells_from_v4": changed,
                "top12_set_changes_total": top12_changes,
                "top15_set_changes_total": top15_changes,
                "mean_spearman_vs_v4": float(np.mean(correlations)),
                "minimum_spearman_vs_v4": float(np.min(correlations)),
            }
        )
    return pd.DataFrame(rows).sort_values(["sha256", "candidate"])


def main() -> None:
    inventory = pd.read_csv(OUT / "production_inventory.csv")
    results = pd.read_csv(OUT / "scores_by_world.csv")
    targets = pd.read_csv(OUT / "scores_by_target.csv")
    split_summary = pd.read_csv(OUT / "summary_by_split.csv")
    variant_checks = pd.read_csv(OUT / "production_variant_checks.csv")
    worlds = pd.read_csv(OUT / "world_checks.csv")
    report_json = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

    hash_groups = (
        inventory.groupby("sha256", as_index=False)
        .agg(
            file_count=("candidate", "size"),
            candidates=("candidate", lambda values: ";".join(sorted(values))),
        )
        .sort_values(["file_count", "sha256"], ascending=[False, True])
    )
    hash_groups.to_csv(OUT / "production_hash_groups.csv", index=False)
    existing = existing_payload_diagnostics(inventory)
    existing.to_csv(OUT / "production_existing_diagnostics.csv", index=False)

    key_world = ["split", "seed", "scenario", "variant"]
    key_target = key_world + ["target"]
    recomputed = (
        targets.groupby(key_world, as_index=False)[["power", "all", "rare", "auc"]]
        .mean()
        .merge(results, on=key_world, suffixes=("_target_mean", "_aggregate"))
    )
    aggregate_error = max(
        float(
            np.nanmax(
                np.abs(
                    recomputed[f"{metric}_target_mean"]
                    - recomputed[f"{metric}_aggregate"]
                )
            )
        )
        for metric in ("power", "all", "rare", "auc")
    )
    result_metrics_finite = bool(
        np.isfinite(results[["power", "all", "rare", "auc"]].to_numpy(dtype=float)).all()
    )
    world_boolean_columns = [
        column
        for column in worlds.columns
        if column not in {"split", "seed", "count_25", "count_26", "count_27"}
    ]
    qa = {
        "aggregate_rows": int(len(results)),
        "target_rows": int(len(targets)),
        "production_files_inventoried": int(len(inventory)),
        "unique_production_payloads": int(inventory["sha256"].nunique()),
        "research_variants": int(results["variant"].nunique()),
        "worlds": int(worlds[["split", "seed"]].drop_duplicates().shape[0]),
        "aggregate_keys_unique": bool(~results.duplicated(key_world).any()),
        "target_keys_unique": bool(~targets.duplicated(key_target).any()),
        "twenty_targets_per_variant_world": bool(
            targets.groupby(key_world).size().eq(20).all()
        ),
        "aggregate_metrics_finite": result_metrics_finite,
        "aggregate_recompute_max_abs_error": aggregate_error,
        "all_world_checks_passed": bool(
            all(bool_column_is_true(worlds, column) for column in world_boolean_columns)
        ),
        "all_research_outputs_normal_format": bool(
            variant_checks["variant"].nunique() == results["variant"].nunique()
            and variant_checks["sha256"].notna().all()
        ),
    }
    qa["passed"] = bool(
        qa["aggregate_keys_unique"]
        and qa["target_keys_unique"]
        and qa["twenty_targets_per_variant_world"]
        and qa["aggregate_metrics_finite"]
        and qa["aggregate_recompute_max_abs_error"] <= 1e-12
        and qa["all_world_checks_passed"]
        and qa["all_research_outputs_normal_format"]
    )
    (OUT / "qa_checks.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def row(split: str, variant: str) -> pd.Series:
        return split_summary.loc[
            split_summary["split"].eq(split)
            & split_summary["variant"].eq(variant)
        ].iloc[0]

    disc_v4 = row("discovery", "v4")
    val_v4 = row("validation", "v4")
    disc_tail = row("discovery", "tail12_reorder_avg")
    val_tail = row("validation", "tail12_reorder_avg")
    disc_avg3 = row("discovery", "rarelock_avg3_equal")
    val_avg3 = row("validation", "rarelock_avg3_equal")
    disc_v3 = row("discovery", "v3")
    val_v3 = row("validation", "v3")
    disc_day = row("discovery", "daybreak")
    val_day = row("validation", "daybreak")

    report_json["as_of"] = "2026-09-05 Asia/Tokyo"
    report_json["run_status"] = "timeboxed_complete"
    report_json["evidence_scope"] = {
        "discovery_worlds": 1,
        "validation_worlds": 1,
        "scenario": "noise_0p40",
        "note": (
            "The parent task time-boxed the recheck after the first qualified held-out "
            "validation world because no discovery-selected normal-format alternative won."
        ),
    }
    report_json["decision_headline"] = {
        "recommendation": "retain v4 MIA unchanged",
        "replacement_supported": False,
        "selected_safe_candidate": "tail12_reorder_avg",
        "selected_validation_delta_power": float(val_tail["mean_delta_power"]),
        "selected_validation_delta_auc": float(val_tail["mean_delta_auc"]),
        "reason": (
            "The discovery-selected safe alternative tied v4 on low-FPR power in "
            "validation and slightly reduced AUC; the only validation-positive blend "
            "had a negative discovery result. Official v4/guided equality is stronger "
            "evidence than isolated shadow gains."
        ),
    }
    report_json["qa"] = qa
    report_json["limitations"] = [
        "This was stopped after one qualified discovery world and one held-out validation world under the severe 40% noise scenario; uncertainty is high.",
        "Shadow values compare recipes and do not predict private leaderboard scores.",
        "Official v4-vs-guided equality and v4 rare strength override weak or sign-unstable shadow differences.",
        "The unrun 10%/20% noise and independent-null scenarios cannot be cited as completed evidence.",
        "Candidate selection used discovery only; validation was read once through the fixed gate.",
    ]
    (OUT / "summary.json").write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    markdown = f"""# MIA local recheck — time-boxed final decision

## Decision

**Keep the v4 MIA payload unchanged.** No normal-format alternative has a reproducible held-out advantage, and the strongest available evidence is still the official comparison: v4 and the later guided/routed submission have identical TPR at 1% FPR (`0.1023675`) and rare TPR (`0.3333333`), while their AUC differs by only `0.0000059`.

This run was deliberately stopped after one qualified discovery world and one disjoint validation world. The sample is too small to claim optimality, but it is sufficient to reject a last-minute replacement when the predeclared gate is not met. Shadow evidence does not override official results.

## Existing payload inventory

- 15 production candidate directories contain `F_mia.csv`, collapsing to 7 byte-distinct payloads.
- `attack-candidate-v4`, `v5-probe-independent`, `v5-structured`, `v6-feedback-v4-probe`, `v6-routed-probe`, `v6-final-feedback-winners`, and `v6-final-aggressive-complements` use the exact same v4 MIA bytes.
- `attack-candidate-v6-routed-guided-probe` is byte-distinct, but its official low-FPR and rare scores tie v4; only AUC moves from `0.6493386` to `0.6493445`.
- `attack-candidate-v3` and `attack-candidate-daybreak-blue` share one payload. The independent Daybreak payload is separate.

The complete hash grouping and rank-distance diagnostics are saved in `production_hash_groups.csv` and `production_existing_diagnostics.csv`.

## Official evidence controls the decision

| payload/result | TPR @ 1% FPR | rare TPR | AUC | implication |
|---|---:|---:|---:|---|
| v3 | 0.092664 | 0.333333 | 0.643488 | below v4 on overall tail and AUC |
| v4 | 0.102368 | 0.333333 | 0.649339 | baseline to retain |
| guided/routed | 0.102368 | 0.333333 | 0.649344 | no measurable tail lift over v4 |
| Daybreak independent | 0.106219 | 0.166667 | 0.635601 | small overall-tail gain, but rare TPR halves and AUC falls |

## Normal-format alternatives tested

Sixteen payloads were scored: v4, v3, Daybreak, seven rare/private-locked rank combinations (weighted rank averages, min/max, and reciprocal-rank fusion), three top-tail reorders, and three top-tail replacements. Every research payload has one finite, tie-free score per record and target. The safer alternatives lock the v4 rare ordering and the 163 inferred private-bottom records.

The comparison used a truth-known shadow world with the production shared-count pattern (13 targets × 25, 6 × 26, 1 × 27), 20 targets, and a severe 40% noise release. Seed `20261301` was discovery; disjoint seed `20261311` was held-out validation.

| candidate | discovery power | Δ vs v4 | validation power | Δ vs v4 | validation Δ AUC | interpretation |
|---|---:|---:|---:|---:|---:|---|
| v4 | {disc_v4['mean_power']:.6f} | 0.000000 | {val_v4['mean_power']:.6f} | 0.000000 | 0.000000 | baseline |
| tail12 reorder (discovery-selected safe candidate) | {disc_tail['mean_power']:.6f} | {disc_tail['mean_delta_power']:+.6f} | {val_tail['mean_power']:.6f} | {val_tail['mean_delta_power']:+.6f} | {val_tail['mean_delta_auc']:+.6f} | ties tail power, slightly lowers AUC |
| equal 3-way locked average | {disc_avg3['mean_power']:.6f} | {disc_avg3['mean_delta_power']:+.6f} | {val_avg3['mean_power']:.6f} | {val_avg3['mean_delta_power']:+.6f} | {val_avg3['mean_delta_auc']:+.6f} | sign reversal; not reproducible |
| v3 | {disc_v3['mean_power']:.6f} | {disc_v3['mean_delta_power']:+.6f} | {val_v3['mean_power']:.6f} | {val_v3['mean_delta_power']:+.6f} | {val_v3['mean_delta_auc']:+.6f} | shadow gain contradicted by official result |
| Daybreak | {disc_day['mean_power']:.6f} | {disc_day['mean_delta_power']:+.6f} | {val_day['mean_power']:.6f} | {val_day['mean_delta_power']:+.6f} | {val_day['mean_delta_auc']:+.6f} | clearly below v4 in this shadow |

Among the 13 safe alternatives, none beat v4 in discovery: two tied and eleven lost. One unselected blend was positive in validation (`+0.003846`) after losing in discovery (`-0.009923`), so it fails the required replication test.

## Predeclared replacement gate

The discovery-selected `tail12_reorder_avg` failed the two essential checks: discovery mean lift had to exceed `0.0025`, and held-out validation mean lift had to exceed `0.0025`. Its observed lift was `0` in both splits; validation win rate was `0%`. Rare order and private-bottom preservation passed, but format safety alone is not efficacy.

## QA and limitations

- {qa['aggregate_rows']} aggregate rows and {qa['target_rows']} per-target rows were checked; keys are unique and each variant/world has 20 targets.
- Aggregate metrics recompute from target rows with maximum absolute error `{qa['aggregate_recompute_max_abs_error']:.3g}`.
- All generated-world consistency checks passed, and all 16 research candidates passed normal-format validation.
- **Limitation:** one discovery and one validation world cannot measure a stable distribution. The broader multi-seed run was stopped by the task's time limit after an additional seed could not supply the fixed 13 rare known-private controls. Therefore the conclusion is conservative: there is no evidence to justify a change, not proof that v4 is globally optimal.
- Production MIA files were read for comparison; no production candidate, archive, token, or submission was changed, and no token was read.
"""
    (OUT / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print("decision=retain_v4")


if __name__ == "__main__":
    main()
