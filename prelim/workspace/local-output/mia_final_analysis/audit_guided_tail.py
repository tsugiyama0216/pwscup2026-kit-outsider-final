"""Audit generated Daybreak-guided tail variants against v4 invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TEAM = ROOT / "submissions" / "prelim" / "team24"
TARGETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("record_id")


def ordered(frame: pd.DataFrame, target: str) -> list[int]:
    return frame[target].sort_values(ascending=False, kind="mergesort").index.astype(int).tolist()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v4_path = TEAM / "attack-candidate-v4" / "F_mia.csv"
    tail_path = TEAM / "attack-candidate-v6-feedback-tail15" / "F_mia.csv"
    guided_path = OUT / "F_mia_guided_v4_margin3.csv"
    diagnostics = json.loads((ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8"))
    private = set(map(int, diagnostics["mia"]["private_record_ids"]))
    rare = set(map(int, diagnostics["mia"]["eligible_rare_record_ids"]))
    v4, tail, guided = map(load, (v4_path, tail_path, guided_path))

    original_flat = [rid for target in TARGETS for rid in ordered(v4, target)[:15]]
    guided_flat = [rid for target in TARGETS for rid in ordered(guided, target)[:15]]
    per_target: dict[str, dict[str, object]] = {}
    changed_slot_pairs = 0
    for target in TARGETS:
        v4_order = ordered(v4, target)
        tail_order = ordered(tail, target)
        guided_order = ordered(guided, target)
        v4_top = set(v4_order[:15])
        tail_top = set(tail_order[:15])
        guided_top = set(guided_order[:15])
        removed = sorted(tail_top - guided_top)
        added = sorted(guided_top - tail_top)
        changed_slot_pairs += len(removed)
        prefix_changes = {
            str(width): {
                "removed_vs_tail15": sorted(set(tail_order[:width]) - set(guided_order[:width])),
                "added_vs_tail15": sorted(set(guided_order[:width]) - set(tail_order[:width])),
            }
            for width in range(1, 17)
            if set(tail_order[:width]) != set(guided_order[:width])
        }
        per_target[target] = {
            "v4_retained_top15": len(v4_top & guided_top),
            "removed_vs_tail15_top15": removed,
            "added_vs_tail15_top15": added,
            "prefix_changes_through_16": prefix_changes,
            "whole_rank_order_equals_v4": guided_order == v4_order,
            "rare_order_equals_v4": [rid for rid in guided_order if rid in rare] == [rid for rid in v4_order if rid in rare],
            "private_bottom_equals_v4": set(guided_order[-len(private):]) == set(v4_order[-len(private):]) == private,
        }

    retained = sum(row["v4_retained_top15"] for row in per_target.values())
    report = {
        "variant": "guided_v4_margin3",
        "parameters": {
            "top_l": 15,
            "minimum_daybreak_rank_gain": 5,
            "maximum_v4_rank_penalty": 3,
        },
        "sha256": sha256(guided_path),
        "row_count": len(guided),
        "target_count": len(TARGETS),
        "tail_slots": len(guided_flat),
        "v4_original_unique_top15_records": len(set(original_flat)),
        "v4_original_duplicate_slots": len(original_flat) - len(set(original_flat)),
        "guided_unique_top15_records": len(set(guided_flat)),
        "guided_duplicate_slots": len(guided_flat) - len(set(guided_flat)),
        "retained_v4_top15_slots": retained,
        "changed_v4_top15_slots": len(guided_flat) - retained,
        "changed_slot_pairs_vs_tail15": changed_slot_pairs,
        "all_rare_orders_preserved": all(row["rare_order_equals_v4"] for row in per_target.values()),
        "all_private_bottom_sets_preserved": all(row["private_bottom_equals_v4"] for row in per_target.values()),
        "targets_1_2_whole_rank_order_equals_v4": all(per_target[target]["whole_rank_order_equals_v4"] for target in ("1", "2")),
        "per_target": per_target,
    }
    (OUT / "guided_v4_margin3_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_target"}, indent=2))


if __name__ == "__main__":
    main()
