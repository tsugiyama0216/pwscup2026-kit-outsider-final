"""Production-public diagnostics for predeclared positive-target variants."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
research = runpy.run_path(
    str(ROOT / "local-output" / "analysis_v6" / "evaluate_positive_copula_link.py")
)
SELECTORS = research["SELECTORS"]
POSITIVE_IDS = sorted(research["POSITIVE_IDS"], key=int)
copula_links = research["copula_links"]
decoded_variants = research["decoded_variants"]
aligned_times = research["aligned_times"]

from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_proximity_score,
    proximity_score,
    stratified_half_selection,
)


def main() -> None:
    data = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidate_root = ROOT / "submissions" / "prelim" / "team24"
    v4 = pd.read_csv(candidate_root / "attack-candidate-v4" / "F_aia.csv")
    independent = pd.read_csv(
        candidate_root / "attack-candidate-v5-probe-independent" / "F_aia.csv"
    )
    rows: list[dict] = []
    frame_parts: dict[str, list[pd.DataFrame]] = {}

    candidate_variants = {
        "equal_ind": ("cons_eq_mix", "ind_event_margin"),
        "current_v4_margin": ("cons_v4_mix", "v4_event_margin"),
        "current_v4_only": ("cons_v4_mix", "v4_only"),
        "existing_cons_ind": ("cons_v4_mix", "ind_fb"),
        "current_ind": ("ind_raw", "ind_fb"),
        "aggr_t3": ("cons_eq_cop", "cop_consensus_all"),
        "aggr_t4": ("cons_eq_cop", "ind_new_unique"),
        "aggr_t5": ("cons_eq_cop", "cop_consensus_all"),
        "aggr_t10": ("cons_ind_mix", "cop_consensus_all"),
        "aggr_t12": ("soft_cop", "cop_consensus_all"),
        "aggr_t13": ("cons_eq_cop", "cop_consensus_all"),
        "aggr_t22": ("ind_cop", "cop_consensus_all"),
        "aggr_t23": ("cons_eq_mix", "ind_event_agree"),
        "aggr_t26": ("soft_cop", "ind_event_margin"),
        "aggr_t29": ("cons_v4_mix", "cop_consensus_all"),
        "route_t3": ("cons_eq_mix", "ind_event_margin"),
        "route_t4": ("cons_eq_cop", "v4_event_strict"),
        "route_t5": ("cons_eq_cop", "v4_event_margin"),
        "route_t10": ("cons_v4_mix", "v4_event_margin"),
        "route_t12": ("cons_eq_cop", "v4_event_strict"),
        "route_t13": ("cons_eq_cop", "v4_event_mutual"),
        "route_t22": ("ind_cop", "ind_event_margin"),
        "route_t23": ("cons_eq_mix", "ind_event_agree"),
        "route_t26": ("soft_cop", "ind_event_margin"),
        "route_t29": ("cons_v4_mix", "v4_event_margin"),
        "final_t3": ("cons_eq_mix", "ind_event_margin"),
        "final_t4": ("cons_eq_cop", "v4_event_margin"),
        "final_t5": ("cons_eq_cop", "v4_event_margin"),
        "final_t10": ("cons_v4_mix", "v4_event_margin"),
        "final_t12": ("cons_eq_cop", "v4_event_margin"),
        "final_t13": ("cons_eq_cop", "ind_event_margin"),
        "final_t22": ("cons_eq_cop", "ind_event_margin"),
        "final_t23": ("cons_eq_cop", "ind_event_margin"),
        "final_t26": ("soft_cop", "ind_event_margin"),
        "final_t29": ("cons_v4_mix", "v4_event_margin"),
    }

    for target_id in POSITIVE_IDS:
        released = pd.read_csv(data / f"C_{target_id}.csv")
        challenge = pd.read_csv(data / f"aia_challenge_{target_id}.csv")
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_time = aligned_times(v4, target_id, ids)
        ind_time = aligned_times(independent, target_id, ids)
        active_v4 = ~np.isclose(v4_time, 11.0)
        active_ind = ~np.isclose(ind_time, 11.0)
        raw = proximity_score(released, challenge)
        copula = copula_proximity_score(released, challenge)
        links = copula_links(released, challenge)
        nearest_time = research["prediction_from_link"](released, links["nearest"])
        event_link = nearest_time < 10.0 - 1e-9
        margin_gate = links["mutual"] & (links["margin_rank"] >= 0.50)

        materialized: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label, (selector_name, decoder_name) in candidate_variants.items():
            wv4, wind, wraw, wcop = SELECTORS[selector_name]
            score = wv4 * active_v4 + wind * active_ind + wraw * raw + wcop * copula
            selected, _ = stratified_half_selection(challenge, score)
            decoded = decoded_variants(selected, v4_time, ind_time, released, links)
            output = decoded[decoder_name]
            if decoder_name.startswith("v4_event_"):
                base_output = decoded["v4_only"]
            elif decoder_name.startswith("ind_event_"):
                base_output = decoded["ind_fb"]
            else:
                base_output = output
            materialized[label] = (selected, output, base_output)
            frame_parts.setdefault(label, []).append(
                pd.DataFrame(
                    {
                        "target_k": int(target_id),
                        "challenge_row_id": ids,
                        "time_hat": output,
                    }
                )
            )

        current_label = (
            "current_ind" if target_id in {"3", "22", "23", "26"}
            else "current_v4_only"
        )
        current_selected, current_time, _ = materialized[current_label]
        for label, (selected, output, base_output) in materialized.items():
            changed_time = ~np.isclose(output, current_time)
            changed_active = selected != current_selected
            event_override = selected & event_link & margin_gate
            rows.append(
                {
                    "target": int(target_id),
                    "variant": label,
                    "active_changes_current": int(changed_active.sum()),
                    "time_changes_current": int(changed_time.sum()),
                    "active_changes_v4": int((selected != active_v4).sum()),
                    "active_changes_ind": int((selected != active_ind).sum()),
                    "event_predictions": int(np.count_nonzero(output < 9.5)),
                    "margin_event_overrides_selected": int(event_override.sum()),
                    "event_override_time_changes_from_base": int(
                        np.count_nonzero(~np.isclose(output, base_output))
                    ),
                    "selected_mutual_links": int(np.count_nonzero(selected & links["mutual"])),
                    "selected_agreement_ge_2of3": int(
                        np.count_nonzero(selected & (links["agreement"] >= (2 / 3)))
                    ),
                    "selected_margin_gate": int(np.count_nonzero(selected & margin_gate)),
                    "selected_event_margin_gate": int(event_override.sum()),
                    "all_mutual_links": int(links["mutual"].sum()),
                    "all_agreement_ge_2of3": int(
                        np.count_nonzero(links["agreement"] >= (2 / 3))
                    ),
                    "all_event_margin_gate": int(np.count_nonzero(event_link & margin_gate)),
                }
            )

    out_dir = ROOT / "local-output" / "positive-target-copula-link"
    pd.DataFrame(rows).to_csv(out_dir / "production_diagnostics.csv", index=False)
    for label, parts in frame_parts.items():
        pd.concat(parts, ignore_index=True).sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        ).to_csv(out_dir / f"production_{label}.csv", index=False)
    (out_dir / "production_algorithm.json").write_text(
        json.dumps(
            {
                "event_margin_gate": {
                    "link": "full-five-QI nearest release row within sex+prefecture, with sex/all fallback",
                    "mutual": "linked release row has the challenge row as its reverse nearest neighbor in the same full copula distance matrix",
                    "margin": "second-nearest minus nearest full-copula distance ranked globally among 326 challenge rows",
                    "threshold": "top 50% margin rank (>=0.50)",
                    "override": "only linked event times (<10); otherwise retain selected expert/fallback time",
                },
                "selector_equal_mix": {
                    "active_v4": 1.0,
                    "active_independent": 1.0,
                    "raw_proximity_rank": 0.2,
                    "copula_proximity_rank": 0.2,
                },
                "submission_performed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
