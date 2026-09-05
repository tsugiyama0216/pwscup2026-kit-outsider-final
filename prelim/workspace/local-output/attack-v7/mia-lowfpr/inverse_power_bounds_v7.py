"""MILP bounds for the v7 MIA rare-lock candidate.

The model never guesses an individual truth label.  It ranges over *all*
membership assignments satisfying the official v3/v4/Daybreak/guided
aggregate low-FPR results, all published per-target bands, target cardinality
counts, exclusivity, the observed rare-channel hit counts, and the displayed
v4/guided total-score intervals.  Its output is therefore a compatibility
bound, not a reconstructed ground truth.

The rare model uses the production-consistent interpretation forced by the
official fractions: six targets contain one eligible rare member each; the
top rare member is hit by v3/v4/guided in 2/6 cases and by Daybreak in 1/6.
This assumption is reported explicitly and the candidate is not selected from
the MILP solution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "local-output" / "attack-v7" / "mia-lowfpr"
TEAM = ROOT / "submissions" / "prelim" / "team24"
TARGETS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
N_ROWS = 1_049
SCALE = 17_550  # lcm(25, 26, 27)
OFFICIAL_ALL = {"v3": 32_525, "v4": 35_931, "day": 37_283, "guided": 35_931}
OFFICIAL_RARE_WINS = {"v3": 2, "v4": 2, "day": 1, "guided": 2}


def load_frames() -> dict[str, pd.DataFrame]:
    paths = {
        "v3": TEAM / "attack-candidate-v3" / "F_mia.csv",
        "v4": TEAM / "attack-candidate-v4" / "F_mia.csv",
        "day": TEAM / "attack-candidate-daybreak-independent" / "F_mia.csv",
        "guided": TEAM / "attack-candidate-v6-routed-guided-probe" / "F_mia.csv",
        "candidate": OUT / "production_variants" / "F_mia_daybreak_v4_rarelock.csv",
    }
    frames = {name: pd.read_csv(path).set_index("record_id") for name, path in paths.items()}
    reference_ids = frames["v4"].index
    for name, frame in frames.items():
        if set(frame.index) != set(reference_ids) or set(frame.columns) != set(TARGETS):
            raise ValueError(f"unaligned frame: {name}")
        frames[name] = frame.loc[reference_ids, TARGETS]
    return frames


def orders(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, frame in frames.items():
        ids = frame.index.to_numpy(dtype=int)
        output[name] = {
            target: ids[np.argsort(-frame[target].to_numpy(dtype=float), kind="mergesort")]
            for target in TARGETS
        }
    return output


def official_k(target: str, n: int) -> range:
    if target in {"1", "2"}:
        return range((3 * n) // 4 + 1, n + 1)
    return range(0, n // 4 + 1)


def candidate_k(target: str, n: int) -> range:
    # The candidate is not officially measured, so do not assume its band.
    return range(0, n + 1)


class Model:
    def __init__(self) -> None:
        self.frames = load_frames()
        self.orders = orders(self.frames)
        diagnostics = json.loads(
            (ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8")
        )
        self.private = set(map(int, diagnostics["mia"]["private_record_ids"]))
        self.rare = tuple(map(int, diagnostics["mia"]["eligible_rare_record_ids"]))
        self.keys: list[tuple[Any, ...]] = []
        self.index: dict[tuple[Any, ...], int] = {}
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []
        self.score_terms: dict[str, dict[int, float]] = {}

    def var(self, key: tuple[Any, ...], lower: float = 0.0, upper: float = 1.0) -> int:
        if key in self.index:
            return self.index[key]
        idx = len(self.keys)
        self.keys.append(key)
        self.index[key] = idx
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(1)
        return idx

    def constraint(self, terms: dict[int, float], lower: float, upper: float) -> None:
        self.rows.append(terms)
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def add_and(self, result: int, left: int, right: int) -> None:
        self.constraint({result: 1.0, left: -1.0}, -np.inf, 0.0)
        self.constraint({result: 1.0, right: -1.0}, -np.inf, 0.0)
        self.constraint({result: 1.0, left: -1.0, right: -1.0}, -1.0, np.inf)

    def build(self) -> tuple[Bounds, LinearConstraint]:
        versions = tuple(self.frames)
        # Only membership variables that can affect one of the five operating
        # prefixes are required.  Add all eligible rare rows for rare-channel
        # constraints even when they lie outside every such prefix.
        relevant: dict[str, set[int]] = {target: set(self.rare) for target in TARGETS}
        for version in versions:
            for target in TARGETS:
                k_values = candidate_k(target, 27) if version == "candidate" else official_k(target, 27)
                maximum_k = max(k_values)
                relevant[target].update(map(int, self.orders[version][target][: 11 + maximum_k]))
        for target, values in relevant.items():
            for record_id in sorted(values):
                self.var(
                    ("x", target, record_id),
                    0.0,
                    0.0 if record_id in self.private else 1.0,
                )
            for n in (25, 26, 27):
                self.var(("z", target, n))
        for version in versions:
            for target in TARGETS:
                for n in (25, 26, 27):
                    values = candidate_k(target, n) if version == "candidate" else official_k(target, n)
                    for k in values:
                        self.var(("q", version, target, n, k))

        # One target size per column; exact public allocation 13/6/1.
        for target in TARGETS:
            self.constraint(
                {self.index[("z", target, n)]: 1.0 for n in (25, 26, 27)}, 1.0, 1.0
            )
            relevant_count = {
                self.index[("x", target, record_id)]: 1.0 for record_id in relevant[target]
            }
            for n in (25, 26, 27):
                relevant_count[self.index[("z", target, n)]] = -float(n)
            self.constraint(relevant_count, -np.inf, 0.0)
        for n, count in zip((25, 26, 27), (13, 6, 1), strict=True):
            self.constraint(
                {self.index[("z", target, n)]: 1.0 for target in TARGETS},
                float(count),
                float(count),
            )

        # A record belongs to at most one target.  Missing membership rows can
        # always be completed below every relevant prefix.
        by_record: dict[int, list[int]] = {}
        for key, idx in self.index.items():
            if key[0] == "x":
                by_record.setdefault(int(key[2]), []).append(idx)
        for values in by_record.values():
            if len(values) > 1:
                self.constraint({idx: 1.0 for idx in values}, -np.inf, 1.0)

        # Tie-free TPR@1% FPR: k positives precede the 11th negative.  q selects
        # exactly one feasible k for the selected target size.
        for version in versions:
            for target in TARGETS:
                for n in (25, 26, 27):
                    k_values = candidate_k(target, n) if version == "candidate" else official_k(target, n)
                    q_indices = [self.index[("q", version, target, n, k)] for k in k_values]
                    terms = {idx: 1.0 for idx in q_indices}
                    terms[self.index[("z", target, n)]] = -1.0
                    self.constraint(terms, 0.0, 0.0)
                    for k in k_values:
                        q = self.index[("q", version, target, n, k)]
                        prefix = self.orders[version][target][: 10 + k]
                        terms = {self.index[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[q] = 10.0
                        self.constraint(terms, -np.inf, float(k + 10))
                        terms = {self.index[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[q] = -float(k)
                        self.constraint(terms, 0.0, np.inf)
                        next_id = int(self.orders[version][target][10 + k])
                        self.constraint(
                            {self.index[("x", target, next_id)]: 1.0, q: 1.0},
                            -np.inf,
                            1.0,
                        )

        # Official aggregate all-channel equalities.
        for version, expected in OFFICIAL_ALL.items():
            terms: dict[int, float] = {}
            for target in TARGETS:
                for n in (25, 26, 27):
                    for k in official_k(target, n):
                        terms[self.index[("q", version, target, n, k)]] = float(SCALE * k // n)
            self.constraint(terms, float(expected), float(expected))

        # Six eligible rare members, at most one per target.  With 13 known
        # private rare rows this produces the official 1/3 and 1/6 granularity.
        all_rare_terms: dict[int, float] = {}
        for target in TARGETS:
            target_terms = {
                self.index[("x", target, record_id)]: 1.0 for record_id in self.rare
            }
            all_rare_terms.update(target_terms)
            self.constraint(target_terms, -np.inf, 1.0)
        self.constraint(all_rare_terms, 6.0, 6.0)

        win_indices: dict[str, dict[str, int]] = {name: {} for name in OFFICIAL_RARE_WINS}
        for version in OFFICIAL_RARE_WINS:
            for target in TARGETS:
                top_rare = next(
                    int(record_id)
                    for record_id in self.orders[version][target]
                    if int(record_id) in set(self.rare)
                )
                win_indices[version][target] = self.index[("x", target, top_rare)]
            self.constraint(
                {idx: 1.0 for idx in win_indices[version].values()},
                float(OFFICIAL_RARE_WINS[version]),
                float(OFFICIAL_RARE_WINS[version]),
            )

        # Construct exact scaled target-wise max(all, rare) expressions.  The
        # candidate has v4's rare order by construction, hence uses v4 wins.
        score_win_source = {
            "v4": "v4",
            "guided": "guided",
            "candidate": "v4",
        }
        for version, win_source in score_win_source.items():
            terms: dict[int, float] = {}
            for target in TARGETS:
                w = win_indices[win_source][target]
                for n in (25, 26, 27):
                    k_values = candidate_k(target, n) if version == "candidate" else official_k(target, n)
                    for k in k_values:
                        q = self.index[("q", version, target, n, k)]
                        terms[q] = terms.get(q, 0.0) + float(SCALE * k // n)
                        u = self.var(("u", version, target, n, k))
                        self.add_and(u, q, w)
                        terms[u] = terms.get(u, 0.0) + float(SCALE * (n - k) // n)
            self.score_terms[version] = terms

        # Displayed totals constrain, but do not pretend to identify, the exact
        # integer-valued MIA power.  Endpoints are conservative inclusive bounds.
        aia_v4 = 0.1191885120769747
        v4_lo = math.ceil((0.27705 - aia_v4) * 20.0 * SCALE - 1e-9)
        v4_hi = math.floor((0.27715 - aia_v4) * 20.0 * SCALE + 1e-9)
        self.constraint(self.score_terms["v4"], float(v4_lo), float(v4_hi))
        aia_guided = 0.14248115204199743
        guided_lo = math.ceil((0.30035 - aia_guided) * 20.0 * SCALE - 1e-9)
        guided_hi = math.floor((0.30045 - aia_guided) * 20.0 * SCALE + 1e-9)
        self.constraint(self.score_terms["guided"], float(guided_lo), float(guided_hi))

        rr: list[int] = []
        cc: list[int] = []
        vv: list[float] = []
        for row, terms in enumerate(self.rows):
            for col, value in terms.items():
                rr.append(row)
                cc.append(col)
                vv.append(value)
        matrix = coo_matrix(
            (vv, (rr, cc)), shape=(len(self.rows), len(self.keys))
        ).tocsr()
        return (
            Bounds(np.asarray(self.lower), np.asarray(self.upper)),
            LinearConstraint(matrix, self.row_lower, self.row_upper),
        )

    def difference_objective(self, maximize: bool) -> np.ndarray:
        objective = np.zeros(len(self.keys), dtype=float)
        for idx, value in self.score_terms["candidate"].items():
            objective[idx] += value
        for idx, value in self.score_terms["v4"].items():
            objective[idx] -= value
        return -objective if maximize else objective

    def decode(self, result: Any, maximize: bool) -> dict[str, Any]:
        sign = -1.0 if maximize else 1.0
        scaled_delta = sign * float(result.fun)
        values = np.asarray(result.x)
        size_by_target = {
            target: next(n for n in (25, 26, 27) if values[self.index[("z", target, n)]] > 0.5)
            for target in TARGETS
        }
        wins = {}
        for version in OFFICIAL_RARE_WINS:
            wins[version] = [
                target
                for target in TARGETS
                if values[
                    self.index[
                        (
                            "x",
                            target,
                            next(
                                int(record_id)
                                for record_id in self.orders[version][target]
                                if int(record_id) in set(self.rare)
                            ),
                        )
                    ]
                ]
                > 0.5
            ]
        return {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            "scaled_power_delta": scaled_delta,
            "power_delta_vs_v4": scaled_delta / (20.0 * SCALE),
            "size_by_target": size_by_target,
            "rare_win_targets": wins,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--lp-relaxation", action="store_true")
    args = parser.parse_args()
    model = Model()
    bounds, constraints = model.build()
    output: dict[str, Any] = {
        "candidate": "daybreak_v4_rarelock",
        "assumptions": {
            "target_sizes": {"25": 13, "26": 6, "27": 1},
            "exclusive_membership": True,
            "eligible_rare_members": 6,
            "at_most_one_rare_member_per_target": True,
            "official_all_equalities": OFFICIAL_ALL,
            "official_rare_win_counts": OFFICIAL_RARE_WINS,
            "v4_and_guided_displayed_total_intervals": True,
            "auc_constraints": False,
        },
        "variables": len(model.keys),
        "constraints": len(model.rows),
        "runs": {},
        "warning": "Feasible extrema are ambiguity bounds, never individual-label inferences.",
        "lp_relaxation": bool(args.lp_relaxation),
    }
    for maximize in (False, True):
        name = "maximum" if maximize else "minimum"
        print(f"solve {name}: vars={len(model.keys)} constraints={len(model.rows)}", flush=True)
        integrality = (
            np.zeros(len(model.integrality), dtype=int)
            if args.lp_relaxation
            else np.asarray(model.integrality)
        )
        result = milp(
            model.difference_objective(maximize),
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": args.time_limit, "mip_rel_gap": 0.0, "presolve": True},
        )
        if result.x is None:
            output["runs"][name] = {
                "success": False,
                "status": int(result.status),
                "message": str(result.message),
            }
        elif args.lp_relaxation:
            sign = -1.0 if maximize else 1.0
            scaled_delta = sign * float(result.fun)
            output["runs"][name] = {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "scaled_power_delta": scaled_delta,
                "power_delta_vs_v4": scaled_delta / (20.0 * SCALE),
            }
        else:
            output["runs"][name] = model.decode(result, maximize)
        print(output["runs"][name], flush=True)
    path = OUT / (
        "inverse_power_bounds_lp.json" if args.lp_relaxation else "inverse_power_bounds.json"
    )
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
