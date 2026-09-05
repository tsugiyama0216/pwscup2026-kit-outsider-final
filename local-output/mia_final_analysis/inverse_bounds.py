"""Bound a production MIA variant under all released aggregate constraints.

This is research-only.  It treats organizer truth as unknown binary variables,
enforces the public cohort exclusivity/cardinality rules, the known-private
rows, both official v3/v4 TPR@1%-FPR aggregates and both official v3/v4 AUC
aggregates, then minimizes/maximizes a candidate variant's low-FPR aggregate.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")
N_ROWS = 1049
LOW_FPR_NEGATIVES = 10
OFFICIAL = {
    "v3_tpr": Fraction(1301, 14040),
    "v4_tpr": Fraction(11977, 117000),
    "v3_auc": 0.6434884498600183,
    "v4_auc": 0.6493386125210912,
}


def load_mia(candidate: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "submissions" / "prelim" / "team24" / candidate / "F_mia.csv").set_index("record_id")


def descending_positions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        order = np.argsort(-frame[target].to_numpy(float), kind="mergesort")
        positions[target] = frame.index.to_numpy(int)[order]
    return positions


class Model:
    def __init__(self, variant_candidate: str, enforce_auc: bool, exact_total: int | None):
        self.frames = {
            "v3": load_mia("attack-candidate-v3"),
            "v4": load_mia("attack-candidate-v4"),
            "variant": load_mia(variant_candidate),
        }
        if not all(frame.index.equals(self.frames["v4"].index) for frame in self.frames.values()):
            raise ValueError("record orders differ")
        self.records = self.frames["v4"].index.to_numpy(int)
        self.record_index = {rid: i for i, rid in enumerate(self.records)}
        self.orders = {name: descending_positions(frame) for name, frame in self.frames.items()}
        self.enforce_auc = enforce_auc
        self.exact_total = exact_total
        diagnostics = json.loads((ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8"))
        self.private_ids = set(int(v) for v in diagnostics["mia"]["private_record_ids"])

        self.var_names: list[tuple] = []
        self.indices: dict[tuple, int] = {}
        self.integrality: list[int] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.rows: list[dict[int, float]] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []

    def add_var(self, key: tuple, integral: bool, lower: float = 0.0, upper: float = 1.0) -> int:
        idx = len(self.var_names)
        self.var_names.append(key)
        self.indices[key] = idx
        self.integrality.append(1 if integral else 0)
        self.lb.append(lower)
        self.ub.append(upper)
        return idx

    def add_constraint(self, terms: dict[int, float], lower: float, upper: float) -> None:
        self.rows.append({idx: value for idx, value in terms.items() if value})
        self.row_lb.append(lower)
        self.row_ub.append(upper)

    def build(self):
        # Unknown membership labels.
        for target in TARGETS:
            for rid in self.records:
                private = int(rid) in self.private_ids
                self.add_var(("x", target, int(rid)), True, 0.0, 0.0 if private else 1.0)
        for target in TARGETS:
            for n in (25, 26, 27):
                self.add_var(("z", target, n), True)

        # Low-FPR operating-point choices for each observed or tested rank matrix.
        for version in ("v3", "v4", "variant"):
            for target in TARGETS:
                high = target in ("1", "2")
                for n in (25, 26, 27):
                    allowed = range((3 * n) // 4 + 1, n + 1) if high else range(0, n // 4 + 1)
                    for k in allowed:
                        self.add_var(("q", version, target, n, k), True)

        if self.enforce_auc:
            for version in ("v3", "v4"):
                for target in TARGETS:
                    for n in (25, 26, 27):
                        self.add_var(("auc", version, target, n), False)

        # Exactly one size per target and its member count.
        for target in TARGETS:
            zterms = {self.indices[("z", target, n)]: 1.0 for n in (25, 26, 27)}
            self.add_constraint(zterms, 1.0, 1.0)
            count = {self.indices[("x", target, int(rid))]: 1.0 for rid in self.records}
            count.update({self.indices[("z", target, n)]: -float(n) for n in (25, 26, 27)})
            self.add_constraint(count, 0.0, 0.0)

        if self.exact_total is not None:
            total = {self.indices[("x", target, int(rid))]: 1.0 for target in TARGETS for rid in self.records}
            self.add_constraint(total, float(self.exact_total), float(self.exact_total))

        # One candidate can be shared with at most one target.
        for rid in self.records:
            terms = {self.indices[("x", target, int(rid))]: 1.0 for target in TARGETS}
            self.add_constraint(terms, -np.inf, 1.0)

        # Choice q implies size n and exactly k positives before the 11th negative.
        for version in ("v3", "v4", "variant"):
            for target in TARGETS:
                high = target in ("1", "2")
                order = self.orders[version][target]
                for n in (25, 26, 27):
                    allowed = range((3 * n) // 4 + 1, n + 1) if high else range(0, n // 4 + 1)
                    qterms = {self.indices[("q", version, target, n, k)]: 1.0 for k in allowed}
                    qterms[self.indices[("z", target, n)]] = -1.0
                    self.add_constraint(qterms, 0.0, 0.0)
                    for k in allowed:
                        qidx = self.indices[("q", version, target, n, k)]
                        prefix = order[: LOW_FPR_NEGATIVES + k]
                        terms = {self.indices[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[qidx] = 40.0
                        # q=1 -> prefix count = k; q=0 -> safely relaxed.
                        self.add_constraint(terms, -np.inf, float(k + 40))
                        terms = {self.indices[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[qidx] = -40.0
                        self.add_constraint(terms, float(k - 40), np.inf)
                        next_rid = int(order[LOW_FPR_NEGATIVES + k])
                        terms = {
                            self.indices[("x", target, next_rid)]: 1.0,
                            qidx: 1.0,
                        }
                        self.add_constraint(terms, -np.inf, 1.0)

        # Exact official aggregate TPR constraints.
        for version in ("v3", "v4"):
            terms: dict[int, float] = {}
            for target in TARGETS:
                high = target in ("1", "2")
                for n in (25, 26, 27):
                    allowed = range((3 * n) // 4 + 1, n + 1) if high else range(0, n // 4 + 1)
                    for k in allowed:
                        terms[self.indices[("q", version, target, n, k)]] = k / n
            value = float(OFFICIAL[f"{version}_tpr"] * len(TARGETS))
            self.add_constraint(terms, value - 2e-12, value + 2e-12)

        if self.enforce_auc:
            big_m = 2.0
            for version in ("v3", "v4"):
                for target in TARGETS:
                    descending = self.orders[version][target]
                    asc_rank = {int(rid): N_ROWS - pos for pos, rid in enumerate(descending)}
                    for n in (25, 26, 27):
                        denominator = float(n * (N_ROWS - n))
                        constant = n * (n + 1) / (2.0 * denominator)
                        uidx = self.indices[("auc", version, target, n)]
                        zidx = self.indices[("z", target, n)]
                        base = {uidx: 1.0, zidx: big_m}
                        base.update({self.indices[("x", target, int(rid))]: -asc_rank[int(rid)] / denominator for rid in self.records})
                        self.add_constraint(base, -np.inf, big_m - constant)
                        base = {uidx: 1.0, zidx: -big_m}
                        base.update({self.indices[("x", target, int(rid))]: -asc_rank[int(rid)] / denominator for rid in self.records})
                        self.add_constraint(base, -big_m - constant, np.inf)
            for version in ("v3", "v4"):
                terms = {self.indices[("auc", version, target, n)]: 1.0 for target in TARGETS for n in (25, 26, 27)}
                value = OFFICIAL[f"{version}_auc"] * len(TARGETS)
                self.add_constraint(terms, value - 2e-10, value + 2e-10)

        rr: list[int] = []
        cc: list[int] = []
        vv: list[float] = []
        for row_index, terms in enumerate(self.rows):
            for col, value in terms.items():
                rr.append(row_index)
                cc.append(col)
                vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.var_names))).tocsr()
        constraint = LinearConstraint(matrix, np.asarray(self.row_lb), np.asarray(self.row_ub))
        return Bounds(np.asarray(self.lb), np.asarray(self.ub)), constraint

    def objective(self, maximize: bool) -> np.ndarray:
        c = np.zeros(len(self.var_names))
        for target in TARGETS:
            high = target in ("1", "2")
            for n in (25, 26, 27):
                allowed = range((3 * n) // 4 + 1, n + 1) if high else range(0, n // 4 + 1)
                for k in allowed:
                    c[self.indices[("q", "variant", target, n, k)]] = k / n / len(TARGETS)
        return -c if maximize else c

    def decode(self, result, maximize: bool) -> dict:
        values = result.x
        sign = -1.0 if maximize else 1.0
        score = sign * float(result.fun)
        selected = {}
        per_target = {}
        for target in TARGETS:
            n = next(n for n in (25, 26, 27) if values[self.indices[("z", target, n)]] > 0.5)
            ks = {}
            for version in ("v3", "v4", "variant"):
                high = target in ("1", "2")
                allowed = range((3 * n) // 4 + 1, n + 1) if high else range(0, n // 4 + 1)
                k = next(k for k in allowed if values[self.indices[("q", version, target, n, k)]] > 0.5)
                ks[version] = k
            members = [int(rid) for rid in self.records if values[self.indices[("x", target, int(rid))]] > 0.5]
            selected[target] = members
            per_target[target] = {"n": n, **ks, "delta": ks["variant"] / n - ks["v4"] / n}
        return {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "score": score,
            "delta_vs_v4": score - float(OFFICIAL["v4_tpr"]),
            "mip_gap": getattr(result, "mip_gap", None),
            "mip_node_count": getattr(result, "mip_node_count", None),
            "per_target": per_target,
            "members": selected,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="attack-candidate-v6-feedback-tail15")
    parser.add_argument("--without-auc", action="store_true")
    parser.add_argument("--exact-total", type=int)
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args()

    model = Model(args.variant, not args.without_auc, args.exact_total)
    bounds, constraints = model.build()
    output = {
        "variant": args.variant,
        "enforce_auc": not args.without_auc,
        "exact_total": args.exact_total,
        "variables": len(model.var_names),
        "constraints": len(model.rows),
        "runs": {},
    }
    for maximize in (False, True):
        label = "maximum" if maximize else "minimum"
        print(f"solve {label}: variables={len(model.var_names)} constraints={len(model.rows)}", flush=True)
        result = milp(
            model.objective(maximize),
            integrality=np.asarray(model.integrality),
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": args.time_limit, "mip_rel_gap": 0.0, "presolve": True},
        )
        print(result.message, flush=True)
        if result.x is None:
            output["runs"][label] = {"success": False, "status": int(result.status), "message": str(result.message)}
        else:
            output["runs"][label] = model.decode(result, maximize)
            print(label, output["runs"][label]["score"], output["runs"][label]["delta_vs_v4"], flush=True)

    suffix = "auc" if not args.without_auc else "no_auc"
    if args.exact_total is not None:
        suffix += f"_total{args.exact_total}"
    destination = Path(__file__).resolve().parent / f"inverse_bounds_{args.variant}_{suffix}.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
