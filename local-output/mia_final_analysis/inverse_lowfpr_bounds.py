"""Fast exact low-FPR inverse bounds using all released MIA aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")
DENOMINATOR_SCALE = 17_550  # lcm(25, 26, 27)
OFFICIAL_SCALED_SUMS = {"v3": 32_525, "v4": 35_931, "ind": 37_283}


def load(candidate: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "submissions" / "prelim" / "team24" / candidate / "F_mia.csv").set_index("record_id")


def orders(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ids = frame.index.to_numpy(int)
    return {target: ids[np.argsort(-frame[target].to_numpy(float), kind="mergesort")] for target in TARGETS}


def allowed_k(target: str, n: int) -> range:
    return range((3 * n) // 4 + 1, n + 1) if target in ("1", "2") else range(0, n // 4 + 1)


class FastModel:
    def __init__(self, variant: str, use_independent: bool, exact_total: int | None, size_counts: tuple[int, int, int] | None = None, high_sizes: tuple[int, int] | None = None):
        frames = {
            "v3": load("attack-candidate-v3"),
            "v4": load("attack-candidate-v4"),
            "ind": load("attack-candidate-daybreak-independent"),
            "variant": load(variant),
        }
        self.rank_orders = {name: orders(frame) for name, frame in frames.items()}
        self.official_versions = ("v3", "v4", "ind") if use_independent else ("v3", "v4")
        self.versions = (*self.official_versions, "variant")
        self.exact_total = exact_total
        self.size_counts = size_counts
        self.high_sizes = high_sizes
        diagnostics = json.loads((ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8"))
        self.private = set(int(v) for v in diagnostics["mia"]["private_record_ids"])
        self.keys: list[tuple] = []
        self.index: dict[tuple, int] = {}
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integral: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []

    def var(self, key: tuple, lower: float = 0.0, upper: float = 1.0) -> int:
        if key in self.index:
            return self.index[key]
        idx = len(self.keys)
        self.keys.append(key)
        self.index[key] = idx
        self.lb.append(lower)
        self.ub.append(upper)
        self.integral.append(1)
        return idx

    def constraint(self, terms: dict[int, float], lower: float, upper: float) -> None:
        self.rows.append(terms)
        self.row_lb.append(lower)
        self.row_ub.append(upper)

    def build(self):
        # Only rows that can affect a low-FPR operating point are needed.  All
        # other members can be completed below these prefixes without altering
        # the metric; the full cohort has ample unused eligible records.
        relevant: dict[str, set[int]] = {target: set() for target in TARGETS}
        for version in self.versions:
            for target in TARGETS:
                maximum_k = max(max(allowed_k(target, n)) for n in (25, 26, 27))
                relevant[target].update(map(int, self.rank_orders[version][target][: 11 + maximum_k]))
        for target, ids in relevant.items():
            for rid in ids:
                self.var(("x", target, rid), 0.0, 0.0 if rid in self.private else 1.0)
        for target in TARGETS:
            for n in (25, 26, 27):
                self.var(("z", target, n))
        for version in self.versions:
            for target in TARGETS:
                for n in (25, 26, 27):
                    for k in allowed_k(target, n):
                        self.var(("q", version, target, n, k))

        for target in TARGETS:
            self.constraint({self.index[("z", target, n)]: 1.0 for n in (25, 26, 27)}, 1.0, 1.0)
            for version in self.versions:
                for n in (25, 26, 27):
                    terms = {self.index[("q", version, target, n, k)]: 1.0 for k in allowed_k(target, n)}
                    terms[self.index[("z", target, n)]] = -1.0
                    self.constraint(terms, 0.0, 0.0)
                    for k in allowed_k(target, n):
                        q = self.index[("q", version, target, n, k)]
                        prefix = self.rank_orders[version][target][: 10 + k]
                        # Tight indicator formulation.  A prefix has 10+k rows,
                        # so its count can exceed k by at most 10.  The lower
                        # implication is simply count >= k*q.
                        terms = {self.index[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[q] = 10.0
                        self.constraint(terms, -np.inf, float(k + 10))
                        terms = {self.index[("x", target, int(rid))]: 1.0 for rid in prefix}
                        terms[q] = -float(k)
                        self.constraint(terms, 0.0, np.inf)
                        next_rid = int(self.rank_orders[version][target][10 + k])
                        self.constraint({self.index[("x", target, next_rid)]: 1.0, q: 1.0}, -np.inf, 1.0)

        if self.exact_total is not None:
            self.constraint(
                {self.index[("z", target, n)]: float(n) for target in TARGETS for n in (25, 26, 27)},
                float(self.exact_total),
                float(self.exact_total),
            )
        if self.size_counts is not None:
            for n, count in zip((25, 26, 27), self.size_counts, strict=True):
                self.constraint({self.index[("z", target, n)]: 1.0 for target in TARGETS}, float(count), float(count))
        if self.high_sizes is not None:
            for target, chosen in zip(("1", "2"), self.high_sizes, strict=True):
                self.constraint({self.index[("z", target, chosen)]: 1.0}, 1.0, 1.0)

        # Exact candidate exclusivity among all relevant target occurrences.
        by_record: dict[int, list[int]] = {}
        for key, idx in self.index.items():
            if key[0] == "x":
                by_record.setdefault(int(key[2]), []).append(idx)
        for indices in by_record.values():
            if len(indices) > 1:
                self.constraint({idx: 1.0 for idx in indices}, -np.inf, 1.0)

        for version in self.official_versions:
            terms = {}
            for target in TARGETS:
                for n in (25, 26, 27):
                    for k in allowed_k(target, n):
                        terms[self.index[("q", version, target, n, k)]] = float(DENOMINATOR_SCALE * k // n)
            value = float(OFFICIAL_SCALED_SUMS[version])
            self.constraint(terms, value, value)

        rr, cc, vv = [], [], []
        for row, terms in enumerate(self.rows):
            for col, value in terms.items():
                rr.append(row)
                cc.append(col)
                vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.keys))).tocsr()
        return Bounds(np.asarray(self.lb), np.asarray(self.ub)), LinearConstraint(matrix, self.row_lb, self.row_ub)

    def objective(self, maximize: bool) -> np.ndarray:
        objective = np.zeros(len(self.keys))
        for target in TARGETS:
            for n in (25, 26, 27):
                for k in allowed_k(target, n):
                    objective[self.index[("q", "variant", target, n, k)]] = DENOMINATOR_SCALE * k // n
        return -objective if maximize else objective

    def decode(self, result, maximize: bool) -> dict:
        v = result.x
        scaled_sum = round((-1.0 if maximize else 1.0) * float(result.fun))
        per_target = {}
        for target in TARGETS:
            n = next(n for n in (25, 26, 27) if v[self.index[("z", target, n)]] > 0.5)
            values = {}
            for version in self.versions:
                values[version] = next(k for k in allowed_k(target, n) if v[self.index[("q", version, target, n, k)]] > 0.5)
            per_target[target] = {"n": n, **values, "delta_scaled": (values["variant"] - values["v4"]) * (DENOMINATOR_SCALE // n)}
        score = scaled_sum / DENOMINATOR_SCALE / len(TARGETS)
        return {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": getattr(result, "mip_gap", None),
            "scaled_sum": scaled_sum,
            "score": score,
            "delta_vs_v4": (scaled_sum - OFFICIAL_SCALED_SUMS["v4"]) / DENOMINATOR_SCALE / len(TARGETS),
            "per_target": per_target,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="attack-candidate-v6-feedback-tail15")
    parser.add_argument("--without-independent", action="store_true")
    parser.add_argument("--exact-total", type=int, default=508)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--size-counts", help="comma-separated counts for n=25,26,27")
    parser.add_argument("--high-sizes", help="comma-separated n for targets 1,2")
    args = parser.parse_args()
    size_counts = tuple(map(int, args.size_counts.split(","))) if args.size_counts else None
    high_sizes = tuple(map(int, args.high_sizes.split(","))) if args.high_sizes else None
    model = FastModel(args.variant, not args.without_independent, args.exact_total, size_counts, high_sizes)
    bounds, constraint = model.build()
    output = {"variant": args.variant, "independent_constraints": not args.without_independent, "exact_total": args.exact_total, "size_counts": size_counts, "high_sizes": high_sizes, "variables": len(model.keys), "constraints": len(model.rows), "runs": {}}
    for maximize in (False, True):
        name = "maximum" if maximize else "minimum"
        print(f"solve {name}: {len(model.keys)} variables, {len(model.rows)} constraints", flush=True)
        result = milp(model.objective(maximize), integrality=np.asarray(model.integral), bounds=bounds, constraints=constraint, options={"time_limit": args.time_limit, "mip_rel_gap": 0.0, "presolve": True})
        print(result.message, flush=True)
        output["runs"][name] = model.decode(result, maximize) if result.x is not None else {"success": False, "status": int(result.status), "message": str(result.message)}
        if result.x is not None:
            print(name, output["runs"][name]["score"], output["runs"][name]["delta_vs_v4"], flush=True)
    suffix = "with_ind" if not args.without_independent else "without_ind"
    suffix += f"_total{args.exact_total}" if args.exact_total is not None else "_totalfree"
    if size_counts:
        suffix += "_counts" + "-".join(map(str, size_counts))
    if high_sizes:
        suffix += "_high" + "-".join(map(str, high_sizes))
    dest = Path(__file__).resolve().parent / f"fast_bounds_{args.variant}_{suffix}.json"
    dest.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
