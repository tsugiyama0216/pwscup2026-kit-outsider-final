"""Public toy diagnosis only: no contest data or DP release is used."""
from pathlib import Path
import json
from importlib.metadata import version
import inspect
import textwrap

import numpy as np
from mbi import CliqueVector, Domain, MarkovRandomField, marginal_oracles, junction_tree


def shuffled_strata_sampler():
    """Create an in-memory research variant; installed library is unchanged."""
    original = MarkovRandomField.synthetic_data
    source = textwrap.dedent(inspect.getsource(original))
    original_line = 'perm = np.argsort(inverse, kind="stable")'
    source = source.replace(
        original_line,
        'row_order = np.random.permutation(total)\n'
        '                perm = row_order[np.argsort(inverse[row_order], kind="stable")]',
        1,
    )
    namespace = dict(original.__globals__)
    exec(compile(source, "<public-toy-shuffled-strata>", "exec"), namespace)
    return namespace["synthetic_data"]


def audit(cliques, label):
    domain = Domain.fromdict({"H": 2, "A": 8, "B": 8, "C": 8})
    potentials = CliqueVector.zeros(domain, cliques)
    marginals = marginal_oracles.message_passing_stable(potentials)
    model = MarkovRandomField(potentials=potentials, marginals=marginals, total=1)
    _, elimination_order = junction_tree.make_junction_tree(domain, [set(c) for c in cliques])
    methods = {}
    patched = shuffled_strata_sampler()
    for method in ["round", "sample", "round_shuffled_strata"]:
        results = []
        for seed in range(10):
            np.random.seed(seed)
            if method == "round_shuffled_strata":
                frame = patched(model, rows=1049, method="round").df
            else:
                frame = model.synthetic_data(rows=1049, method=method).df
            c = frame[["A", "B", "C"]].corr()
            s = frame[["A", "B", "C"]].corr(method="spearman")
            corrs = {a + ":" + b: float(c.loc[a, b]) for a, b in [("A", "B"), ("A", "C"), ("B", "C")]}
            spearman = {a + ":" + b: float(s.loc[a, b]) for a, b in [("A", "B"), ("A", "C"), ("B", "C")]}
            results.append({"seed": seed, "correlations": corrs, "spearman_correlations": spearman})
        methods[method] = results
    return {"label": label, "cliques": cliques, "sampling_order": elimination_order[::-1], "methods": methods}


def main():
    report = {
        "description": "Uniform model: H,A,B,C are mutually independent, so true pairwise correlations are zero.",
        "correlation_method": "pearson",
        "additional_correlation_method": "spearman",
        "rows_per_sample": 1049,
        "seeds": list(range(10)),
        "versions": {name: version(name) for name in ["opendp", "mbi", "numpy"]},
        "runs": [
            audit([("H", "A"), ("H", "B"), ("H", "C")], "uniform_star"),
            audit([("H",), ("A",), ("B",), ("C",)], "uniform_disconnected"),
        ],
    }
    target = Path(__file__).with_name("toy_sampler_results.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for run in report["runs"]:
        print(run["label"], "sampling_order", run["sampling_order"])
        for method, results in run["methods"].items():
            values = np.array([list(row["correlations"].values()) for row in results])
            print(method, "mean correlations", values.mean(axis=0).tolist(), "min", float(values.min()), "max", float(values.max()))
            spearman = np.array([list(row["spearman_correlations"].values()) for row in results])
            print(method, "Spearman min", float(spearman.min()), "max", float(spearman.max()))


if __name__ == "__main__":
    main()
