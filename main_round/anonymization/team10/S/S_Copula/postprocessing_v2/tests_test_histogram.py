"""Unit tests and explicitly PUBLIC-ONLY replicated histogram experiment."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dp_copula as dc
from histogram_refine import geometric_variance, refine_models, sample_refined


def public_model(proxy, public, schema):
    categories = sorted(public.prefecture.unique())
    defaults = {c: float(public[c].median()) for c in dc.NUMERIC}
    public = dc._prepare(public, schema, categories, defaults)
    proxy = dc._prepare(proxy, schema, categories, defaults)
    model = dc._public_spec(public, schema, bins=8)
    model.update(version=dc.VERSION, privacy=dc.privacy_plan(8), schema=schema,
                 public_rows=len(public), categories=categories, public_defaults=defaults,
                 bins=8, public_support={c:[float(public[c].min()), float(public[c].max())]
                                        for c in dc.NUMERIC},
                 default_sample_options=dict(dc.DEFAULT_SAMPLE))
    gid = dc._group_ids(proxy, schema["rare_detector"])
    model["dp_group_counts"] = np.bincount(gid, minlength=24).tolist()
    model["dp_prefecture_counts"] = proxy.prefecture.value_counts().reindex(categories, fill_value=0).tolist()
    counts = public.prefecture.value_counts().reindex(categories, fill_value=0).to_numpy()
    model["public_prefecture_probabilities"] = (counts/counts.sum()).tolist()
    for g, spec in enumerate(model["groups"]):
        for col in dc.NUMERIC:
            spec["margins"][col]["dp_counts"] = np.histogram(
                proxy.loc[gid == g, col], bins=spec["margins"][col]["edges"])[0].tolist()
    return model


def public_noise(truth, seed):
    """Seeded noise is ONLY for public A_bg simulations, never actual B."""
    rng = np.random.default_rng(seed)
    models = []
    for total_epsilon in [1, 3, 10]:
        model = copy.deepcopy(truth)
        model["privacy"] = dc.privacy_plan(str(total_epsilon*0.8))
        qs = {q["name"]:q for q in model["privacy"]["queries"]}
        def noisy(values, q):
            p = 1-q["geometric_q_numerator"]/q["geometric_q_denominator"]
            y = np.asarray(values)
            return (y+rng.geometric(p, y.shape)-rng.geometric(p, y.shape)).tolist()
        for name in ["group_counts", "prefecture_counts"]:
            model["dp_"+name] = noisy(model["dp_"+name], qs[name])
        for spec in model["groups"]:
            for col in dc.NUMERIC:
                spec["margins"][col]["dp_counts"] = noisy(spec["margins"][col]["dp_counts"], qs["marginal_"+col])
        models.append({"copula":model, "accounting":{"epsilon_exact":str(total_epsilon)}})
    return models


def cox_beta(frame, scale):
    x = frame[["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]].copy()
    x["TG"] = np.log(x.TG.clip(lower=1e-6))
    x["ALT"] = np.log(x.ALT.clip(lower=1e-6))
    x["time"] = frame.time.to_numpy()
    x["onset"] = frame.onset.to_numpy()
    model = CoxPHFitter(penalizer=0.0).fit(x, duration_col="time", event_col="onset")
    return model.params_.to_numpy()*scale


def experiment():
    data = HERE.parents[2]/"participant_data"
    frame = pd.read_csv(data/"A_bg.csv")
    schema = json.loads((data/"schema.json").read_text(encoding="utf-8"))
    rows = []
    for split in [471, 472]:
        perm = np.random.default_rng(split).permutation(len(frame))
        proxy = frame.iloc[perm[:1049]].reset_index(drop=True)
        public = frame.iloc[perm[1049:]].reset_index(drop=True)
        truth = public_model(proxy, public, schema)
        cols = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]
        transformed = public[cols].copy()
        transformed["TG"] = np.log(transformed.TG)
        transformed["ALT"] = np.log(transformed.ALT)
        scale = transformed.std().to_numpy()
        truth_beta = cox_beta(proxy, scale)
        sd = public[dc.NUMERIC].std().to_numpy()
        truth_event_means = np.array([proxy.loc[proxy.onset == ev, dc.NUMERIC].mean().to_numpy() for ev in [0, 1]])
        for draw in range(3):
            releases = public_noise(truth, split*100+draw)
            for label, mult in [("frozen_eps10", None), ("bayes_0.5", 0.5), ("bayes_1", 1.0), ("bayes_2", 2.0), ("bayes_4", 4.0)]:
                if mult is None:
                    model = releases[-1]["copula"]
                    g = dc._simplex(dc._released_float(model["dp_group_counts"]), dc.PRIVATE_N)
                    pred = []
                    for gid, spec in enumerate(model["groups"]):
                        for col in dc.NUMERIC:
                            margin = spec["margins"][col]
                            alloc = float(next(q["epsilon_effective_upper"] for q in model["privacy"]["queries"] if q["name"] == "marginal_"+col))
                            prior = 15*2/alloc*np.sqrt(len(margin["dp_counts"]))
                            p = (dc._simplex(dc._released_float(margin["dp_counts"]), g[gid])+prior*np.array(margin["public_probabilities"]))/max(g[gid]+prior, 1e-12)
                            pred.extend(g[gid]*p)
                    synthetic = dc.sample(model, n=10490, seed=917,
                        marginal_prior_noise_multiplier=15, public_radial_strength=0)
                else:
                    model = refine_models(releases, prior_multiplier=mult)
                    g = np.asarray(model["posterior_group_counts"])
                    pred = [c for spec in model["groups"] for col in dc.NUMERIC for c in spec["margins"][col]["posterior_counts"]]
                    synthetic = sample_refined(model, n=10490, seed=917, public_radial_strength=0)
                true = [c for spec in truth["groups"] for col in dc.NUMERIC for c in spec["margins"][col]["dp_counts"]]
                event_means = np.array([synthetic.loc[synthetic.onset == ev, dc.NUMERIC].mean().to_numpy() for ev in [0, 1]])
                row = {"split":split, "draw":draw, "method":label,
                    "histogram_count_rmse":float(np.sqrt(np.mean((np.array(pred)-np.array(true))**2))),
                    "group_count_rmse":float(np.sqrt(np.mean((g-np.array(truth["dp_group_counts"]))**2))),
                    "event_mean_standardized_rmse":float(np.sqrt(np.mean(((event_means-truth_event_means)/sd)**2))),
                    "cox_standardized_l2_error":float(np.linalg.norm(cox_beta(synthetic, scale)-truth_beta))}
                rows.append(row)
            print(f"Public split {split}, noise draw {draw}: complete", flush=True)
    summary = pd.DataFrame(rows).groupby("method").agg({k:["mean", "std"] for k in rows[0] if k not in ["split", "draw", "method"]})
    means = {method:{metric:{stat:float(summary.loc[method,(metric,stat)]) for stat in ["mean", "std"]}
                    for metric in rows[0] if metric not in ["split", "draw", "method"]} for method in summary.index}
    result = {"provenance":"PUBLIC A_bg only, 2 disjoint-within-split 1049-row pseudo-private cohorts, 3 independent noise draws each; seeded noise used only in public simulation",
              "public_prior_source":"disjoint remainder of A_bg in each split", "selected_prior_multiplier":1.0,
              "selection":"multiplier 1 is the public multinomial sampling model; parameter sweep is sensitivity analysis, no private evaluation",
              "synthetic_sample_rows":10490, "public_proxy_results":rows, "summary":means}
    (HERE/"public_histogram_experiment.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)


class HistogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = HERE.parents[2]/"participant_data"
        frame = pd.read_csv(data/"A_bg.csv")
        schema = json.loads((data/"schema.json").read_text(encoding="utf-8"))
        cls.truth = public_model(frame.iloc[:1049], frame.iloc[1049:], schema)
        cls.releases = public_noise(cls.truth, 196)

    def test_variance(self):
        self.assertAlmostEqual(geometric_variance({"geometric_q_numerator":2,"geometric_q_denominator":3}), 12)

    def test_counts_composition_deterministic_and_no_mutation(self):
        before = copy.deepcopy(self.releases)
        result = refine_models(self.releases)
        self.assertEqual(self.releases, before)
        self.assertEqual(result, refine_models(self.releases))
        self.assertEqual(result["histogram_postprocessing"]["parent_epsilon_exact"], "14")
        self.assertEqual(sum(result["dp_group_counts"]), 1049)
        for g, spec in enumerate(result["groups"]):
            for margin in spec["margins"].values():
                self.assertEqual(sum(margin["dp_counts"]), result["dp_group_counts"][g])
                self.assertAlmostEqual(sum(margin["posterior_counts"]), result["posterior_group_counts"][g])
                self.assertAlmostEqual(sum(margin["posterior_probabilities"]), 1)
                self.assertGreaterEqual(min(margin["posterior_counts"]), 0)
        c = sample_refined(result, seed=16)
        self.assertEqual(len(c), 1049)
        self.assertFalse(c.isna().any().any())

    def test_query_identity_and_duplicate_guards(self):
        with self.assertRaises(ValueError):
            refine_models([self.releases[0], self.releases[0]])
        broken = copy.deepcopy(self.releases)
        broken[1]["copula"]["groups"][0]["margins"]["age"]["edges"][1] += 0.01
        with self.assertRaises(ValueError):
            refine_models(broken)


if __name__ == "__main__":
    if "--experiment" in sys.argv:
        experiment()
    else:
        unittest.main()
