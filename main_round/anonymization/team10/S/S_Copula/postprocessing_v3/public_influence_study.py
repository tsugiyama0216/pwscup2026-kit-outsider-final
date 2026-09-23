"""Four fixed PUBLIC validation draws; never opens an actual release/cohort."""
from pathlib import Path
import json
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[2]
WORK = HERE.parents[6]
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "postprocessing_v2"),
                str(WORK / "official-kit" / "starter" / "src")]
from influence_target import fit_public_regression, estimate_target
from joint_cox import combine_cox
from pwscup2026.scoring.utility import _fit_cox, _io_overlap


def official_io_contributions(beta, se, true_beta, true_se):
    z = norm.ppf(.975)
    result = []
    for b, s, tb, ts in zip(beta, se, true_beta, true_se):
        significant, true_significant = abs(b) > z*s, abs(tb) > z*ts
        io = _io_overlap(tb-z*ts, tb+z*ts, b-z*s, b+z*s, tb, b, ts)
        result.append(0.0 if (significant != true_significant or
                             (significant and np.sign(b) != np.sign(tb))) else io)
    return result


def study():
    source = TEAM / "participant_data" / "A_bg.csv"
    frame = pd.read_csv(source)
    rows = []
    for seed in (2026091811, 2026091827):
        order = np.random.default_rng(seed).permutation(len(frame))
        held_out = frame.iloc[order[:1049]].sort_values("record_id").reset_index(drop=True)
        public = frame.iloc[order[1049:]].reset_index(drop=True)
        fitted_truth = _fit_cox(held_out)
        regression = None
        group_regression = None
        for replicate in range(2):
            folder = WORK / "public_postprocessing_v2" / f"cohort{seed}_noise{replicate}"
            models = [json.loads((folder/f"epsilon{e}.json").read_text(encoding="utf-8")) for e in (1, 3, 10)]
            if any(m.get("source") != "PUBLIC A_bg proxy only; does not consume protected B budget" for m in models):
                raise ValueError("Only clearly marked PUBLIC proxy caches are accepted")
            nuisance = models[0]["public_cox"]
            if regression is None:
                regression = fit_public_regression(public, nuisance)
                group_regression = fit_public_regression(public, nuisance, schema=models[0]["copula"]["schema"])
            influence = estimate_target(models, regression)
            influence_groups = estimate_target(models, group_regression)
            baseline = combine_cox(models)
            true_beta = fitted_truth.params_.to_numpy()*np.asarray(nuisance["scale"])
            true_se = fitted_truth.standard_errors_.to_numpy()*np.asarray(nuisance["scale"])
            public_se = np.asarray(influence["target_standard_error_public"])
            for label, estimate in [("public_only", nuisance), ("v2_fixed_hazard", baseline),
                                    ("v3_public_influence", influence), ("v3_public_influence_groups", influence_groups)]:
                beta = np.asarray(estimate["beta_standardized"])
                contributions = official_io_contributions(beta, public_se, true_beta, true_se)
                row = {
                    "cohort_seed": seed, "noise_replicate": replicate, "method": label,
                    "standardized_coefficient_l2_error": float(np.linalg.norm(beta-true_beta)),
                    "standardized_coefficient_se_rmse": float(np.sqrt(np.mean(((beta-true_beta)/true_se)**2))),
                    "official_IO_mean_public_SE": float(np.mean(contributions)),
                    "official_IO_contributions_public_SE": contributions,
                    "true_public_proxy_beta_standardized": true_beta.tolist(),
                    "target_beta_standardized": beta.tolist(),
                    "true_public_proxy_SE": true_se.tolist(), "predicted_public_SE": public_se.tolist(),
                }
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if not isinstance(v, list)}), flush=True)
    summary = {}
    for label in ["public_only", "v2_fixed_hazard", "v3_public_influence", "v3_public_influence_groups"]:
        chosen = [r for r in rows if r["method"] == label]
        summary[label] = {k:float(np.mean([r[k] for r in chosen])) for k in [
            "standardized_coefficient_l2_error", "standardized_coefficient_se_rmse", "official_IO_mean_public_SE"]}
    improved_l2 = summary["v3_public_influence"]["standardized_coefficient_l2_error"] < summary["v2_fixed_hazard"]["standardized_coefficient_l2_error"]
    improved_io = summary["v3_public_influence"]["official_IO_mean_public_SE"] > summary["v2_fixed_hazard"]["official_IO_mean_public_SE"]
    result = {
        "source": "PUBLIC A_bg only; fixed two cohorts x two cached noise draws",
        "fit_scope": "Each regression fitted only on corresponding 4094-row disjoint public background",
        "target_truth_usage": "Held-out public Cox truth is used only for validation metrics, never covariance training",
        "extra_private_queries": 0, "actual_release_evaluated": False,
        "metric_scope": "Official IO contribution formula evaluated on estimated target beta plus public-information SE; not a synthetic whole-utility score",
        "rows": rows, "summary": summary,
        "supports_v3_over_v2_on_both_primary_metrics": bool(improved_l2 and improved_io),
        "hashes": {str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, HERE/"influence_target.py", Path(__file__)]},
    }
    (HERE/"public_influence_study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    study()
