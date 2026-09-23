"""PUBLIC/synthetic-only calibration tests and oracle control."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "postprocessing_v2"),
               str(HERE.parents[6] / "official-kit" / "starter" / "src")]
from target_calibration import score_information_derivatives, calibrate_target, EfronRiskSets


class TargetCalibrationTests(unittest.TestCase):
    def test_information_derivative_including_ties(self):
        rng = np.random.default_rng(821)
        x = rng.normal(size=(90, 9))
        time = np.repeat(np.arange(15), 6)
        event = rng.binomial(1, .6, len(x))
        beta = rng.normal(size=9)*.2
        risk = EfronRiskSets(time, event)
        _, _, _, b, c = score_information_derivatives(x, beta, risk)
        for row in [0, 21, 89]:
            direction = rng.normal(size=9)
            predicted = np.outer(direction, b[row])+np.outer(b[row], direction)+(direction@beta)*c[row]
            step = 1e-5
            x[row] += step*direction
            plus = risk.evaluate(x, beta, information=True)[1]
            x[row] -= 2*step*direction
            minus = risk.evaluate(x, beta, information=True)[1]
            x[row] += step*direction
            np.testing.assert_allclose((plus-minus)/(2*step), predicted, atol=2e-7, rtol=2e-6)

    def test_exact_stratum_preservation_and_determinism(self):
        from dp_cox_target import build_public_target
        frame = pd.read_csv(HERE.parents[2]/"participant_data"/"A_bg.csv")
        schema = json.loads((HERE.parents[2]/"participant_data"/"schema.json").read_text(encoding="utf-8"))
        nuisance = build_public_target(frame.iloc[600:1600], public_n=600)
        target = {"beta_standardized": nuisance["beta_standardized"],
                  "standard_errors_standardized": nuisance["expected_se_standardized"]}
        original = frame.iloc[:600].drop(columns=["record_id"]).reset_index(drop=True)
        before = original.copy(deep=True)
        detector = schema["rare_detector"]
        result, trace = calibrate_target(original, target, nuisance, detector, iterations=10, candidates=128, trials=3)
        repeated, repeated_trace = calibrate_target(original, target, nuisance, detector, iterations=10, candidates=128, trials=3)
        pd.testing.assert_frame_equal(before, original)
        pd.testing.assert_frame_equal(result, repeated)
        self.assertEqual(trace, repeated_trace)
        static = [c for c in original if c not in ["time", "onset", "death"]]
        pd.testing.assert_frame_equal(result[static], original[static])
        centered = original[detector["columns"]].to_numpy()-detector["mu"]
        rare = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(detector["sigma"]), centered) >= detector["band"]["lo"]**2
        cols = ["time", "onset", "death"]
        for value in [False, True]:
            a = original.loc[rare == value, cols].sort_values(cols).reset_index(drop=True)
            b = result.loc[rare == value, cols].sort_values(cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)
        self.assertTrue(np.all(np.diff(trace["objective_history"]) < 0))


def oracle_experiment():
    from pipeline import construct
    from pwscup2026.scoring.utility import _fit_cox, _score_u_spec
    frame = pd.read_csv(HERE.parents[2]/"participant_data"/"A_bg.csv")
    settings = json.loads((HERE.parent/"postprocessing_v2"/"joint14_config.json").read_text(encoding="utf-8"))["settings"]
    rows = []
    for seed in [2026091811, 2026091827]:
        order = np.random.default_rng(seed).permutation(len(frame))
        held = frame.iloc[order[:1049]].sort_values("record_id").reset_index(drop=True)
        folder = HERE.parents[6]/"public_postprocessing_v2"/f"cohort{seed}_noise0"
        models = [json.loads((folder/f"epsilon{e}.json").read_text(encoding="utf-8")) for e in [1,3,10]]
        if any(not m.get("source", "").startswith("PUBLIC A_bg proxy") for m in models):
            raise ValueError("Only public caches allowed")
        _, _, original = construct(models, settings)
        public = models[0]["public_cox"]
        detector = models[0]["copula"]["schema"]["rare_detector"]
        true_fit = _fit_cox(held)
        target = {"beta_standardized": (true_fit.params_.to_numpy()*public["scale"]).tolist(),
                  "standard_errors_standardized": (true_fit.standard_errors_.to_numpy()*public["scale"]).tolist()}
        result, trace = calibrate_target(original, target, public, detector)
        score, detail = _score_u_spec(result, held)
        predictors = [k for k in original.columns if k not in ["time", "onset", "death"]]
        pd.testing.assert_frame_equal(result[predictors], original[predictors])
        before_hash = pd.util.hash_pandas_object(original.sort_values(["time", "onset", "death"])[["time", "onset", "death"]], index=False).sum()
        after_hash = pd.util.hash_pandas_object(result.sort_values(["time", "onset", "death"])[["time", "onset", "death"]], index=False).sum()
        assert before_hash == after_hash
        assert np.all(np.diff(trace["objective_history"]) < 0)
        row = {"public_cohort_seed": seed, "official_U_spec": score, "detail": detail, "trace": trace}
        rows.append(row)
        print(json.dumps({"seed": seed, "U_spec": score, "beta_error": trace["selected_coefficient_l2_error"],
                          "SE_log_RMSE": trace["selected_se_log_rmse"], "swaps": trace["swaps"]}), flush=True)
    payload = {"source": "PUBLIC A_bg held-out cohorts, true PUBLIC beta and SE oracle control at infinite epsilon",
               "actual_private_access": False, "not_a_released_private_target": True, "rows": rows}
    (HERE/"public_oracle_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if "--oracle" in sys.argv:
        oracle_experiment()
    else:
        unittest.main()
