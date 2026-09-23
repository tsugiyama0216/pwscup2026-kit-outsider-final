"""PUBLIC/synthetic-only mathematical and preservation tests."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from cox_optimizer import EfronRiskSets, calibrate, features, _fit
import dp_cox


class OptimizerTests(unittest.TestCase):
    def test_exact_efron_information_and_score_jacobian(self):
        rng = np.random.default_rng(654)
        x = rng.normal(size=(90, 9))
        time = np.repeat(np.arange(1, 16), 6)
        event = rng.binomial(1, .6, len(time))
        beta = rng.normal(size=9) * .15
        risk = EfronRiskSets(time, event)
        score, info, jacobian = risk.evaluate(x, beta, information=True, score_jacobian=True)
        step = 1e-5
        for j in range(9):
            change = np.eye(9)[j] * step
            finite = (risk.evaluate(x, beta + change)[0] - risk.evaluate(x, beta - change)[0]) / (2*step)
            np.testing.assert_allclose(finite, -info[:, j], rtol=1e-7, atol=1e-7)
        for row, col in [(0, 2), (33, 4), (89, 7)]:
            x[row, col] += step
            plus = risk.evaluate(x, beta)[0]
            x[row, col] -= 2*step
            minus = risk.evaluate(x, beta)[0]
            x[row, col] += step
            np.testing.assert_allclose((plus-minus)/(2*step), jacobian[row, :, col], rtol=1e-6, atol=1e-7)
        fitted, _ = _fit(x, time, event)
        np.testing.assert_allclose(risk.evaluate(x, fitted)[0], 0, atol=1e-3)

    def test_public_data_exact_invariants_and_determinism(self):
        data = HERE.parents[2] / "participant_data"
        public = pd.read_csv(data / "A_bg.csv").iloc[:1600]
        nuisance = dp_cox.build_public_cox_model(public.iloc[700:])
        frame = public.iloc[:500].drop(columns=["record_id"], errors="ignore").reset_index(drop=True)
        before = frame.copy(deep=True)
        target = np.asarray(nuisance["beta_standardized"])
        groups = (frame.age.to_numpy() > 58).astype(int)
        result, diag = calibrate(frame, target, nuisance, strata=groups, iterations=50)
        repeated, repeated_diag = calibrate(frame, target, nuisance, strata=groups, iterations=50)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(result, repeated)
        self.assertEqual(diag, repeated_diag)
        predictors = [c for c in frame if c not in ["time", "onset", "death"]]
        pd.testing.assert_frame_equal(result[predictors], before[predictors])
        for group in [0, 1]:
            cols = ["time", "onset", "death"]
            a = result.loc[groups == group, cols].sort_values(cols).reset_index(drop=True)
            b = before.loc[groups == group, cols].sort_values(cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)
        self.assertTrue(np.all(np.diff(diag["objective_history"]) < 0))
        self.assertLess(diag["selected_coefficient_l2_error"], diag["initial_coefficient_l2_error"])


def public_experiment():
    """One held-out PUBLIC 1049-row cohort; proxy-oracle checks solver ceiling.

    The oracle target is only an experimental control, never a release target.
    Parent pipeline separately tests its deployed DP target on public splits.
    """
    from histogram_refine import refine_models, sample_refined
    from tests_test_histogram import public_model, public_noise
    data = HERE.parents[2] / "participant_data"
    root = next(p for p in HERE.parents if (p / "work" / "official-kit").exists())
    sys.path.insert(0, str(root / "work" / "official-kit" / "starter" / "src"))
    from pwscup2026.scoring.utility import _score_u_spec
    frame = pd.read_csv(data / "A_bg.csv")
    schema = json.loads((data / "schema.json").read_text(encoding="utf-8"))
    order = np.random.default_rng(471).permutation(len(frame))
    proxy = frame.iloc[order[:1049]].reset_index(drop=True)
    public = frame.iloc[order[1049:]].reset_index(drop=True)
    nuisance = dp_cox.build_public_cox_model(public)
    truth = public_model(proxy, public, schema)
    refined = refine_models(public_noise(truth, 47100))
    synthetic = sample_refined(refined, seed=917, public_radial_strength=0)
    x = (features(proxy)-nuisance["center"])/nuisance["scale"]
    target, _ = _fit(x, proxy.time.to_numpy(), proxy.onset.to_numpy())
    detector = schema["rare_detector"]
    centered = synthetic[detector["columns"]].to_numpy()-detector["mu"]
    rare = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(detector["sigma"]), centered) >= detector["band"]["lo"]**2
    calibrated, diagnostics = calibrate(synthetic, target, nuisance, strata=rare)
    before, _ = _score_u_spec(synthetic, proxy)
    after, _ = _score_u_spec(calibrated, proxy)
    report = {
        "provenance": "PUBLIC A_bg split seed471, first1049 proxy, remaining public nuisance; seeded synthetic noise47100",
        "target": "PUBLIC proxy exact coefficient ORACLE control, never used for any private candidate",
        "before_official_U_spec": before, "after_official_U_spec": after,
        "diagnostics": diagnostics,
        "interpretation": "Tests optimizer ceiling only. DP target estimation error remains and can dominate deployed utility.",
    }
    (HERE / "public_optimizer_experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "diagnostics"}, indent=2))
    print("Cox L2", diagnostics["initial_coefficient_l2_error"], diagnostics["selected_coefficient_l2_error"])


if __name__ == "__main__":
    if "--experiment" in sys.argv:
        public_experiment()
    else:
        unittest.main()
