"""Public-only derivative and posterior tests."""
from pathlib import Path
import json
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from influence_target import public_influence, fit_public_regression, estimate_target, dp_cox


class InfluenceTests(unittest.TestCase):
    def test_weighted_efron_derivative_including_ties(self):
        frame = pd.read_csv(HERE.parents[2]/"participant_data"/"A_bg.csv").iloc[:900].copy()
        # Explicitly create tied event sets in this public mathematical test.
        frame["time"] = np.maximum(.1, np.round(frame.time.to_numpy(), 1))
        nuisance = dp_cox.build_public_cox_model(frame)
        influence, info = public_influence(frame, nuisance)
        x = (dp_cox._features(frame)-nuisance["center"])/nuisance["scale"]
        design = pd.DataFrame(x, columns=dp_cox.COX_COLUMNS)
        design["time"] = frame.time.to_numpy()
        design["onset"] = frame.onset.to_numpy()
        design["case_weight"] = 1.0
        tied_event = frame.index[(frame.onset == 1) & frame.time.duplicated(keep=False)][0]
        for row in [int(tied_event), int(np.flatnonzero(frame.onset.to_numpy() == 0)[0])]:
            fits = []
            step = .01
            for sign in [1, -1]:
                design.loc[row, "case_weight"] = 1+sign*step
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = CoxPHFitter().fit(design, duration_col="time", event_col="onset", weights_col="case_weight",
                        fit_options={"precision":1e-10, "r_precision":1e-10})
                fits.append(fit.params_.to_numpy())
            design.loc[row, "case_weight"] = 1.0
            derivative = (fits[0]-fits[1])/(2*step)
            np.testing.assert_allclose(derivative, influence[row]/len(frame), atol=3e-6, rtol=3e-3)

    def test_public_posterior_psd_and_no_mutation(self):
        frame = pd.read_csv(HERE.parents[2]/"participant_data"/"A_bg.csv")
        seed = 2026091811
        order = np.random.default_rng(seed).permutation(len(frame))
        public = frame.iloc[order[1049:]].reset_index(drop=True)
        folder = HERE.parents[6]/"public_postprocessing_v2"/f"cohort{seed}_noise0"
        models = [json.loads((folder/f"epsilon{e}.json").read_text(encoding="utf-8")) for e in [1,3,10]]
        before = json.dumps(models, sort_keys=True)
        nuisance = models[0]["public_cox"]
        for schema in [None, models[0]["copula"]["schema"]]:
            reg = fit_public_regression(public, nuisance, schema=schema)
            result = estimate_target(models, reg)
            self.assertEqual(result["parent_epsilon_exact"], "14")
            self.assertEqual(result["additional_epsilon_exact"], "0")
            self.assertGreaterEqual(np.linalg.eigvalsh(result["posterior_beta_covariance"]).min(), -1e-10)
            self.assertEqual(result, estimate_target(models, reg))
        self.assertEqual(before, json.dumps(models, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
