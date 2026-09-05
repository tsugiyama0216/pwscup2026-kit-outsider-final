from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v9.registered_linkage import (
    RegistrationConfig,
    cross_view_anchor_audit,
    overlay_registered_mia,
    registered_event_times,
    registered_linkage_scores,
)


def _synthetic_world(seed: int = 9, *, planted: bool = True):
    rng = np.random.default_rng(seed)
    n = 220
    source = pd.DataFrame(
        {
            "age": rng.integers(35, 80, n),
            "BMI": rng.normal(23, 3.2, n),
            "SBP": rng.normal(128, 14, n),
            "TG": rng.lognormal(4.7, 0.38, n),
            "HDL": rng.normal(57, 12, n),
            "ALT": rng.lognormal(3.1, 0.42, n),
            "FPG": rng.normal(99, 12, n),
            "sex": rng.integers(0, 2, n),
            "prefecture": rng.choice(["A", "B", "C", "D", "E", "F"], n),
            "smoking": rng.integers(0, 2, n),
        }
    )
    slopes = np.array([1.04, 0.93, 1.08, 0.88, 1.03, 1.11, 0.96])
    intercepts = np.array([2.0, 1.5, -4.0, 9.0, -1.0, 0.8, 4.0])
    numeric = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
    # Null rows have the correct marginals and categorical distribution, so
    # recovery cannot be won merely by marginal alignment.
    null_source = source.sample(n=n, replace=True, random_state=seed + 1).reset_index(drop=True)
    release = null_source.copy()
    release[numeric] = release[numeric].to_numpy(float) * slopes + intercepts
    for column in numeric:
        release[column] += rng.normal(0, 0.08 * np.std(release[column]), n)
    release[["sex", "prefecture", "smoking"]] = pd.DataFrame(
        {
            "sex": rng.integers(0, 2, n),
            "prefecture": rng.choice(["A", "B", "C", "D", "E", "F"], n),
            "smoking": rng.integers(0, 2, n),
        }
    )
    planted_indices = np.arange(70, 110)
    if planted:
        release_rows = np.arange(10, 50)
        release.loc[release_rows, numeric] = (
            source.loc[planted_indices, numeric].to_numpy(float) * slopes
            + intercepts
            + rng.normal(0, [0.02, 0.01, 0.03, 0.08, 0.02, 0.02, 0.03], (40, 7))
        )
        release.loc[release_rows, ["sex", "prefecture", "smoking"]] = source.loc[
            planted_indices, ["sex", "prefecture", "smoking"]
        ].to_numpy()
    known_negative = np.zeros(n, dtype=bool)
    known_negative[:60] = True
    protected = np.zeros(n, dtype=bool)
    protected[110:120] = True
    return source, release, known_negative, protected, set(planted_indices.tolist())


class RegisteredLinkageTests(unittest.TestCase):
    def test_planted_affine_links_are_precise(self):
        source, release, negative, protected, planted = _synthetic_world()
        result = registered_linkage_scores(
            source,
            release,
            negative,
            protected_mask=protected,
            config=RegistrationConfig(iterations=3, max_tentative_anchors=140),
        )
        selected = set(np.flatnonzero(result.accepted).tolist())
        true_positive = len(selected & planted)
        self.assertGreaterEqual(true_positive, 12)
        self.assertGreaterEqual(true_positive / max(len(selected), 1), 0.80)
        self.assertEqual(int(result.accepted[negative | protected].sum()), 0)
        self.assertLessEqual(result.diagnostics["audit_false_links"], 2)
        holdout = cross_view_anchor_audit(
            result.model_a,
            source,
            release,
            validation_columns=("BMI", "HDL", "FPG"),
            permutations=32,
        )
        self.assertEqual(holdout["status"], "ok")
        self.assertLess(holdout["residual_ratio"], 0.75)

    def test_null_source_swap_does_not_create_a_large_cluster(self):
        source, release, negative, protected, _ = _synthetic_world(seed=31, planted=False)
        result = registered_linkage_scores(
            source,
            release,
            negative,
            protected_mask=protected,
            config=RegistrationConfig(iterations=2, max_tentative_anchors=120),
        )
        self.assertLessEqual(int(result.accepted.sum()), 5)
        self.assertLessEqual(result.diagnostics["audit_false_link_rate"], 0.05)

    def test_registered_event_decoder_is_normal_format(self):
        source, release, negative, protected, _ = _synthetic_world()
        release["time"] = 11.0
        release["onset"] = 0
        release["death"] = 0
        release.loc[10:14, "time"] = np.arange(1.0, 6.0)
        release.loc[10:14, "onset"] = 1
        result = registered_linkage_scores(source, release, negative, protected_mask=protected)
        decoded, audit = registered_event_times(source.iloc[70:75], release, result.model_a)
        self.assertEqual(decoded.shape, (5,))
        self.assertTrue(np.isfinite(decoded).all())
        self.assertTrue((decoded >= 0).all())
        self.assertGreaterEqual(audit["decoded_event_rows"], 1)

    def test_mia_overlay_preserves_private_and_rare(self):
        source, _, negative, _, _ = _synthetic_world()
        n = len(source)
        base = pd.DataFrame({"record_id": np.arange(n), "3": np.linspace(0, 1, n)})
        rare = np.zeros(n, dtype=bool)
        rare[70] = True
        accepted = np.zeros(n, dtype=bool)
        accepted[[0, 70, 71]] = True
        confidence = np.arange(n, dtype=float)
        output, audit = overlay_registered_mia(
            base,
            "3",
            accepted,
            confidence,
            known_negative_mask=negative,
            rare_mask=rare,
        )
        self.assertEqual(output.loc[0, "3"], base.loc[0, "3"])
        self.assertEqual(output.loc[70, "3"], base.loc[70, "3"])
        self.assertGreater(output.loc[71, "3"], base.loc[71, "3"])
        self.assertEqual(audit["changed_rows"], 1)
        with self.assertRaises(ValueError):
            overlay_registered_mia(
                base,
                "1",
                accepted,
                confidence,
                known_negative_mask=negative,
                rare_mask=rare,
            )


if __name__ == "__main__":
    unittest.main()
