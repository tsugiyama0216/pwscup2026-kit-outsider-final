"""R1 regression tests; toy data only, no contest tokens."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

import generate as r1


def toy_data(n=400):
    rng = np.random.default_rng(71)
    age = rng.integers(40, 75, n)
    sex = rng.integers(0, 2, n)
    bmi = np.clip(rng.normal(24, 3, n), 15, 38)
    fpg = np.clip(90 + 2 * (bmi - 24) + rng.normal(0, 10, n), 50, 180)
    onset_time = rng.exponential(40 * np.exp(-0.02 * (fpg - 90)), n)
    death_time = rng.exponential(100 * np.exp(-0.02 * (age - 55)), n)
    b = pd.DataFrame({"age": age, "sex": sex, "prefecture": rng.choice(["A", "B", "C"], n),
                      "BMI": bmi, "SBP": 80 + age + rng.normal(0, 8, n),
                      "TG": rng.lognormal(4.6, 0.4, n), "HDL": rng.normal(50, 5, n),
                      "ALT": rng.lognormal(3, 0.3, n), "smoking": rng.integers(0, 2, n), "FPG": fpg})
    return pd.concat([b, r1.competing_outcomes(onset_time, death_time, 10)], axis=1)[r1.COLUMNS]


CFG = {"population": {"age_range": [40, 74]}, "onset": {"horizon_years": 10},
       "schema": {"ranges": {"BMI": [10, 55], "SBP": [40, 250], "TG": [1, 3000],
                              "HDL": [10, 160], "ALT": [0.1, 700], "FPG": [40, 400]}}}


class CopulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = toy_data()
        cls.c, cls.strata = r1.synthesize(cls.b, CFG)

    def test_shape_order_and_no_identifier(self):
        self.assertEqual(self.c.shape, (len(self.b), 13))
        self.assertEqual(self.c.columns.tolist(), r1.COLUMNS)

    def test_finite_allowed_values(self):
        self.assertFalse(self.c.isna().any().any())
        self.assertTrue(np.isfinite(self.c.drop(columns="prefecture")).all().all())
        for col, (lo, hi) in CFG["schema"]["ranges"].items():
            self.assertTrue(self.c[col].between(lo, hi).all())
        self.assertTrue(self.c.age.between(40, 74).all())
        self.assertTrue((self.c.age == np.floor(self.c.age)).all())

    def test_source_stratum_counts_preserved(self):
        pd.testing.assert_series_equal(self.b.groupby(["sex", "smoking"]).size(),
                                       self.c.groupby(["sex", "smoking"]).size())

    def test_no_exact_source_or_duplicate_rows(self):
        self.assertEqual(len(self.c.merge(self.b, on=r1.COLUMNS)), 0)
        self.assertEqual(int(self.c.duplicated().sum()), 0)

    def test_outcome_consistency(self):
        self.assertTrue((self.c.onset + self.c.death <= 1).all())
        censored = (self.c.onset + self.c.death) == 0
        self.assertTrue((self.c.loc[censored, "time"] == 10).all())
        self.assertTrue(self.c.time.between(r1.TIME_FLOOR, 10).all())

    def test_reproducible_and_ignores_private_labels_identifiers(self):
        decorated = self.b.assign(record_id=np.arange(len(self.b)), is_rare=True)
        c, _ = r1.synthesize(decorated, CFG)
        pd.testing.assert_frame_equal(self.c, c, check_exact=True)

    def test_changed_seed_changes_output(self):
        c, _ = r1.synthesize(self.b, CFG, seed=123)
        self.assertFalse(c.equals(self.c))

    def test_sparse_stratum_fallback(self):
        b = self.b.copy()
        b["smoking"] = 0
        b.loc[0, "smoking"] = 1
        c, strata = r1.synthesize(b, CFG)
        self.assertTrue(any(s["fallback"] == "sex" for s in strata))
        self.assertEqual(len(c), len(b))

    def test_no_event_training_data(self):
        b = self.b.assign(onset=0, death=0, time=10.0)
        c, _ = r1.synthesize(b, CFG)
        self.assertTrue((c.time == 10).all())
        self.assertEqual(int(c.onset.sum() + c.death.sum()), 0)

    def test_constant_columns_correlation(self):
        corr = r1.correlation(np.ones((30, 7)))
        np.testing.assert_allclose(corr, np.eye(7), atol=1e-12)
        self.assertTrue((np.linalg.eigvalsh(corr) > 0).all())

    def test_copula_preserves_monotone_relation(self):
        b = toy_data(1000)
        b["SBP"] = 3 * b["BMI"] + 40
        c = r1.sample_copula(b, 2000, np.random.default_rng(2))
        self.assertGreater(c[["BMI", "SBP"]].corr(method="spearman").iloc[0, 1], 0.9)

    def test_hazard_inversion_flat_tail_and_no_extrapolation(self):
        target = np.array([0.05, 0.15, 0.25])
        got = r1.invert_hazard(np.ones(3), np.array([1., 2., 3., 4.]),
                               np.array([0.1, 0.1, 0.2, 0.2]), np.exp(-target))
        np.testing.assert_equal(got, [1., 3., np.inf])

    def test_competing_risks_ties_and_censoring(self):
        got = r1.competing_outcomes(np.array([2., 5., np.inf, 10., 0.]),
                                    np.array([4., 3., np.inf, 10., 2.]), 10)
        self.assertEqual(got.onset.tolist(), [1, 0, 0, 1, 1])
        self.assertEqual(got.death.tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(got.time.tolist(), [2, 3, 10, 10, r1.TIME_FLOOR])

    def test_csv_round_trip(self):
        raw = self.c.to_csv(index=False, lineterminator="\n").encode("utf-8")
        c = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
        pd.testing.assert_frame_equal(self.c, c, check_exact=True)

    def test_package_exact_bytes_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            raw, token = b"a,b\n1,2\n", b"TEST-ONLY-NON-CREDENTIAL\r\n"
            r1.write_package(path, raw, token)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), ["C.csv", "token.txt"])
                self.assertEqual(archive.read("C.csv"), raw)
                self.assertEqual(archive.read("token.txt"), token)
            with self.assertRaises(FileExistsError):
                r1.write_package(path, raw, token)


if __name__ == "__main__":
    unittest.main()
