"""R2 integration and packaging tests using artificial data and fake tokens only."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

import evaluation
import generate as r2
from survival import TIME_FLOOR


CFG = {"population": {"age_range": [40, 74]}, "onset": {"horizon_years": 10},
       "schema": {"ranges": {"BMI": [10, 55], "SBP": [40, 250], "TG": [1, 3000],
                              "HDL": [10, 160], "ALT": [0.1, 700], "FPG": [40, 400]}}}


def toy_data(n=500):
    """An independent artificial cohort with correlated FPG/BMI and two causes."""
    rng = np.random.default_rng(71)
    age = rng.integers(40, 75, n)
    bmi = np.clip(rng.normal(24, 3, n), 15, 38)
    fpg = np.clip(90 + 2 * (bmi - 24) + rng.normal(0, 10, n), 50, 180)
    onset_time = rng.exponential(25 * np.exp(-0.03 * (fpg - 90)), n)
    death_time = rng.exponential(65 * np.exp(-0.03 * (age - 55)), n)
    b = pd.DataFrame({"age": age, "sex": rng.integers(0, 2, n),
                      "prefecture": rng.choice(["A", "B", "C"], n), "BMI": bmi,
                      "SBP": 80 + age + rng.normal(0, 8, n),
                      "TG": rng.lognormal(4.6, 0.4, n), "HDL": rng.normal(50, 5, n),
                      "ALT": rng.lognormal(3, 0.3, n), "smoking": rng.integers(0, 2, n),
                      "FPG": fpg})
    b["onset"] = ((onset_time <= death_time) & (onset_time <= 10)).astype(int)
    b["death"] = ((death_time < onset_time) & (death_time <= 10)).astype(int)
    b["time"] = np.maximum(np.minimum(np.minimum(onset_time, death_time), 10), TIME_FLOOR)
    return b[r2.COLUMNS]


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = toy_data()
        cls.c, cls.audit = r2.synthesize(cls.b, CFG)

    def test_shape_column_order_and_no_identifier(self):
        self.assertEqual(self.c.shape, (len(self.b), 13))
        self.assertEqual(self.c.columns.tolist(), r2.COLUMNS)
        self.assertEqual(self.c.index.tolist(), list(range(len(self.b))))

    def test_finite_allowed_values(self):
        self.assertFalse(self.c.isna().any().any())
        self.assertTrue(np.isfinite(self.c.drop(columns="prefecture").to_numpy()).all())
        for column, (lo, hi) in CFG["schema"]["ranges"].items():
            self.assertTrue(self.c[column].between(lo, hi).all(), column)
        self.assertTrue(self.c.age.between(40, 74).all())
        self.assertTrue((self.c.age == np.floor(self.c.age)).all())
        for column in ("sex", "smoking", "onset", "death"):
            self.assertTrue(self.c[column].isin([0, 1]).all(), column)
        self.assertTrue(self.c.prefecture.isin(self.b.prefecture.unique()).all())

    def test_sex_smoking_stratum_counts_preserved(self):
        pd.testing.assert_series_equal(self.b.groupby(["sex", "smoking"]).size(),
                                       self.c.groupby(["sex", "smoking"]).size())
        expected = self.b.groupby(["sex", "smoking"]).size().to_dict()
        recorded = {(entry["sex"], entry["smoking"]): entry["n"]
                    for entry in self.audit["strata"]}
        self.assertEqual(recorded, expected)

    def test_outcome_invariants_survive_final_assembly(self):
        self.assertTrue((self.c.onset + self.c.death <= 1).all())
        censored = (self.c.onset + self.c.death) == 0
        self.assertTrue((self.c.loc[censored, "time"] == 10).all())
        self.assertTrue(self.c.time.between(TIME_FLOOR, 10).all())
        for cause in ("onset", "death"):
            self.assertEqual(int(self.c[cause].sum()), int(self.b[cause].sum()))

    def test_no_exact_source_or_duplicate_rows(self):
        self.assertEqual(len(self.c.merge(self.b, on=r2.COLUMNS)), 0)
        self.assertEqual(int(self.c.duplicated().sum()), 0)

    def test_fixed_seed_reproduces_table_and_audit(self):
        c, audit = r2.synthesize(self.b, CFG, seed=r2.SEED)
        pd.testing.assert_frame_equal(self.c, c, check_exact=True)
        self.assertEqual(audit, self.audit)

    def test_identifiers_private_labels_and_original_index_ignored(self):
        decorated = self.b.assign(record_id=np.arange(len(self.b)) + 7000,
                                  is_rare=np.arange(len(self.b)) % 2 == 0)
        decorated.index = pd.Index(np.arange(len(self.b)) * 7 + 9, name="source_index")
        c, audit = r2.synthesize(decorated, CFG, seed=r2.SEED)
        pd.testing.assert_frame_equal(self.c, c, check_exact=True)
        self.assertEqual(audit, self.audit)

    def test_input_frame_and_configuration_unchanged(self):
        b, cfg = self.b.copy(deep=True), copy.deepcopy(CFG)
        before = b.copy(deep=True)
        r2.synthesize(b, cfg)
        pd.testing.assert_frame_equal(before, b, check_exact=True)
        self.assertEqual(cfg, CFG)

    def test_changed_seed_changes_release(self):
        c, _ = r2.synthesize(self.b, CFG, seed=123)
        self.assertFalse(c.equals(self.c))

    def test_covariates_use_no_outcomes(self):
        c, audit = r2.synthesize_covariates(self.b, CFG)
        modified = self.b.assign(onset=0, death=0, time=10.0)
        other, other_audit = r2.synthesize_covariates(modified, CFG)
        pd.testing.assert_frame_equal(c, other, check_exact=True)
        self.assertEqual(audit, other_audit)

    def test_covariate_fallback_preserves_sparse_stratum_size(self):
        b = self.b.assign(smoking=0)
        b.loc[0, "smoking"] = 1
        c, audit = r2.synthesize_covariates(b, CFG)
        self.assertTrue(any(entry["fallback"] == "sex" for entry in audit))
        pd.testing.assert_series_equal(b.groupby(["sex", "smoking"]).size(),
                                       c.groupby(["sex", "smoking"]).size())

    def test_covariate_fallback_for_small_sex_group(self):
        b = self.b.assign(sex=0, smoking=0)
        b.loc[0, ["sex", "smoking"]] = 1
        c, audit = r2.synthesize_covariates(b, CFG)
        rare_group = next(entry for entry in audit if entry["sex"] == 1)
        self.assertEqual(rare_group["fallback"], "all")
        self.assertEqual(rare_group["fit_n"], len(b))
        self.assertEqual(int((c.sex == 1).sum()), 1)

    def test_copula_preserves_strong_monotone_relation(self):
        b = toy_data(1000)
        b["SBP"] = 3 * b["BMI"] + 40
        c = r2.sample_copula(b, 2000, np.random.default_rng(2))
        self.assertGreater(c[["BMI", "SBP"]].corr(method="spearman").iloc[0, 1], 0.9)

    def test_constant_covariates_have_finite_correlation(self):
        corr = r2.correlation(np.ones((30, len(r2.CONT))))
        np.testing.assert_allclose(corr, np.eye(len(r2.CONT)), atol=1e-12)
        self.assertTrue((np.linalg.eigvalsh(corr) > 0).all())

    def test_csv_exact_float_round_trip(self):
        raw = self.c.to_csv(index=False, lineterminator="\n").encode("utf-8")
        loaded = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
        pd.testing.assert_frame_equal(self.c, loaded, check_exact=True)
        self.assertEqual(loaded.to_csv(index=False, lineterminator="\n").encode("utf-8"), raw)


class PackageTests(unittest.TestCase):
    def test_flat_members_exact_bytes_and_reproducible_archive(self):
        raw = "prefecture,value\n東京,1.25\n".encode("utf-8")
        token = b"TEST-ONLY-NON-CREDENTIAL\r\n"
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "nested" / "first.zip"
            second = Path(temp) / "second.zip"
            evaluation.write_package(first, raw, token)
            evaluation.write_package(second, raw, token)
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), ["C.csv", "token.txt"])
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read("C.csv"), raw)
                self.assertEqual(archive.read("token.txt"), token)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_existing_package_refused_and_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            raw, token = b"a,b\n1,2\n", b"TEST-ONLY-NON-CREDENTIAL\r\n"
            evaluation.write_package(path, raw, token)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                evaluation.write_package(path, b"changed data", b"fake changed token")
            self.assertEqual(path.read_bytes(), before)


class OfficialValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the pinned local kit, but never call check_environment(), which
        # reads contest inputs. Every table and credential below is artificial.
        sys.path.insert(0, str(evaluation.KIT / "starter/src"))
        from pwscup2026.common import submission_schema
        from pwscup2026.kit import validate
        cls.schema, cls.validator = submission_schema, validate
        cls.cfg = evaluation.yaml.safe_load(
            (evaluation.KIT / "starter/src/pwscup2026/kit/kit_config.yaml").read_text(encoding="utf-8"))
        cls.b = toy_data(80)
        cls.c = cls.b.copy(deep=True)
        cls.c["BMI"] += 0.123

    def test_rejects_wrong_row_count_before_reference_access(self):
        with patch.object(self.schema, "read_scoring_csv") as read_reference:
            with self.assertRaisesRegex(ValueError, "Invalid synthetic table"):
                evaluation.evaluate_frame(self.c.iloc[:-1], self.b, self.cfg)
            read_reference.assert_not_called()

    def test_rejects_schema_and_outcome_errors_before_reference_access(self):
        wrong_order = self.c[self.c.columns[::-1]]
        invalid_time = self.c.copy()
        invalid_time.loc[0, "time"] = 0.0
        non_admin = self.c.copy()
        non_admin.loc[0, ["onset", "death", "time"]] = [0, 0, 9.0]
        for frame in (self.c.assign(record_id=1), wrong_order, invalid_time,
                      self.c.assign(onset=1, death=1), non_admin):
            with self.subTest(columns=list(frame.columns), first_time=frame.time.iloc[0]):
                with patch.object(self.schema, "read_scoring_csv") as read_reference:
                    with self.assertRaises(ValueError):
                        evaluation.evaluate_frame(frame, self.b, self.cfg)
                    read_reference.assert_not_called()

    def test_rejects_exact_source_rows_and_synthetic_duplicates(self):
        duplicate = self.c.copy()
        duplicate.iloc[1] = duplicate.iloc[0]
        for frame in (self.b.copy(), duplicate):
            with self.subTest(source_copy=frame.equals(self.b)):
                with patch.object(self.schema, "read_scoring_csv") as read_reference:
                    with self.assertRaisesRegex(ValueError, "Exact source rows or duplicate"):
                        evaluation.evaluate_frame(frame, self.b, self.cfg)
                    read_reference.assert_not_called()

    def test_release_rejects_csv_and_token_byte_mismatches(self):
        raw = self.c.to_csv(index=False, lineterminator="\n").encode("utf-8")
        fake_token = b"FAKETESTTOKEN123456789012\r\n"
        for mismatch in ("csv", "token"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data, output = root / "fake_data", root / "release"
                data.mkdir()
                output.mkdir()
                (data / "token.txt").write_bytes(fake_token)
                (output / "C.csv").write_bytes(raw + (b"\n" if mismatch == "csv" else b""))
                archive_token = b"OTHERFAKETOKEN123456789012" if mismatch == "token" else fake_token
                evaluation.write_package(output / "submission" / evaluation.ZIP_NAME, raw, archive_token)
                with patch.object(evaluation, "DATA", data), patch.object(evaluation, "evaluate_frame") as evaluate:
                    with self.assertRaisesRegex(ValueError, "CSV differ|credential bytes differ"):
                        evaluation.evaluate_release(output, self.b, self.cfg)
                    evaluate.assert_not_called()

    def test_official_zip_dist_loader_needs_explicit_r2_row_check(self):
        # The kit only discovers defense_rows from B_practice.csv, while real
        # main-round allocations use B.csv. R2's n_expected=len(b) is essential.
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            self.b.to_csv(data / "B.csv", index=False)
            self.assertNotIn("defense_rows", self.validator.load_dist(data))
            self.b.to_csv(data / "B_practice.csv", index=False)
            self.assertEqual(self.validator.load_dist(data)["defense_rows"], len(self.b))


class CommandProtectionTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(evaluation.KIT / "starter/src"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        self.b = toy_data(40)
        self.raw = self.b.to_csv(index=False, lineterminator="\n").encode("utf-8")
        self.provenance = {"official_kit_commit": "fake-pin", "scoring_config_sha256": "fake-config",
                           "rulebook_sha256": "fake-rulebook", "inputs": {"B.csv": "fake-input"},
                           "dependencies": {"pandas": "fake-version"}, "python": "fake-python"}
        self.source_hashes = {"generate.py": "fake-generator", "survival.py": "fake-survival",
                              "evaluation.py": "fake-evaluation"}
        self.score = {"utility": 0.75, "facets": {"U_gen": 0.75}}
        self.validation = {"C_sha256": "fake-csv", "ZIP_sha256": "fake-zip"}
        self.saved = {"provenance": {**copy.deepcopy(self.provenance), "source_sha256": self.source_hashes},
                      "score": self.score, "validation": self.validation}
        (self.output / "C.csv").write_bytes(self.raw)
        self.write_report()
        for target, replacement in (("HERE", self.output), ("check_environment", None),
                                    ("read_training_data", None), ("source_hashes", None),
                                    ("evaluate_release", None), ("synthesize", None), ("write_package", None)):
            patcher = patch.object(r2, target, replacement) if replacement is not None else patch.object(r2, target)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if replacement is None:
                setattr(self, target, mocked)
        self.check_environment.return_value = (CFG, copy.deepcopy(self.provenance))
        self.read_training_data.return_value = self.b
        self.source_hashes.return_value = self.saved["provenance"]["source_sha256"]
        self.evaluate_release.return_value = (self.score, self.validation)
        self.synthesize.return_value = (self.b, {})

    def write_report(self):
        (self.output / "report.json").write_text(json.dumps(self.saved), encoding="utf-8")

    def invoke(self, *args):
        with patch.object(sys, "argv", ["generate.py", *args]), patch("builtins.print"):
            r2.main()

    def test_verify_only_validates_without_generation_or_writes(self):
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.invoke("--verify-only")
        self.evaluate_release.assert_called_once()
        self.synthesize.assert_not_called()
        self.write_package.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_verify_rejects_each_provenance_change_before_evaluation(self):
        for field in self.provenance:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.provenance)
                changed[field] = "changed"
                self.check_environment.return_value = (CFG, changed)
                with self.assertRaises(RuntimeError):
                    self.invoke("--verify-only")
                self.evaluate_release.assert_not_called()
                self.synthesize.assert_not_called()

    def test_verify_rejects_changed_generator_and_frozen_artifact(self):
        self.source_hashes.return_value = {"changed": "source"}
        with self.assertRaisesRegex(RuntimeError, "implementation changed"):
            self.invoke("--verify-only")
        self.evaluate_release.assert_not_called()
        self.source_hashes.return_value = self.saved["provenance"]["source_sha256"]
        self.evaluate_release.return_value = (self.score, {**self.validation, "C_sha256": "changed"})
        with self.assertRaisesRegex(RuntimeError, "Frozen R2 artifact changed"):
            self.invoke("--verify-only")

    def test_verify_rejects_changed_self_score(self):
        for score in ({**self.score, "utility": 0.5}, {**self.score, "facets": {"U_gen": 0.5}}):
            self.evaluate_release.return_value = (score, self.validation)
            with self.assertRaisesRegex(RuntimeError, "self-score changed"):
                self.invoke("--verify-only")

    def test_reproduce_check_compares_exact_csv_bytes_without_writes(self):
        self.invoke("--reproduce-check")
        self.synthesize.assert_called_once()
        self.write_package.assert_not_called()
        altered = self.b.copy()
        altered.loc[0, "BMI"] += 0.001
        self.synthesize.return_value = (altered, {})
        with self.assertRaisesRegex(RuntimeError, "fixed-seed reproduction differs"):
            self.invoke("--reproduce-check")
        self.assertEqual((self.output / "C.csv").read_bytes(), self.raw)

    def test_existing_release_is_not_overwritten(self):
        before = (self.output / "C.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.synthesize.assert_not_called()
        self.write_package.assert_not_called()
        self.assertEqual((self.output / "C.csv").read_bytes(), before)

    def test_output_outside_model_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, "must remain inside"):
            self.invoke("--output", str(self.output.parent / "outside-model"))
        self.synthesize.assert_not_called()
        self.write_package.assert_not_called()


if __name__ == "__main__":
    unittest.main()
