"""Check the actual result and rejection of damaged submissions, without sending it."""
import io
import unittest
import warnings
import zipfile

from common import HERE, load_actual
from run import validate_package


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_actual()
        cls.w, cls.np, cls.pd, cls.validator, cls.cfg, cls.dist, *_ = cls.runtime
        cls.output = HERE / "submission"
        cls.content = {name: (cls.output / name).read_bytes()
                       for name in ("F_mia.csv", "F_aia.csv")}
        cls.mia = cls.pd.read_csv(io.BytesIO(cls.content["F_mia.csv"]), float_precision="round_trip")
        cls.aia = cls.pd.read_csv(io.BytesIO(cls.content["F_aia.csv"]), float_precision="round_trip")

    def validate(self, mia, aia):
        return self.w.validate_frames(self.np, self.pd, self.validator, mia, aia, self.cfg, self.dist)

    def test_real_zip(self):
        result = validate_package(self.output / "team10_R_v1_attack.zip", self.content, self.runtime)
        self.assertTrue(result["ok"])

    def test_mia_invalid_variants_are_rejected(self):
        bad_id = self.mia.copy()
        bad_id.loc[0, "record_id"] = -999999
        duplicate = self.mia.copy()
        duplicate.loc[0, "record_id"] = duplicate.loc[1, "record_id"]
        fractional = self.mia.astype({"record_id": float})
        fractional.loc[0, "record_id"] += .5
        for label, bad in [("missing_target", self.mia.drop(columns=["0"])),
                           ("wrong_id", bad_id), ("duplicate_id", duplicate), ("fractional_id", fractional)]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.validate(bad, self.aia)
        for value in (-.1, 1.1, self.np.nan, self.np.inf):
            bad = self.mia.copy()
            bad.loc[0, "0"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(bad, self.aia)

    def test_aia_invalid_variants_are_rejected(self):
        for column, value in [("target_k", 10), ("challenge_row_id", -999999),
                              ("time_hat", -1.), ("time_hat", self.np.nan), ("time_hat", self.np.inf)]:
            bad = self.aia.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.validate(self.mia, bad)
        for bad in (self.aia.iloc[:-1].copy(), self.pd.concat([self.aia, self.aia.iloc[:1]], ignore_index=True)):
            with self.assertRaises(ValueError):
                self.validate(self.mia, bad)

    def test_bad_zip_members_csv_and_token_are_rejected_in_memory(self):
        with zipfile.ZipFile(self.output / "team10_R_v1_attack.zip") as archive:
            original = [(name, archive.read(name)) for name in archive.namelist()]
        variants = [original + [("extra.txt", b"extra")], original[:2],
                    original + [original[0]],
                    [(name, b"altered" if name == "F_mia.csv" else data) for name, data in original],
                    [(name, b"invalid-test-credential" if name == "token.txt" else data)
                     for name, data in original]]
        for items in variants:
            buffer = io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(buffer, "w") as archive:
                    for name, data in items:
                        archive.writestr(name, data)
            buffer.seek(0)
            with self.assertRaises(ValueError):
                validate_package(buffer, self.content, self.runtime)


if __name__ == "__main__":
    unittest.main()
