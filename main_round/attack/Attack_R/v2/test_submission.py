"""Apply the frozen R v1 malformed-submission suite to R v2 in memory."""
import io
import sys
import unittest
import runtime as rt
from run import load_validator, production_check

validate_package, load_actual = load_validator()
saved_run = sys.modules.get("run")
sys.modules["run"] = type(sys)("run")
sys.modules["run"].validate_package = validate_package
try:
    original = rt.load_file("_rv2_old_submission_tests", rt.HERE.parent/"v1/test_submission.py")
finally:
    if saved_run is None:
        del sys.modules["run"]
    else:
        sys.modules["run"] = saved_run


class V2SubmissionTests(original.SubmissionTests):
    @classmethod
    def setUpClass(cls):
        production_check()
        cls.runtime = load_actual()
        cls.w, cls.np, cls.pd, cls.validator, cls.cfg, cls.dist, *_ = cls.runtime
        cls.output = rt.HERE/"submission"
        cls.content = {name: (cls.output/name).read_bytes() for name in ("F_mia.csv", "F_aia.csv")}
        cls.mia = cls.pd.read_csv(io.BytesIO(cls.content["F_mia.csv"]), float_precision="round_trip")
        cls.aia = cls.pd.read_csv(io.BytesIO(cls.content["F_aia.csv"]), float_precision="round_trip")

    def test_real_zip(self):
        self.assertTrue(validate_package(self.output/"team10_R_v2_attack.zip", self.content, self.runtime)["ok"])

    def test_bad_zip_members_csv_and_token_are_rejected_in_memory(self):
        import zipfile
        from unittest.mock import patch
        real = zipfile.ZipFile
        def open_zip(path, *args, **kwargs):
            if isinstance(path, type(self.output)) and path.name == "team10_R_v1_attack.zip":
                path = self.output/"team10_R_v2_attack.zip"
            return real(path, *args, **kwargs)
        with patch.object(original.zipfile, "ZipFile", side_effect=open_zip):
            super().test_bad_zip_members_csv_and_token_are_rejected_in_memory()


if __name__ == "__main__":
    unittest.main()
