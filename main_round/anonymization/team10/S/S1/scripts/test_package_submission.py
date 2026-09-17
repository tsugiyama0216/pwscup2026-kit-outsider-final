"""Boundary checks for submission rejection; uses synthetic in-memory records only."""
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from package_submission import COLUMNS, ValidationError, check_csv, verify_zip


class PackageValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.schema = json.loads((root.parents[1] / "participant_data/schema.json").read_text(encoding="utf-8"))
        cls.config = json.loads((root / "config/aim_main_team10_S1.json").read_text(encoding="utf-8"))

    def row(self):
        return [40, 0, "東京都", 10, 40, 1, 10, 0.1, 0, 40, 1 / 365.25, 0, 0]

    def raw(self, rows, columns=COLUMNS):
        handle = io.StringIO(newline="")
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
        return handle.getvalue().encode("utf-8")

    def validate(self, rows, columns=COLUMNS):
        return check_csv(self.raw(rows, columns), self.schema, self.config, expected_rows=len(rows))

    def test_duplicate_rows_are_reported_not_silently_dropped(self):
        result = self.validate([self.row(), self.row()])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["duplicate_rows"], 1)

    def test_finite_and_nonnumeric_checks(self):
        for bad in ("nan", "inf", "-inf", "oops", ""):
            with self.subTest(bad=bad):
                row = self.row()
                row[COLUMNS.index("BMI")] = bad
                with self.assertRaises(ValidationError):
                    self.validate([row])

    def test_time_floor_and_horizon_are_inclusive(self):
        for time in (1 / 365.25, 10):
            row = self.row()
            row[COLUMNS.index("time")] = time
            self.assertTrue(self.validate([row])["ok"])
        for time in (math.nextafter(1 / 365.25, 0), math.nextafter(10, math.inf)):
            row = self.row()
            row[COLUMNS.index("time")] = time
            with self.assertRaises(ValidationError):
                self.validate([row])

    def test_integer_binary_and_exclusive_outcome_checks(self):
        for name, bad in (("age", 40.5), ("sex", 2), ("smoking", 0.5)):
            row = self.row()
            row[COLUMNS.index(name)] = bad
            with self.assertRaises(ValidationError):
                self.validate([row])
        row = self.row()
        row[-2:] = [1, 1]
        with self.assertRaises(ValidationError):
            self.validate([row])

    def test_column_order_and_missing_fields_rejected(self):
        with self.assertRaises(ValidationError):
            self.validate([self.row()], list(reversed(COLUMNS)))
        with self.assertRaises(ValidationError):
            self.validate([self.row()[:-1]])

    def test_wrong_row_count_rejected(self):
        with self.assertRaises(ValidationError):
            check_csv(self.raw([self.row()]), self.schema, self.config)

    def test_zip_requires_exact_members_and_source_bytes(self):
        csv_bytes = self.raw([self.row()])
        fake_token = b"synthetic-test-token-no-real-credential\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("C.csv", csv_bytes)
                archive.writestr("token.txt", fake_token)
            self.assertTrue(verify_zip(path, csv_bytes, fake_token)["ok"])
            with self.assertRaises(ValidationError):
                verify_zip(path, csv_bytes + b"\n", fake_token)
            with self.assertRaises(ValidationError):
                verify_zip(path, csv_bytes, b"different-synthetic-token")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "a") as archive:
                    archive.writestr("C.csv", csv_bytes)
            with self.assertRaises(ValidationError):
                verify_zip(path, csv_bytes, fake_token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
