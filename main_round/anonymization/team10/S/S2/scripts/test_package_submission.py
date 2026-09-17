"""Boundary checks for submission rejection; uses synthetic in-memory records only."""
import csv
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

from package_submission import COLUMNS, ValidationError, check_csv, check_lineage, digest, verify_zip


class PackageValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.schema = json.loads((root.parents[1] / "participant_data/schema.json").read_text(encoding="utf-8"))
        cls.config = json.loads((root / "config/aim_main_team10_S2.json").read_text(encoding="utf-8"))

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


class ReleaseLineageTests(unittest.TestCase):
    """Synthetic files exercise the S2 method and saved-model checks without private data."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "config").mkdir()
        (self.root / "scripts").mkdir()
        for relative, content in {
            "config/s1_sample_base.json": b"{}\n",
            "scripts/generate_pwscup.py": b"# synthetic generator\n",
            "scripts/model_snapshot.py": b"# synthetic model snapshot module\n",
            "uv.lock": b"# synthetic lock\n",
            "C.csv": b"synthetic csv bytes\n",
            "model_snapshot.npz": b"synthetic model snapshot bytes\n",
        }.items():
            (self.root / relative).write_bytes(content)
        self.originals = {"sha256": {"A_bg.csv": "public-background", "schema.json": "public-schema", "B.csv": "private-input"}}
        self.config = {
            "config_version": "synthetic-S2", "expected_input_rows": 1049, "output_rows": 1049,
            "synthesis": {"method": "sample"}, "privacy": {"rho": 0.1},
            "public_config_provenance": {
                "private_inputs_used": False,
                "base_config_sha256": digest(self.root / "config/s1_sample_base.json"),
                "public_background_sha256": "public-background", "public_schema_sha256": "public-schema",
            },
            "postprocess": {"public_outcome_donor": {"risk": {"source_background_sha256": "public-background"}}},
        }
        self.write("config/aim_main_team10_S2.json", self.config)
        hashes = {
            "config_sha256": digest(self.root / "config/aim_main_team10_S2.json"),
            "generator_sha256": digest(self.root / "scripts/generate_pwscup.py"),
            "snapshot_module_sha256": digest(self.root / "scripts/model_snapshot.py"),
            "public_background_sha256": "public-background", "public_schema_sha256": "public-schema",
            "output_sha256": digest(self.root / "C.csv"),
            "model_snapshot_sha256": digest(self.root / "model_snapshot.npz"),
        }
        self.report = {**hashes, "config_version": "synthetic-S2", "output_rows_public": 1049,
                       "dp_release_count": 1, "privacy": {"rho_zcdp": 0.1}}
        self.write("privacy_report.json", self.report)
        self.ledger = {
            **hashes, "status": "complete", "input_sha256": "private-input", "rho_zcdp": 0.1,
            "privacy_report_sha256": digest(self.root / "privacy_report.json"),
            "opendp_uv_lock_sha256": digest(self.root / "uv.lock"), "release_attempt_id": "synthetic-attempt",
        }
        self.write("release_ledger.json", self.ledger)

    def write(self, relative, value):
        (self.root / relative).write_text(json.dumps(value), encoding="utf-8")

    def test_valid_sample_release_and_saved_model(self):
        config, lineage = check_lineage(self.root, self.originals)
        self.assertEqual(config["synthesis"]["method"], "sample")
        self.assertEqual(lineage["model_snapshot_sha256"], digest(self.root / "model_snapshot.npz"))

    def test_round_release_rejected_before_packaging(self):
        config = deepcopy(self.config)
        config["synthesis"]["method"] = "round"
        self.write("config/aim_main_team10_S2.json", config)
        with self.assertRaisesRegex(ValidationError, "requires the sample"):
            check_lineage(self.root, self.originals)

    def test_modified_model_snapshot_rejected(self):
        (self.root / "model_snapshot.npz").write_bytes(b"modified synthetic model")
        with self.assertRaisesRegex(ValidationError, "model_snapshot_sha256"):
            check_lineage(self.root, self.originals)

    def test_modified_snapshot_module_rejected(self):
        (self.root / "scripts/model_snapshot.py").write_bytes(b"# modified synthetic snapshot module\n")
        with self.assertRaisesRegex(ValidationError, "snapshot_module_sha256"):
            check_lineage(self.root, self.originals)

    def test_report_and_ledger_must_both_record_model_snapshot(self):
        for filename, original in (("privacy_report.json", self.report), ("release_ledger.json", self.ledger)):
            with self.subTest(filename=filename):
                changed = deepcopy(original)
                changed.pop("model_snapshot_sha256")
                self.write(filename, changed)
                with self.assertRaisesRegex(ValidationError, "model_snapshot_sha256"):
                    check_lineage(self.root, self.originals)
                self.write(filename, original)


@unittest.skipUnless(shutil.which("powershell.exe"), "PowerShell runner requires Windows")
class RunnerGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "repo/main_round/anonymization/team10/S/S2"
        (self.root / "config").mkdir(parents=True)
        (self.root / "scripts").mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1] / "run.ps1", self.root / "run.ps1")
        self.config_path = self.root / "config/aim_main_team10_S2.json"
        self.config_path.write_bytes(b'{"synthetic_frozen_config": true}\n')
        self.ledger_path = self.root / "release_ledger.json"
        self.ledger_path.write_bytes(b'{"status": "synthetic-existing-release"}\n')
        self.config_hash = digest(self.config_path)
        self.ledger_hash = digest(self.ledger_path)
        (self.root / "scripts/generate_pwscup.py").write_text(
            "import sys\nassert '--validate-only' in sys.argv\nprint('synthetic preflight only')\n",
            encoding="utf-8",
        )

    def run_mode(self, mode):
        result = subprocess.run(
            [shutil.which("powershell.exe"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.root / "run.ps1"),
             "-Mode", mode, "-AimPython", sys.executable,
             "-ScorePython", str(self.root / "missing-public-builder-python.exe")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(digest(self.config_path), self.config_hash)
        self.assertEqual(digest(self.ledger_path), self.ledger_hash)
        return result

    def test_existing_release_preflight_skips_public_config_rebuild(self):
        result = self.run_mode("Preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("synthetic preflight only", result.stdout)

    def test_existing_release_generate_rejected_before_any_python(self):
        result = self.run_mode("Generate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("An S2 release ledger already exists", result.stderr)
        self.assertNotIn("synthetic preflight only", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
