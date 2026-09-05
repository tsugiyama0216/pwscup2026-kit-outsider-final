import unittest
import io
import zipfile

from archive_prelim_work import excluded_reason, scan_payload, scan_existing_sample_zip


class PrivateArchiveTests(unittest.TestCase):
    def test_credentials_archives_and_cache_excluded(self):
        for path in ("submissions/prelim/team24/token.txt", "local-output/attack-v9/result.zip",
                     "attack_phase_v9/__pycache__/file.pyc", ".env"):
            with self.subTest(path=path):
                self.assertIsNotNone(excluded_reason(path))

    def test_research_sources_reports_and_csv_allowed(self):
        for path in ("local-output/attack-v9/REPORT.md", "local-output/experiment.py",
                     "local-output/results.csv", "local-output/report.json"):
            self.assertIsNone(excluded_reason(path))

    def test_exact_credential_match_is_detected_without_returning_value(self):
        fake = b"a-value-only-used-in-this-test"
        reasons = scan_payload(b"prefix " + fake + b" suffix", [fake])
        self.assertEqual(reasons, ["exact known credential value"])

    def test_common_key_pattern_and_clean_text(self):
        fake = b"ghp_" + b"a" * 30
        self.assertIn("credential/private-key pattern", scan_payload(fake, []))
        self.assertEqual(scan_payload(b'{"token_read": false, "score": 0.3407}', []), [])

    def test_compressed_sample_members_are_scanned(self):
        stream = io.BytesIO()
        fake = b"dummy-value-for-archive-test"
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("token.txt", fake)
        reasons = scan_existing_sample_zip(stream.getvalue(), [fake])
        self.assertEqual(reasons, ["credential/raw-data match inside existing sample archive"])


if __name__ == "__main__":
    unittest.main()
