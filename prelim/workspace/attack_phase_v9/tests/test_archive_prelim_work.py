import unittest
import io
import zipfile

from archive_prelim_work import (
    ROOT, WORKSPACE_ROOT, WORKSPACE_PREFIX, MANIFEST, PATHSPEC,
    KIT_SAMPLE_TOKEN_PATHS, KIT_SAMPLE_ZIP_PATHS, assert_staged_scope,
    excluded_reason, is_research_path, scan_payload, scan_existing_sample_zip,
)


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

    def test_relocated_research_paths_and_windows_separators(self):
        for suffix in ("REPORT.md", "experiment.py", "results.csv", "report.json"):
            for prefix in ("local-output/", f"{WORKSPACE_PREFIX}/local-output/"):
                path = prefix + suffix
                self.assertIsNone(excluded_reason(path))
                self.assertTrue(is_research_path(path))
                self.assertTrue(is_research_path(path.replace("/", "\\")))
        for suffix in ("result.zip", "cache.npy", "figure.png", "paths.nul"):
            self.assertIsNotNone(excluded_reason(f"{WORKSPACE_PREFIX}/local-output/{suffix}"))

    def test_nested_environment_caches_and_relocated_tokens_are_excluded(self):
        for path in (
            "prelim/workspace/starter/.venv/Scripts/python.exe",
            "prelim/workspace/opendp_aim/.venv/pyvenv.cfg",
            "prelim/workspace/.tools/uv/uv.exe",
            "prelim/workspace/.uv-cache/file.json",
            "prelim/workspace/attack_phase_v9/__pycache__/module.pyc",
            "prelim/workspace/library.egg-info/PKG-INFO",
            "prelim/workspace/.ipynb_checkpoints/notebook.ipynb",
            "prelim/anonymization/anonymization-submitted-v5/token.txt",
            "prelim/attack/final/submission.zip",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(excluded_reason(path))

    def test_manifest_and_pathspec_are_relative_to_git_root(self):
        self.assertEqual(WORKSPACE_PREFIX, "prelim/workspace")
        self.assertEqual(ROOT / WORKSPACE_PREFIX, WORKSPACE_ROOT)
        self.assertEqual(ROOT / MANIFEST,
                         WORKSPACE_ROOT / "research_archive/prelim-2026-manifest.json")
        self.assertEqual(ROOT / PATHSPEC,
                         WORKSPACE_ROOT / "local-output/prelim-archive-pathspec.nul")

    def test_root_docs_and_curated_prelim_collection_are_allowed(self):
        for path in ("README.md", "main_round/README.md", "prelim/attack/final/F_mia.csv",
                     "prelim/old_attack_study_materials/README.md"):
            self.assertIsNone(excluded_reason(path))
            self.assertFalse(is_research_path(path))

    def test_index_scope_allows_only_known_kit_token_exceptions(self):
        expected = {"README.md", *KIT_SAMPLE_ZIP_PATHS}
        indexed = expected | KIT_SAMPLE_TOKEN_PATHS
        assert_staged_scope(indexed, expected)
        for path in (
            "prelim/anonymization/anonymization-submitted-v5/token.txt",
            "other_directory/TOKEN.TXT",
            "submissions/prelim/team24/submitted-v5/token.txt",
            "prelim/workspace/local-output/attack-v9/final.zip",
            "prelim/workspace/starter/.venv/pyvenv.cfg",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                assert_staged_scope(indexed | {path}, expected | {path})

    def test_index_scope_rejects_uninspected_additions(self):
        with self.assertRaisesRegex(ValueError, "absent from the inspected manifest"):
            assert_staged_scope({"README.md", "uninspected.py"}, {"README.md"})

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
