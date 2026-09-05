from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from attack_phase_v5.generate_submission_v5 import (
    verify_input_hashes,
    verify_targets_manifest,
)
from attack_phase_v5.blend_feedback_winners import (
    blend_target_winners,
    load_aia_per_target,
)
from attack_phase_v5.package_submission_v5 import package_candidate


class V5InputIntegrityTests(unittest.TestCase):
    def test_input_hash_check_accepts_exact_and_rejects_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes(b"stable\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            with patch(
                "attack_phase_v5.generate_submission_v5.EXPECTED_INPUT_HASHES",
                {"sample": digest},
            ):
                self.assertEqual(verify_input_hashes({"sample": path}), {"sample": digest})
                path.write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                    verify_input_hashes({"sample": path})

    def test_targets_manifest_verifies_named_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "target.csv"
            payload.write_bytes(b"x\n1\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / "SHA256SUMS.txt").write_text(
                f"{digest}  target.csv\n", encoding="utf-8"
            )
            observed = verify_targets_manifest(root)
            self.assertEqual(observed["target.csv"], digest.upper())
            payload.write_bytes(b"x\n2\n")
            with self.assertRaisesRegex(ValueError, "public target hash mismatch"):
                verify_targets_manifest(root)

    def test_packager_keeps_flat_exact_files_and_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            mia = b"record_id,1\n1,0.5\n"
            aia = b"target_k,challenge_row_id,time_hat\n1,1,10\n"
            (candidate / "F_mia.csv").write_bytes(mia)
            (candidate / "F_aia.csv").write_bytes(aia)
            token = root / "token.txt"
            token.write_text("practice-team-token\n", encoding="utf-8")
            targets = root / "targets"
            targets.mkdir()
            output = root / "submission.zip"
            with (
                patch(
                    "attack_phase_v5.package_submission_v5.config_mod.load_config",
                    return_value={},
                ),
                patch(
                    "attack_phase_v5.package_submission_v5.official_validate.validate_submission_zip",
                    return_value=SimpleNamespace(ok=True, errors=[]),
                ),
            ):
                result = package_candidate(candidate, token, output, targets)
            self.assertEqual(result["official_validator"], "PASS")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["F_aia.csv", "F_mia.csv", "token.txt"],
                )
                self.assertEqual(archive.read("F_mia.csv"), mia)
                self.assertEqual(archive.read("F_aia.csv"), aia)
            first_hash = result["archive_sha256"]
            with (
                patch(
                    "attack_phase_v5.package_submission_v5.config_mod.load_config",
                    return_value={},
                ),
                patch(
                    "attack_phase_v5.package_submission_v5.official_validate.validate_submission_zip",
                    return_value=SimpleNamespace(ok=True, errors=[]),
                ),
            ):
                repeated = package_candidate(candidate, token, output, targets)
            self.assertEqual(repeated["archive_sha256"], first_hash)

    def test_feedback_blend_copies_whole_winning_target_blocks(self) -> None:
        columns = ["target_k", "challenge_row_id", "time_hat"]
        left = pd.DataFrame(
            [[1, 10, 1.0], [1, 11, 2.0], [2, 20, 3.0], [2, 21, 4.0]],
            columns=columns,
        )
        right = pd.DataFrame(
            [[1, 10, 5.0], [1, 11, 6.0], [2, 20, 7.0], [2, 21, 8.0]],
            columns=columns,
        )
        output, detail = blend_target_winners(
            {"left": left, "right": right},
            {"left": {"1": 0.9, "2": 0.1}, "right": {"1": 0.2, "2": 0.8}},
            target_ids=("1", "2"),
            rows_per_target=2,
        )
        self.assertEqual(detail["winner_counts"], {"left": 1, "right": 1})
        self.assertEqual(output["time_hat"].tolist(), [1.0, 2.0, 7.0, 8.0])

    def test_feedback_score_loader_accepts_nested_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(
                json.dumps({"_breakdown": {"aia_per_target": {"1": 0.25}}}),
                encoding="utf-8",
            )
            self.assertEqual(load_aia_per_target(path), {"1": 0.25})

    def test_feedback_score_loader_skips_official_numeric_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(
                json.dumps(
                    {
                        "mia_power": 0.2,
                        "aia_power": 0.3,
                        "_breakdown": {"aia_per_target": {"1": 0.25}},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_aia_per_target(path), {"1": 0.25})


if __name__ == "__main__":
    unittest.main()
