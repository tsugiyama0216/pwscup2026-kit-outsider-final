from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attack_phase_daybreak.generate_submission_daybreak import (
    EXPECTED_MIA_TOTAL_SHARED,
    EXPECTED_SELF_TARGET,
    EXPECTED_TARGET_IDS,
    strict_extra_checks,
    validate_preliminary_settings,
    verify_sha256_manifest,
)


def settings(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "self_target_id": EXPECTED_SELF_TARGET,
        "exclude_targets": (EXPECTED_SELF_TARGET,),
        "mia_total_shared": EXPECTED_MIA_TOTAL_SHARED,
        "acknowledge_prelim_private_block_rule": True,
        "aia_active_fraction": 0.5,
        "aia_link_trust_distance": 0.2,
        "aia_anomalous_link_trust_distance": 0.08,
        "mia_transport_epsilon": 0.35,
        "aia_anomalous_event_rate": 0.5,
        "mia_density_weight": 0.35,
        "mia_rare_weight": 0.65,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DaybreakFlowTests(unittest.TestCase):
    def test_preliminary_settings_accept_expected_values(self) -> None:
        validate_preliminary_settings(settings())

    def test_preliminary_settings_reject_v3_default_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "508"):
            validate_preliminary_settings(settings(mia_total_shared=736))

    def test_rule_acknowledgement_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            validate_preliminary_settings(
                settings(acknowledge_prelim_private_block_rule=False)
            )

    def test_manifest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            payload = b"local competition fixture\n"
            (directory / "fixture.csv").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (directory / "SHA256SUMS.txt").write_text(
                f"{digest}  fixture.csv\n", encoding="utf-8"
            )
            result = verify_sha256_manifest(directory)
            self.assertEqual(result["verified_files"], 1)

    def test_manifest_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            (directory / "fixture.csv").write_bytes(b"changed")
            (directory / "SHA256SUMS.txt").write_text(
                f"{'0' * 64}  fixture.csv\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "mismatches"):
                verify_sha256_manifest(directory)

    def test_strict_checks_cover_inf_and_duplicate_aia_keys(self) -> None:
        mia = pd.DataFrame({"record_id": [0, 1, 2]})
        for offset, target_id in enumerate(EXPECTED_TARGET_IDS):
            mia[target_id] = np.asarray([0.1, 0.5, 0.9]) + offset * 1e-6
        aia = pd.DataFrame(
            {
                "target_k": [1, 1],
                "challenge_row_id": [0, 1],
                "time_hat": [10.0, 11.0],
            }
        )
        result = strict_extra_checks(mia, aia)
        self.assertTrue(result["all_numeric_values_finite"])

        bad_mia = mia.copy()
        bad_mia.loc[0, EXPECTED_TARGET_IDS[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "無限"):
            strict_extra_checks(bad_mia, aia)

        bad_aia = aia.copy()
        bad_aia.loc[1, "challenge_row_id"] = 0
        with self.assertRaisesRegex(ValueError, "重複"):
            strict_extra_checks(mia, bad_aia)


if __name__ == "__main__":
    unittest.main()
