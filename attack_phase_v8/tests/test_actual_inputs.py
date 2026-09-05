import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attack_phase_v8.actual_inputs import EXPECTED_ACTUAL_HASHES, verify_actual_hashes


class ActualInputTests(unittest.TestCase):
    def test_only_nonsensitive_files_are_allowlisted(self):
        self.assertEqual(set(EXPECTED_ACTUAL_HASHES),{"B.csv","B_self.csv","A_bg.csv","schema.json"})

    def test_missing_data_fails_without_practice_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"no practice-data fallback"):
                verify_actual_hashes(Path(directory))

    def test_wrong_data_fails_before_any_other_file_read(self):
        with patch.object(Path,"is_file",return_value=True):
            with patch.object(Path,"read_bytes",return_value=b"practice data") as read:
                with self.assertRaisesRegex(ValueError,"hash mismatch: B.csv"):
                    verify_actual_hashes(Path("dummy"))
                self.assertEqual(read.call_count,1)

if __name__=="__main__":
    unittest.main()
