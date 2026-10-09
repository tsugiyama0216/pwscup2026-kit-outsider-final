"""Parity detects numeric changes smaller than display rounding."""
import unittest
from uuid import uuid4

from rank_validation.adapters import MOCK
from rank_validation.verify_adapter_parity import compare_csv


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.scratch = MOCK / "local-output" / ("parity-test-" + uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.a, self.b = self.scratch / "a.csv", self.scratch / "b.csv"

    def tearDown(self):
        self.assertTrue(self.scratch.resolve().is_relative_to((MOCK / "local-output").resolve()))
        for path in self.scratch.iterdir():
            path.unlink()
        self.scratch.rmdir()

    def test_crlf_only_differs_in_bytes(self):
        self.a.write_bytes(b"record_id,0\n0,0.10000000000000001\n")
        self.b.write_bytes(b"record_id,0\r\n0,0.10000000000000001\r\n")
        answer = compare_csv(self.a, self.b)
        self.assertFalse(answer["byte_identical"])
        self.assertTrue(answer["newline_normalized_byte_identical"])
        self.assertTrue(answer["numeric_identical"])

    def test_one_ulp_difference_is_not_hidden_by_csv_reader(self):
        self.a.write_bytes(b"record_id,0\n0,0.10000000000000001\n")
        self.b.write_bytes(b"record_id,0\n0,0.10000000000000002\n")
        answer = compare_csv(self.a, self.b)
        self.assertFalse(answer["numeric_identical"])
        self.assertGreater(answer["max_absolute_error"], 0)

    def test_changed_target_header_cannot_pass(self):
        self.a.write_bytes(b"record_id,0\n0,0.5\n")
        self.b.write_bytes(b"record_id,1\n0,0.5\n")
        self.assertFalse(compare_csv(self.a, self.b)["numeric_identical"])

    def test_nan_cannot_pass_even_when_both_files_match(self):
        for path in (self.a, self.b):
            path.write_bytes(b"record_id,0\n0,nan\n")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compare_csv(self.a, self.b)


if __name__ == "__main__":
    unittest.main()