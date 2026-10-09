import copy
import unittest

from rank_validation.panel import validate_case_binding


class PanelBindingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"metadata": {"seed": 61, "protocol": "v5"},
                         "files_sha256": {"public/B.csv": "a", "private/roles.json": "b"}}
        self.state = {"case_metadata": copy.deepcopy(self.manifest["metadata"]),
                      "public_sha256": {"B.csv": "a", "cfg.json": "c"},
                      "truth_sha256": {"roles.json": "b", "aia_floor.csv": "d", "aia_floor_metadata.json": "e"}}

    def test_exact_copies_with_only_prepared_additions_pass(self):
        validate_case_binding(self.state, self.manifest)

    def test_relabelling_old_case_as_fresh_is_rejected(self):
        self.state["case_metadata"]["seed"] = 62
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_case_binding(self.state, self.manifest)

    def test_self_consistent_but_different_public_hash_is_rejected(self):
        self.state["public_sha256"]["B.csv"] = "other"
        with self.assertRaisesRegex(ValueError, "Public"):
            validate_case_binding(self.state, self.manifest)

    def test_self_consistent_but_different_truth_hash_is_rejected(self):
        self.state["truth_sha256"]["roles.json"] = "other"
        with self.assertRaisesRegex(ValueError, "truth"):
            validate_case_binding(self.state, self.manifest)

    def test_unexpected_private_addition_is_rejected(self):
        self.state["truth_sha256"]["another.json"] = "x"
        with self.assertRaisesRegex(ValueError, "truth"):
            validate_case_binding(self.state, self.manifest)


if __name__ == "__main__":
    unittest.main()
