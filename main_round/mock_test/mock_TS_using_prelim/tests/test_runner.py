"""Input separation and cross-case safeguards, using real exported inputs."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mock_TS_using_prelim.common import HERE, OUTPUT_ROOT, new_output, read_json, sha256, verify_bundle, write_json
from mock_TS_using_prelim.runner import export_public, public_tables, run_attack, verify_public
from mock_TS_using_prelim.cli import save_score


class PublicInputsTests(unittest.TestCase):
    def test_authorized_inputs_exclude_target_truths_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "public"
            export_public(bundle)
            tables = public_tables(bundle)
            self.assertEqual(tables["attacker_id"], 24)
            self.assertEqual(len(tables["candidates"]), 1049)
            self.assertEqual(len(tables["targets"]), 20)
            self.assertNotIn(24, tables["targets"])
            self.assertEqual(len(tables["self_release"]), 1049)
            self.assertIn("is_rare", tables["self_labels"].columns)
            for target in tables["targets"].values():
                self.assertEqual(len(target["challenge"]), 326)
                self.assertFalse({"time", "onset", "label", "is_rare", "pool_id"} & set(target["challenge"].columns))
            for path in bundle.rglob("*"):
                self.assertFalse(path.name.startswith(("pool_ids", "aia_truth", "F_mia", "F_aia", "token")))

    def test_rewritten_bundle_manifest_cannot_substitute_other_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "public"
            export_public(bundle)
            candidate = bundle / "team24_processing/B.csv"
            candidate.write_bytes(candidate.read_bytes() + b"\n")
            manifest = read_json(bundle / "bundle_manifest.json")
            manifest["files"]["team24_processing/B.csv"] = sha256(candidate)
            write_json(bundle / "bundle_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "actual preliminary distribution"):
                verify_public(bundle)

    def test_unlisted_truth_file_and_wrong_target_list_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "public"
            export_public(bundle)
            (bundle / "aia_truth_1.csv").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected files"):
                verify_public(bundle)
            (bundle / "aia_truth_1.csv").unlink()
            manifest = read_json(bundle / "bundle_manifest.json")
            manifest["target_ids"][0] = 999
            write_json(bundle / "bundle_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Target IDs"):
                verify_public(bundle)

    def test_all_copied_data_and_official_sources_are_verified(self):
        verification = verify_bundle()
        self.assertEqual(verification["data_manifest.json"]["files"], 178)
        self.assertEqual(verification["official_source_manifest.json"]["files"], 36)

    def test_output_boundaries_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Output must"):
                new_output(Path(tmp) / "outside")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT, prefix="test-preserve-") as tmp:
            marker = Path(tmp) / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                new_output(tmp)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_both_plugin_signatures_run_and_modified_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "public"
            export_public(bundle)
            for style in ("named", "kwargs"):
                with self.subTest(signature=style):
                    signature = "*, candidates, self_labels, background, self_challenge, targets, schema, cfg" if style == "named" else "**inputs"
                    unpack = "" if style == "named" else "    candidates, targets = inputs['candidates'], inputs['targets']\n"
                    source = ("import pandas as pd\n"
                              + f"def predict_world({signature}):\n" + unpack
                              + "    mia = pd.DataFrame({'record_id': candidates.record_id})\n"
                              + "    aia = []\n"
                              + "    for k,t in targets.items():\n"
                              + "        mia[str(k)] = 0.5\n"
                              + "        aia.append(pd.DataFrame({'target_k': k, 'challenge_row_id': t['challenge'].challenge_row_id, 'time_hat': 10.0}))\n"
                              + "    return {'F_mia': mia, 'F_aia': pd.concat(aia, ignore_index=True)}\n")
                    plugin = root / (style + ".py")
                    plugin.write_text(source, encoding="utf-8")
                    predictions = root / (style + "-predictions")
                    run_attack(bundle, plugin, predictions)
                    result = save_score(predictions, root / (style + "-score"))
                    self.assertAlmostEqual(result["summary"]["mia_power"], .01, places=12)
                    self.assertEqual(result["input_attribution"], "verified_runner_hashes")
                    path = predictions / "prediction_provenance.json"
                    provenance = read_json(path)
                    provenance["attacker_id"] = 10
                    write_json(path, provenance)
                    with self.assertRaisesRegex(ValueError, "round/team mismatch"):
                        save_score(predictions, root / "wrong-team")
                    provenance["attacker_id"] = 24
                    provenance["public_input_sha256"]["team24_processing/B.csv"] = "bad"
                    write_json(path, provenance)
                    with self.assertRaisesRegex(ValueError, "different public inputs"):
                        save_score(predictions, root / "wrong-case")


if __name__ == "__main__":
    unittest.main()
