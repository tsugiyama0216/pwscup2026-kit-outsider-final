"""Saved worlds preserve public input equivalence and reject damaged inputs."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from sugiyama_v2.generate import FEATURES, QI, build_world
from sugiyama_v2 import io


class WorldStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = io.workspace.MOCK_OUTPUT / ("sugiyama-v2-io-tests-" + uuid.uuid4().hex)
        cls.scratch.mkdir(parents=True)
        cls.root = cls.scratch / "world"
        cls.world = build_world(2026100861, profile="mixed")
        cls.manifest = io.export_world(cls.world, cls.root)

    @classmethod
    def tearDownClass(cls):
        # This is an ordinary mkdir, rather than TemporaryDirectory's Windows
        # ACL behavior. Delete only the exact newly allocated ignored test tree.
        target = cls.scratch.resolve()
        allowed = io.workspace.MOCK_OUTPUT.resolve()
        if target.is_relative_to(allowed) and target.parent == allowed and target.name.startswith("sugiyama-v2-io-tests-"):
            shutil.rmtree(target)
        else:
            raise AssertionError("Refuse cleanup outside this test's output tree")

    @contextmanager
    def changed(self, *relative_paths):
        paths = [self.root / name for name in relative_paths]
        saved = [path.read_bytes() for path in paths]
        try:
            yield
        finally:
            for path, data in zip(paths, saved):
                path.write_bytes(data)

    def read_manifest(self):
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def refresh_hash(self, relative):
        manifest = self.read_manifest()
        manifest["files_sha256"][relative] = io.sha(self.root / relative)
        io.save_json(self.root / "manifest.json", manifest)

    def test_real_world_roundtrip_and_public_self_proof(self):
        public, private_path, manifest, stamp = io.load_world(self.root)
        self.assertEqual(private_path, self.root / "private")
        self.assertEqual(stamp, io.snapshot(self.root)[1])
        self.assertEqual(manifest, self.manifest)
        for key in ("candidates", "self_labels", "background", "self_challenge"):
            pd.testing.assert_frame_equal(public[key], self.world["public"][key])
        for team, target in public["targets"].items():
            for key in ("release", "challenge"):
                # CSV infers integral rounded features as int; their numerical
                # values, columns and ordering must still be exactly preserved.
                pd.testing.assert_frame_equal(target[key], self.world["public"]["targets"][team][key],
                                              check_dtype=False, check_exact=True)
        self_keys = set(map(tuple, public["self_challenge"][QI].to_numpy()))
        proved = [tuple(row) in self_keys for row in public["candidates"][QI].to_numpy()]
        self.assertEqual(sum(proved), 163)
        metadata = json.loads((private_path / "case_metadata.json").read_text(encoding="utf-8"))
        self.assertIs(metadata["synthetic"], True)
        self.assertEqual(metadata["format"], io.FORMAT)
        self.assertEqual(metadata["target_styles"], manifest["metadata"]["target_styles"])

    def test_prediction_api_is_public_only_and_deeply_copied(self):
        public, _, _, _ = io.load_world(self.root)
        public["truth"] = "PRIVATE_SENTINEL"
        public["seed"] = 123
        cfg = {"aia": {"tau": .5}}
        args = io.prediction_arguments(public, cfg)
        self.assertEqual(set(args), {"candidates", "self_labels", "background",
                                   "self_challenge", "targets", "schema", "cfg"})
        first = next(iter(args["targets"]))
        old = public["targets"][first]["release"].iloc[0].BMI
        args["targets"][first]["release"].loc[0, "BMI"] = old + 7
        args["self_labels"].loc[0, "is_rare"] = 100
        args["schema"]["age_range"][0] = -100
        args["cfg"]["aia"]["tau"] = 7
        self.assertEqual(public["targets"][first]["release"].iloc[0].BMI, old)
        self.assertIn(public["self_labels"].loc[0, "is_rare"], [0, 1])
        self.assertEqual(public["schema"]["age_range"][0], 40)
        self.assertEqual(cfg["aia"]["tau"], .5)
        for target in args["targets"].values():
            self.assertEqual(set(target), {"release", "challenge"})

    def test_case_byte_tampering_is_detected(self):
        relative = "public/B.csv"
        with self.changed(relative):
            path = self.root / relative
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "World file changed"):
                io.load_world(self.root)
        io.snapshot(self.root)

    def test_required_file_and_hash_format_are_checked(self):
        for mutate in (lambda m: m["files_sha256"].pop("public/B.csv"),
                       lambda m: m["files_sha256"].update({"public/B.csv": "not-a-hash"})):
            with self.changed("manifest.json"):
                manifest = self.read_manifest()
                mutate(manifest)
                io.save_json(self.root / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    io.snapshot(self.root)

    def test_manifest_path_escape_and_aliases_are_rejected(self):
        for path in ("../outside", "/absolute", "C:/absolute", "public\\B.csv",
                     "./public/B.csv", "public//B.csv", "public/../B.csv"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    io.contained(self.root, path)
        with self.changed("manifest.json"):
            manifest = self.read_manifest()
            manifest["files_sha256"]["../outside"] = "0" * 64
            io.save_json(self.root / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                io.snapshot(self.root)

    def test_manifest_and_private_metadata_must_agree(self):
        with self.changed("manifest.json"):
            manifest = self.read_manifest()
            manifest["metadata"]["seed"] += 1
            io.save_json(self.root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                io.snapshot(self.root)
        relative = "private/case_metadata.json"
        with self.changed(relative, "manifest.json"):
            metadata = json.loads((self.root / relative).read_text(encoding="utf-8"))
            metadata["synthetic"] = False
            io.save_json(self.root / relative, metadata)
            self.refresh_hash(relative)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                io.snapshot(self.root)

    def test_current_source_changes_do_not_invalidate_frozen_case_inputs(self):
        with self.changed("manifest.json"):
            manifest = self.read_manifest()
            manifest["sources_sha256"] = {"historical/source.py": "1" * 64}
            manifest["original_input_sha256"]["B_self.csv"] = "2" * 64
            io.save_json(self.root / "manifest.json", manifest)
            public, _, loaded, _ = io.load_world(self.root)
            self.assertEqual(len(public["candidates"]), 1049)
            self.assertEqual(loaded["sources_sha256"], manifest["sources_sha256"])
            self.assertIn("provenance", loaded["source_hash_policy"])

    def test_output_must_be_new_empty_child_of_ignored_root(self):
        with self.assertRaises(ValueError):
            io.export_world(self.world, self.root)
        for path in (io.workspace.MOCK_OUTPUT, io.workspace.MOCK_ROOT / "forbidden-world"):
            with self.assertRaises(ValueError):
                io.output_path(path)
        empty = self.scratch / "empty"
        empty.mkdir()
        self.assertEqual(io.output_path(empty), empty.resolve())
        file = self.scratch / "ordinary-file"
        file.write_text("preserve", encoding="utf-8")
        with self.assertRaises(ValueError):
            io.output_path(file)

    def test_rehashed_invalid_ids_labels_and_self_proof_are_rejected(self):
        changes = (
            ("public/B.csv", lambda f: f.assign(record_id=-f.record_id - 1)),
            ("public/B_self.csv", lambda f: f.assign(is_rare=0)),
            ("public/aia_challenge_10.csv", lambda f: f.assign(SBP=f.SBP + .123456)),
            ("public/A_bg.csv", lambda f: f.assign(prefecture=np.nan)),
        )
        for relative, mutate in changes:
            with self.subTest(relative=relative), self.changed(relative, "manifest.json"):
                frame = pd.read_csv(self.root / relative, float_precision="round_trip")
                io.csv(self.root / relative, mutate(frame))
                self.refresh_hash(relative)
                with self.assertRaises(ValueError):
                    io.load_world(self.root)

    def test_explicit_nonsynthetic_world_is_not_exported(self):
        world = {**self.world, "metadata": {**self.world["metadata"], "synthetic": False}}
        path = self.scratch / "not-synthetic"
        with self.assertRaises(ValueError):
            io.export_world(world, path)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
