"""Public toy tests; never read contest records or make a DP release."""
import json
from pathlib import Path
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np
import polars as pl
from mbi import CliqueVector, Domain, Factor, MarkovRandomField, marginal_oracles
from opendp.extras.mbi import ContingencyTable

from model_snapshot import load_table, save_table


def public_table():
    domain = Domain.fromdict({"H": 3, "A": 8, "B": 8, "C": 8})
    cliques = [("H", "A"), ("H", "B"), ("H", "C")]
    potentials = CliqueVector.zeros(domain, cliques)
    marginals = marginal_oracles.message_passing_stable(potentials)
    model = MarkovRandomField(potentials=potentials, marginals=marginals, total=1049.0)
    keys = {name: pl.Series(name, list(range(size - 1)) + [None], dtype=pl.Int32)
            for name, size in domain.config.items()}
    table = ContingencyTable(keys=keys, cuts={}, marginals={}, model=model)
    # Extra objects must never leak into the persistence format.
    table.private_frame = "DO_NOT_SERIALIZE_PRIVATE_RECORDS"
    table.marginals[("H",)] = "DO_NOT_SERIALIZE_MEASUREMENT_HISTORY"
    return table


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.npz"
        self.table = public_table()
        self.metadata = {"public_config_sha256": "0" * 64, "seed": 17}
        save_table(self.table, self.path, self.metadata)

    def rewrite(self, mutate):
        with np.load(self.path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        mutate(arrays)
        np.savez_compressed(self.path, **arrays)

    def test_roundtrip_preserves_sample_and_marginals(self):
        loaded, metadata = load_table(self.path)
        self.assertEqual(metadata, self.metadata)
        for method in ("sample", "round"):
            np.random.seed(17)
            before = self.table.synthesize(1049, method)
            np.random.seed(17)
            after = loaded.synthesize(1049, method)
            self.assertTrue(before.equals(after), method)
        for clique in self.table.model.cliques:
            np.testing.assert_array_equal(self.table.project(clique), loaded.project(clique))
        self.assertEqual(loaded.schema, self.table.schema)
        for name in self.table.keys:
            self.assertEqual(loaded.keys[name].to_list(), self.table.keys[name].to_list())
            self.assertIsNone(loaded.keys[name][-1])

    def test_archive_contains_only_allowlisted_dp_model_and_public_metadata(self):
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertTrue(all(array.dtype.kind != "O" for array in (archive[key] for key in archive.files)))
            text = str(archive["__metadata__"].item())
            self.assertNotIn("DO_NOT_SERIALIZE", text)
            self.assertNotIn("private_frame", text)
            self.assertEqual(set(archive.files), {"__metadata__", *(f"potentials_{i}" for i in range(3)), *(f"marginals_{i}" for i in range(3))})
        loaded, _ = load_table(self.path)
        self.assertEqual(loaded.marginals, {})
        self.assertFalse(hasattr(loaded, "private_frame"))

    def test_nonuniform_model_and_numeric_cuts_are_preserved(self):
        domain = Domain.fromdict({"K": 4})
        potentials = CliqueVector(domain, [("K",)], {
            ("K",): Factor(domain, jnp.asarray([0.0, 0.1, -1.0, 2.0]))
        })
        model = MarkovRandomField(
            potentials=potentials,
            marginals=marginal_oracles.message_passing_stable(potentials),
            total=100.0,
        )
        table = ContingencyTable(
            keys={"K": ["(-inf,1]", "(1,2]", "(2,3]", "(3,inf]"]},
            cuts={"K": pl.Series("K", [1, 2, 3], dtype=pl.Int32)},
            marginals={}, model=model,
        )
        other = self.path.with_name("nonuniform.npz")
        save_table(table, other, {})
        loaded, _ = load_table(other)
        np.testing.assert_array_equal(table.project("K"), loaded.project("K"))
        self.assertTrue(loaded.cuts["K"].equals(table.cuts["K"]))
        np.random.seed(33)
        before = table.synthesize(1049, "sample")
        np.random.seed(33)
        after = loaded.synthesize(1049, "sample")
        self.assertTrue(before.equals(after))

    def test_existing_path_is_not_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_table(self.table, self.path, self.metadata)
        self.assertEqual(before, self.path.read_bytes())

    def test_wrong_shape_is_rejected(self):
        self.rewrite(lambda arrays: arrays.update(potentials_0=np.zeros((1, 1))))
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            load_table(self.path)

    def test_wrong_dtype_is_rejected(self):
        self.rewrite(lambda arrays: arrays.update(potentials_0=arrays["potentials_0"].astype(np.float32)))
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            load_table(self.path)

    def test_object_array_is_rejected(self):
        self.rewrite(lambda arrays: arrays.update(potentials_0=np.array([[object()]], dtype=object)))
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            load_table(self.path)

    def test_unknown_schema_is_rejected(self):
        def mutate(arrays):
            payload = json.loads(str(arrays["__metadata__"].item()))
            payload["schema_version"] = 99
            arrays["__metadata__"] = np.asarray(json.dumps(payload))
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, "schema version"):
            load_table(self.path)

    def test_unexpected_arrays_are_rejected(self):
        self.rewrite(lambda arrays: arrays.update(unexpected=np.zeros(1)))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_table(self.path)


if __name__ == "__main__":
    unittest.main()
