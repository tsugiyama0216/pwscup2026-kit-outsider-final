"""Persist only an already released DP model for synthesis and projection.

The NPZ contains numeric factors and JSON, never pickle, source records,
OpenDP query/context objects, or measurement history. A loaded table supports
``synthesize`` and ``project``; it must not be used for ``std`` or continued DP
estimation because its LinearMeasurement history is intentionally absent.
Caller metadata must contain only public configuration and provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
from mbi import CliqueVector, Domain, Factor, MarkovRandomField
from opendp.extras.mbi import ContingencyTable


SCHEMA_VERSION = 1
_JSON_KEY = "__metadata__"
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.String,
    )
}


def _series_payload(series: pl.Series) -> dict[str, Any]:
    dtype = str(series.dtype)
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported public key/cut dtype: {dtype}")
    return {"dtype": dtype, "values": series.to_list()}


def _series_from_payload(name: str, payload: dict[str, Any]) -> pl.Series:
    if set(payload) != {"dtype", "values"}:
        raise ValueError("Invalid key/cut series schema")
    if payload["dtype"] not in _DTYPES or not isinstance(payload["values"], list):
        raise ValueError("Invalid key/cut dtype or values")
    return pl.Series(name, payload["values"], dtype=_DTYPES[payload["dtype"]], strict=True)


def _validate_array(values: np.ndarray, shape: tuple[int, ...], kind: str) -> None:
    if values.shape != shape or values.dtype.kind not in "fiu":
        raise ValueError(f"Invalid {kind} factor shape or dtype")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"Invalid {kind} factor values")
    # Negative infinity is a valid log potential for impossible states.
    if kind == "marginals" and (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Marginal factors must be finite and nonnegative")


def _encode_vector(vector: CliqueVector, kind: str, arrays: dict[str, np.ndarray]):
    factors = []
    for index, clique in enumerate(vector.cliques):
        factor = vector[clique]
        if tuple(factor.domain.attrs) != tuple(clique):
            raise ValueError("Factor axis order does not match its clique")
        values = np.asarray(factor.values)
        _validate_array(values, vector.domain.project(clique).shape, kind)
        key = f"{kind}_{index}"
        arrays[key] = values
        factors.append({"clique": list(clique), "array": key, "dtype": values.dtype.str})
    return factors


def save_table(table: ContingencyTable, path: str | Path, metadata: dict[str, Any]) -> None:
    """Save a synthesis-only DP snapshot; refuse to replace an existing file."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON object containing public provenance")
    model = table.model
    if not isinstance(model, MarkovRandomField):
        raise TypeError("table.model must be MarkovRandomField")
    if model.potentials.domain != model.marginals.domain:
        raise ValueError("Potential and marginal domains differ")
    total = float(model.total)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Model total must be finite and positive")
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "synthesis_and_projection_only",
        "domain": {
            "attrs": list(model.domain.attrs),
            "shape": list(model.domain.shape),
            "labels": model.domain.labels,
        },
        "total": total,
        "potentials": _encode_vector(model.potentials, "potentials", arrays),
        "marginals": _encode_vector(model.marginals, "marginals", arrays),
        "keys": {name: _series_payload(series) for name, series in table.keys.items()},
        "cuts": {name: _series_payload(series) for name, series in table.cuts.items()},
        "thresholds": dict(table.thresholds),
        "metadata": metadata,
    }
    arrays[_JSON_KEY] = np.asarray(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    # Exclusive creation also protects against a concurrent overwrite.
    with target.open("xb") as stream:
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            stream.close()
            target.unlink(missing_ok=True)
            raise


def load_table(path: str | Path) -> tuple[ContingencyTable, dict[str, Any]]:
    """Read validated JSON/numeric arrays without allowing pickled objects."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if _JSON_KEY not in archive.files:
            raise ValueError("Snapshot metadata is missing")
        encoded = archive[_JSON_KEY]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Snapshot metadata must be a scalar JSON string")
        payload = json.loads(str(encoded.item()))
        expected_fields = {
            "schema_version", "purpose", "domain", "total", "potentials",
            "marginals", "keys", "cuts", "thresholds", "metadata",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("Invalid snapshot schema")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported snapshot schema version")
        if payload["purpose"] != "synthesis_and_projection_only":
            raise ValueError("Unsupported snapshot purpose")
        spec = payload["domain"]
        if not isinstance(spec, dict) or set(spec) != {"attrs", "shape", "labels"}:
            raise ValueError("Invalid domain schema")
        if not isinstance(spec["attrs"], list) or not all(isinstance(a, str) for a in spec["attrs"]):
            raise ValueError("Domain attributes must be strings")
        if not isinstance(spec["shape"], list) or not all(type(n) is int and n > 0 for n in spec["shape"]):
            raise ValueError("Domain dimensions must be positive integers")
        domain = Domain(spec["attrs"], spec["shape"], labels=spec["labels"])
        total = payload["total"]
        if type(total) not in (int, float) or not np.isfinite(total) or total <= 0:
            raise ValueError("Model total must be finite and positive")
        expected_arrays = {_JSON_KEY}

        def decode_vector(kind: str) -> CliqueVector:
            factors = payload[kind]
            if not isinstance(factors, list) or not factors:
                raise ValueError(f"Empty or invalid {kind}")
            cliques, values = [], {}
            for index, record in enumerate(factors):
                if not isinstance(record, dict) or set(record) != {"clique", "array", "dtype"}:
                    raise ValueError("Invalid factor schema")
                clique = record["clique"]
                if not isinstance(clique, list) or not all(isinstance(c, str) for c in clique):
                    raise ValueError("Clique must be a list of strings")
                clique = tuple(clique)
                if len(clique) != len(set(clique)) or clique in values:
                    raise ValueError("Duplicate clique or clique attributes")
                key = f"{kind}_{index}"
                if record["array"] != key or key not in archive.files:
                    raise ValueError("Factor array is missing or mislabeled")
                expected_arrays.add(key)
                factor_domain = domain.project(clique)
                array = archive[key]
                _validate_array(array, factor_domain.shape, kind)
                if array.dtype.str != record["dtype"]:
                    raise ValueError("Factor dtype differs from its metadata")
                cliques.append(clique)
                values[clique] = Factor(factor_domain, jnp.asarray(array))
            return CliqueVector(domain, cliques, values)

        potentials, marginals = decode_vector("potentials"), decode_vector("marginals")
        if set(archive.files) != expected_arrays or len(archive.files) != len(expected_arrays):
            raise ValueError("Snapshot contains unexpected or duplicate arrays")
        if not isinstance(payload["metadata"], dict):
            raise ValueError("Caller metadata must be an object")
        keys = {name: _series_from_payload(name, value) for name, value in payload["keys"].items()}
        cuts = {name: _series_from_payload(name, value) for name, value in payload["cuts"].items()}
        thresholds = payload["thresholds"]
        if not isinstance(thresholds, dict) or any(
            name not in keys or type(n) is not int or n < 0
            for name, n in thresholds.items()
        ):
            raise ValueError("Invalid public key thresholds")
        model = MarkovRandomField(potentials=potentials, marginals=marginals, total=total)
        table = ContingencyTable(keys=keys, cuts=cuts, marginals={}, model=model, thresholds=thresholds)
        return table, payload["metadata"]
