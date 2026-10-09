"""Frozen Rv3 selected components over one supplied public 22-target world.

The worker calls gate() outside its inference read guard. predict(tables) reads
only the three fixed Python files and never imports run/confirm/world or JSON.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4

from .adapters import ROOT, ATTACK, TARGETS, validate_public

BASE = ROOT / "main_round/anonymization/team10/atack/Rv3"
SELECTION = {"mia": "full5", "aia": "forest_contrast5_1.0"}
NUMERIC_SHA256 = {
    "model.py": "d72d8556fda4a2679f9e5e77538db25afa70c0a68c4887d267b5ba374a4c684d",
    "selected.py": "40ac0c24a2db6d4e06c01aa04746e2779cdf35cb512196ef725c95ceb45d582b",
}
MASK = ATTACK / "Attack_M/v1/actual.py"
MASK_SHA256 = "3a9b3ce631d17f60192c1815bf73d9bdfeaba64b415ad4c7f6b4ff0ef49f5393"
CONTROL_SHA256 = {
    "selection.json": "f26a204044f4226a87d087bee9938000dd68ec4f47000f9b3031c5c9cf4a6b22",
    "production_sources.json": "6d95bdda9e7a8e7d22c74f5c8b0c86add75f1b10e6ffe7a440663850412ac2ad",
    "production_dependencies.json": "4c498e9efcbc4558427a82cc7ea9654cd7ba755efe5448480016583c7c015470",
    "confirmation.json": "7e92dcb2dfb02b593f255dbd473ab5636e0c118f00509f0d0730bdf546617940",
}


def _bytes(path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path.name.casefold() == "token.txt":
        raise ValueError("Invalid Rv3 dependency origin")
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Frozen Rv3 dependency mismatch: " + str(path))
    return data


def _sha(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def _verify_mapping(root, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing Rv3 dependency manifest")
    for relative, digest in mapping.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Escaping Rv3 dependency")
        _bytes(path, digest)
    return len(mapping)


def _gate():
    controls = {name: json.loads(_bytes(BASE / name, digest).decode("utf-8-sig"))
                for name, digest in CONTROL_SHA256.items()}
    selection = controls["selection.json"]
    if {key: selection.get(key) for key in SELECTION} != SELECTION:
        raise ValueError("Unexpected frozen Rv3 selection")
    production = controls["production_dependencies.json"]
    confirmation = controls["confirmation.json"]
    if (production.get("selection_sha256") != CONTROL_SHA256["selection.json"]
            or confirmation.get("completed") is not True
            or confirmation.get("selection_sha256") != CONTROL_SHA256["selection.json"]
            or confirmation.get("production_sources_sha256") != CONTROL_SHA256["production_sources.json"]):
        raise ValueError("Rv3 selection/confirmation binding mismatch")
    return {
        "selection_verified_files": _verify_mapping(BASE, selection["source_sha256"]),
        "production_verified_files": _verify_mapping(BASE, controls["production_sources.json"]),
        "dependency_verified_files": _verify_mapping(ROOT, production["files_sha256"]),
        "control_sha256": dict(CONTROL_SHA256), "selection": dict(SELECTION),
    }


def _load_source(label, path, expected):
    path = Path(path).resolve()
    data = _bytes(path, expected)
    name = "_rank_rv3_" + label + "_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Compile the checked source directly, excluding any stale pyc cache.
        exec(compile(data, str(path), "exec"), module.__dict__)
        if Path(module.__file__).resolve() != path:
            raise ImportError("Rv3 numeric module origin changed")
        return module
    finally:
        sys.modules.pop(name, None)


def _load_components():
    model = _load_source("model", BASE / "model.py", NUMERIC_SHA256["model.py"])
    missing = object()
    previous = sys.modules.get("model", missing)
    sys.modules["model"] = model
    try:
        selected = _load_source("selected", BASE / "selected.py", NUMERIC_SHA256["selected.py"])
        if selected.numeric is not model.numeric or selected.ExtraTreesRegressor is not model.ExtraTreesRegressor:
            raise ImportError("Rv3 selected components came from a different model")
    finally:
        if previous is missing:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous
    mask = _load_source("mask", MASK, MASK_SHA256)
    return selected.predict_selected, mask.derive_private_mask


def _predict(tables):
    import numpy as np
    import pandas as pd
    validate_public(**tables)
    selected, derive_mask = _load_components()
    candidates, labels = tables["candidates"], tables["self_labels"]
    schema = tables["schema"]
    private, proof = derive_mask(candidates.astype(str), labels.astype(str), tables["self_challenge"].astype(str),
                                schema["aia_qi_columns"], schema["submitted_c_columns"])
    mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
    parts = []
    for k in TARGETS:
        item = tables["targets"][k]
        result = selected(dict(SELECTION), c=item["release"], background=tables["background"],
                          candidates=candidates, challenge=item["challenge"], cfg=tables["cfg"])
        m, a = np.asarray(result["mia"], dtype=float).copy(), np.asarray(result["aia"], dtype=float)
        if m.shape != (len(candidates),) or a.shape != (len(item["challenge"]),):
            raise ValueError("Invalid Rv3 output vector shape")
        m[private] = 0.0
        mia[str(k)] = m
        parts.append(pd.DataFrame({"target_k": k, "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(), "time_hat": a}))
        print(f"Rv3: public target complete ({k})", flush=True)
    return {"F_mia": mia, "F_aia": pd.concat(parts, ignore_index=True),
            "diagnostics": {"model_id": "Rv3", "selection": dict(SELECTION), "algorithm_changed": False,
                            "known_negative_proof": proof, "private_mask_matches_production": True,
                            "numeric_api": "atack/Rv3/selected.predict_selected"}}


def prepare(model_id):
    """Return (predict(public_tables), gate(), repo_relative_fixed_sources)."""
    if model_id != "Rv3":
        raise ValueError("additional_r supports only Rv3")
    _gate()
    paths = {BASE / name: digest for name, digest in NUMERIC_SHA256.items()}
    paths[MASK] = MASK_SHA256
    for path, digest in paths.items():
        _bytes(path, digest)
    fixed_sources = {path.relative_to(ROOT).as_posix(): digest for path, digest in paths.items()}
    # Preserve all original source/manifest evidence, including unused search
    # components, without importing those components into numerical inference.
    for control, field, root in (("selection.json", "source_sha256", BASE),
                                 ("production_sources.json", None, BASE),
                                 ("production_dependencies.json", "files_sha256", ROOT)):
        record = json.loads(_bytes(BASE / control, CONTROL_SHA256[control]).decode("utf-8-sig"))
        mapping = record if field is None else record[field]
        for relative, digest in mapping.items():
            path = (root / relative).resolve()
            key = path.relative_to(ROOT).as_posix()
            if key in fixed_sources and fixed_sources[key] != digest:
                raise ValueError("Inconsistent Rv3 dependency manifests")
            fixed_sources[key] = digest
    fixed_sources.update({(BASE / name).relative_to(ROOT).as_posix(): digest
                          for name, digest in CONTROL_SHA256.items()})
    fixed_sources[Path(__file__).resolve().relative_to(ROOT).as_posix()] = _sha(__file__)
    return _predict, _gate, fixed_sources