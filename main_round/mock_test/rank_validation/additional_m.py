"""Production-equivalent public-table adapters for the frozen M v1/v2.

prepare/gate perform filesystem verification outside the caller's read guard.
The returned predictor reads only supplied public tables and in-memory choices.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

from . import adapters

ROOT, ATTACK = adapters.ROOT, adapters.ATTACK
CANONICAL = ATTACK / "Attack_M"
COMPATIBILITY = ATTACK / "M"
MODELS = ("Mv1", "Mv2")
PUBLIC_ARGUMENTS = {"candidates", "self_labels", "background", "self_challenge", "targets", "schema", "cfg"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _modules(model_id):
    adapters._bootstrap()
    names = ["M", "M.v1", "M.v1.model", "M.v1.actual", "M.v2", "M.v2.artifacts"]
    if model_id == "Mv2":
        names += ["M.v2.mia", "M.v2.aia", "M.v2.model"]
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        relative = Path(*name.split(".")[1:])
        expected = COMPATIBILITY / relative
        expected = expected / "__init__.py" if name in {"M", "M.v1", "M.v2"} else expected.with_suffix(".py")
        if Path(module.__file__).resolve() != expected.resolve():
            raise ImportError("Wrong frozen M compatibility module origin: " + name)
        modules[name] = module
    return modules


def _source_hashes(modules):
    paths = {Path(__file__).resolve(), Path(adapters.__file__).resolve()}
    for module in modules.values():
        path = Path(module.__file__).resolve()
        relative = path.relative_to(COMPATIBILITY)
        canonical = CANONICAL / relative
        if adapters.sha(path) != adapters.sha(canonical):
            raise ValueError("Canonical and compatibility M source differ: " + relative.as_posix())
        paths.update((path, canonical))
    return {path.relative_to(ROOT).as_posix(): adapters.sha(path) for path in sorted(paths)}


def _verified_snapshot_files(snapshot):
    result = {}
    for relative, expected in snapshot.items():
        path = (ATTACK / relative).resolve()
        if not path.is_relative_to(ATTACK) or adapters.sha(path) != expected:
            raise ValueError("Frozen M snapshot dependency differs: " + relative)
        result[path.relative_to(ROOT).as_posix()] = expected
    return result


def _gate(model_id, modules):
    artifacts = modules["M.v2.artifacts"]
    preservation = artifacts.check_v1_preserved()
    preservation_path = CANONICAL / "local-output/v2/v1_preservation.json"
    verified_files = _verified_snapshot_files(_json(preservation_path)["files"])
    v1_selection = _json(CANONICAL / "v1/selection.json")
    if (v1_selection.get("selected_variant") != "full_reference_aia"
            or v1_selection.get("selected_before_final_check") is not True
            or adapters.sha(CANONICAL / "v1/model.py") != v1_selection.get("model_sha256")
            or adapters.sha(COMPATIBILITY / "v1/selection.json") != adapters.sha(CANONICAL / "v1/selection.json")):
        raise ValueError("Frozen M v1 selection or source changed")
    result = {"model_id": model_id, "M_v1_preservation": preservation,
              "v1_selection_sha256": adapters.sha(CANONICAL / "v1/selection.json"),
              "v1_preservation_sha256": adapters.sha(CANONICAL / "local-output/v2/v1_preservation.json"),
              "fixed_sources": _source_hashes(modules)}
    if model_id == "Mv1":
        result["selection"] = {"variant": "full_reference_aia"}
    else:
        selected = artifacts.load_selection()
        if (selected.get("mia_strength") != 1.0 or selected.get("aia_variant") != "guarded40"
                or selected.get("mia_adjustment") != "subtract_candidate_cross_target_median"):
            raise ValueError("Unexpected frozen M v2 selection")
        if adapters.sha(COMPATIBILITY / "v2/selection.json") != adapters.sha(CANONICAL / "v2/selection.json"):
            raise ValueError("Canonical and compatibility M v2 selection differ")
        final = _json(CANONICAL / "local-output/v2/final_check.json")
        selection_hash = adapters.sha(CANONICAL / "v2/selection.json")
        if final.get("completed") is not True or final.get("selection_sha256") != selection_hash:
            raise ValueError("Selected M v2 did not complete its frozen final check")
        result.update(selection={"mia_strength": 1.0, "aia_variant": "guarded40"},
                      v2_selection_sha256=selection_hash,
                      v2_dependency_files=len(selected["dependency_sha256"]),
                      v2_dependency_sha256=copy.deepcopy(selected["dependency_sha256"]),
                      final_check_sha256=adapters.sha(CANONICAL / "local-output/v2/final_check.json"))
    # Keep canonical numeric files, the unchanged compatibility imports and the
    # complete recorded dependency closure in the worker's fixed identity.
    verified_files.update(result["fixed_sources"])
    records = [CANONICAL / "v1/selection.json", COMPATIBILITY / "v1/selection.json", preservation_path]
    if model_id == "Mv2":
        verified_files.update(_verified_snapshot_files(selected["dependency_sha256"]))
        records += [CANONICAL / "v2/selection.json", COMPATIBILITY / "v2/selection.json",
                    CANONICAL / "local-output/v2/final_check.json"]
    for path in records:
        verified_files[path.resolve().relative_to(ROOT).as_posix()] = adapters.sha(path)
    result["files_sha256"] = dict(sorted(verified_files.items()))
    return result


def _predict(model_id, tables, selection, modules):
    import numpy as np
    import pandas as pd
    from sugiyama_v2.scoring import validate_predictions
    if not isinstance(tables, dict) or set(tables) != PUBLIC_ARGUMENTS:
        raise ValueError("M prediction accepts exactly the seven public-world arguments")
    adapters.validate_public(**tables)
    public = copy.deepcopy(tables)
    b, labels, bg, self_q, targets, schema, cfg = (public[k] for k in
        ("candidates", "self_labels", "background", "self_challenge", "targets", "schema", "cfg"))
    actual = modules["M.v1.actual"]
    mask, proof = actual.derive_private_mask(b.astype(str), labels.astype(str), self_q.astype(str),
                                           schema["aia_qi_columns"], schema["submitted_c_columns"])
    mia = pd.DataFrame({"record_id": b.record_id.to_numpy()})
    aia_parts = []
    if model_id == "Mv1":
        base = modules["M.v1.model"]
        for k in adapters.TARGETS:
            item = targets[k]
            result = base.predict(c=item["release"], background=bg, candidates=b,
                                  challenge=item["challenge"], cfg=cfg, variant=selection["variant"])
            # Production includes the unknown-score affine transform, not only zeroing.
            mia[str(k)] = actual.apply_private_mask(result["mia"], mask)
            aia_parts.append(pd.DataFrame({"target_k": k,
                "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(), "time_hat": result["aia"]}))
            print(f"Mv1: public target complete ({k})", flush=True)
    else:
        result = modules["M.v2.model"].predict_all(
            targets={k: targets[k]["release"] for k in adapters.TARGETS}, background=bg, candidates=b,
            challenges={k: targets[k]["challenge"] for k in adapters.TARGETS}, cfg=cfg,
            private_mask=mask, mia_strength=selection["mia_strength"], aia_variant=selection["aia_variant"])
        if result.get("target_order") != list(adapters.TARGETS):
            raise ValueError("M v2 returned a different joint target order")
        for j, k in enumerate(result["target_order"]):
            # The frozen joint model already applied its private mask exactly once.
            mia[str(k)] = result["mia"][:, j]
            aia_parts.append(pd.DataFrame({"target_k": k,
                "challenge_row_id": targets[k]["challenge"].challenge_row_id.to_numpy(), "time_hat": result["aia"][k]}))
        print("Mv2: one joint 22-target prediction complete", flush=True)
    aia = pd.concat(aia_parts, ignore_index=True)
    validation = validate_predictions(mia, aia, {"team_id": 10, **public})
    if np.any(mia.drop(columns="record_id").to_numpy()[mask] != 0.):
        raise ValueError("Frozen M known-private rows must remain zero")
    return {"F_mia": mia, "F_aia": aia,
            "diagnostics": {"model_id": model_id, "known_negative_proof": proof,
                            "selection": copy.deepcopy(selection), "validation": validation,
                            "algorithm_changed": False, "public_only_arguments": sorted(PUBLIC_ARGUMENTS),
                            "joint_target_count": len(adapters.TARGETS),
                            "stored_prediction_values_used": False}}


def prepare(model_id):
    """Return (predict(tables), gate(), fixed_sources) for an outside-guard worker.

    The worker must call gate before/after inference, compare its results, and
    run predict inside its read audit and single-thread runtime context.
    """
    if model_id not in MODELS:
        raise ValueError("Unknown additional M model: " + str(model_id))
    modules = _modules(model_id)
    initial = _gate(model_id, modules)
    selection = copy.deepcopy(initial["selection"])
    fixed_sources = copy.deepcopy(initial["files_sha256"])
    def gate():
        return _gate(model_id, modules)
    def predict(tables):
        return _predict(model_id, tables, selection, modules)
    return predict, gate, fixed_sources
