"""Public-table adapters preserving the four frozen production algorithms.

No target truth, release-style label, generation seed or private path is an
inference argument. Run each model in a fresh model_worker process. Historical
fitted assets remain part of their models and retain their training exposure.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ATTACK = ROOT / "main_round/attack"
MOCK = ROOT / "main_round/mock_test"
FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
TARGETS = tuple(k for k in (*range(18), 22, 25, 26, 27, 28) if k != 10)
MODELS = ("Rv1", "Rv2", "pl1", "Tv105")


def sha(path):
    path = Path(path)
    if path.name.lower() == "token.txt":
        raise ValueError("No credentials belong in model verification")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        old = sys.modules[name]
        if Path(old.__file__).resolve() != path:
            raise ImportError("Conflicting model module origin: " + name)
        return old
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _manifest(path):
    files = _json(path)["files_sha256"]
    for relative, expected in files.items():
        file = (ROOT / relative).resolve()
        if not file.is_relative_to(ROOT) or sha(file) != expected:
            raise ValueError("Frozen dependency mismatch: " + relative)
    return files


def _selection(directory, key):
    value = _json(directory / "selection.json")
    for name, expected in value[key].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError("Frozen selected source mismatch: " + name)
    return value


def _bootstrap():
    for path in (ATTACK, ATTACK / "Attak_T", MOCK, ATTACK / "official-kit/starter/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import mock_workspace
    mock_workspace.initialize()
    import attack_research
    if Path(attack_research.__file__).resolve() != ATTACK / "Attak_T/attack_research/__init__.py":
        raise ImportError("Wrong frozen T implementation")


def _prepare(model_id):
    """Verify original hash gates before/after inference, outside data guards."""
    _bootstrap()
    rv2 = ATTACK / "Attack_R/v2"
    if model_id in ("Rv1", "Rv2"):
        original = _load("_rank_frozen_rv1_adapter", ATTACK / "Attack_R/sugiyama_adapter.py")
        common, numeric1, verify_t, attack_class, derive_mask, selection_m, preserve_m = original._load_frozen()
        numeric2 = _load("_rank_frozen_rv2_numeric", rv2 / "model.py") if model_id == "Rv2" else None
        def gate():
            value = {"R_v1": common.verify_dependencies(), "T_v88": verify_t(), "M_v1": preserve_m()}
            if selection_m()["mia_strength"] != 1.0:
                raise ValueError("Unexpected frozen M strength")
            if model_id == "Rv2":
                runtime = _load("_rank_frozen_rv2_runtime", rv2 / "runtime.py")
                selected = runtime.selected_check()
                if (selected["mia"], selected["aia"]) != ("T", "T"):
                    raise ValueError("Adapter requires frozen R v2 T/T selection")
                value["R_v2_upstream"] = runtime.upstream_check()
                value["R_v2_dependencies"] = len(_manifest(rv2 / "production_dependencies.json"))
                value["selection_sha256"] = sha(rv2 / "selection.json")
            return value
        paths = _json(ATTACK / "Attack_R/v1/dependencies.json")["files_sha256"]
        return {"gate": gate, "assets": paths, "numeric1": numeric1, "numeric2": numeric2,
                "attack_class": attack_class, "derive_mask": derive_mask}
    if model_id == "pl1":
        base = ATTACK / "attackT/pl1"
        paths = _json(base / "source_manifest.json")["files_sha256"]
        def gate():
            return {"manifest_sha256": sha(base / "source_manifest.json"), "verified_files": len(_manifest(base / "source_manifest.json"))}
        return {"gate": gate, "assets": paths, "model": _load("_rank_frozen_pl1", base / "model.py")}
    if model_id == "Tv105":
        base = ATTACK / "attackT/Tv105"
        runtime = _load("_rank_frozen_tv105_runtime", base / "runtime.py")
        # The frozen numeric file imports generic 'runtime'; bind only while
        # loading it and restore the prior module immediately afterwards.
        previous = sys.modules.get("runtime")
        sys.modules["runtime"] = runtime
        try:
            model = _load("_rank_frozen_tv105_numeric", base / "model.py")
        finally:
            if previous is None:
                sys.modules.pop("runtime", None)
            else:
                sys.modules["runtime"] = previous
        synthesis = _load("_rank_frozen_tv105_synthesis", base / "synthesis.py")
        from M.v1.actual import derive_private_mask
        def gate():
            _selection(base, "source_hashes")
            dependencies = runtime.dependencies()
            return {"verified_files": len(dependencies), "selection_sha256": sha(base / "selection.json"),
                    "Rv4_numeric_sha256": sha(runtime.RV4 / "model.py"),
                    "Rv4_adapter_sha256": sha(runtime.RV4 / "evaluation/mock_TS_comparison_20261008/reproduction/adapter.py")}
        return {"gate": gate, "assets": _json(rv2 / "production_dependencies.json")["files_sha256"],
                "model": model, "synthesis": synthesis, "selection": _selection(base, "source_hashes"),
                "derive_mask": derive_private_mask}
    raise ValueError("Unknown frozen model: " + str(model_id))


@contextmanager
def _inference_guard(assets, schema=None):
    """Detect repository data-file reads; this is not an OS security sandbox."""
    allowed = {(ROOT / name).resolve() for name in assets if Path(name).suffix in (".joblib", ".json")}
    if schema is not None:
        schema_path = ROOT / "main_round/anonymization/team10/participant_data/schema.json"
        if _json(schema_path) != schema:
            raise ValueError("Frozen T public schema differs from supplied schema")
        allowed.add(schema_path.resolve())
    state = {"active": True, "asset_reads": set(), "blocked_reads": []}
    def audit(event, args):
        if event != "open" or not state["active"] or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        if path in allowed:
            state["asset_reads"].add(path.relative_to(ROOT).as_posix())
            return
        if path.is_relative_to(ATTACK / ".venv") or path.is_relative_to(ATTACK / ".tools"):
            return
        suffix = path.suffix.lower()
        if (suffix in {".csv", ".parquet", ".joblib", ".pkl", ".pickle", ".zip"}
                or path.is_relative_to(MOCK / "local-output")
                or path.is_relative_to(ATTACK / "targets")
                or path.is_relative_to(ROOT / "main_round/anonymization/team10/participant_data")):
            state["blocked_reads"].append(path.relative_to(ROOT).as_posix())
            raise RuntimeError("Inference attempted a non-public data-file read: " + str(path))
    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


def validate_public(*, candidates, self_labels, background, self_challenge, targets, schema, cfg, attacker_id=10):
    import numpy as np
    import pandas as pd
    if type(attacker_id) is not int or attacker_id != 10 or set(targets) != set(TARGETS):
        raise ValueError("These production-equivalent adapters require team10 and all 22 main-round targets; preliminary team24 needs a separately verified adapter")
    def frame(value, columns, rows, label):
        if not isinstance(value, pd.DataFrame) or list(value.columns) != columns or (rows is not None and len(value) != rows) or value.empty:
            raise ValueError("Invalid public frame: " + label)
        numeric = value.drop(columns="prefecture").to_numpy(float)
        if not np.isfinite(numeric).all() or value.prefecture.isna().any():
            raise ValueError("Invalid public values: " + label)
        for key in ("record_id", "challenge_row_id"):
            if key in value:
                ids = value[key].to_numpy(float)
                if len(np.unique(ids)) != len(ids) or (ids != np.floor(ids)).any() or (ids < 0).any() or (ids >= 2**53).any():
                    raise ValueError("Invalid public IDs: " + label)
    frame(candidates, ["record_id", *FEATURES], 1049, "B")
    frame(self_labels, ["record_id", *FEATURES, "is_rare"], 1049, "B_self")
    background_columns = ["record_id", *FEATURES] if list(background.columns) == ["record_id", *FEATURES] else FEATURES
    frame(background, background_columns, None, "A_bg")
    frame(self_challenge, ["challenge_row_id", *QI], 326, "self challenge")
    if set(self_labels.record_id) != set(candidates.record_id):
        raise ValueError("B_self IDs differ")
    aligned = self_labels.set_index("record_id").loc[candidates.record_id].reset_index()
    if not aligned[["record_id", *FEATURES]].equals(candidates.reset_index(drop=True)):
        raise ValueError("B_self public values differ")
    if schema.get("submitted_c_columns") != FEATURES or schema.get("aia_qi_columns") != QI or not isinstance(cfg, dict):
        raise ValueError("Unexpected public schema/config")
    for k, item in targets.items():
        if not isinstance(item, dict) or set(item) != {"release", "challenge"}:
            raise ValueError("Target accepts only public release and challenge")
        frame(item["release"], FEATURES, 1049, f"C_{k}")
        frame(item["challenge"], ["challenge_row_id", *QI], 326, f"challenge_{k}")


def predict_world(model_id, *, candidates, self_labels, background, self_challenge, targets, schema, cfg, attacker_id=10):
    """Infer from one complete public world using frozen production components."""
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    if model_id not in MODELS:
        raise ValueError("Unknown frozen model: " + str(model_id))
    public = dict(candidates=candidates, self_labels=self_labels, background=background,
                  self_challenge=self_challenge, targets=targets, schema=schema, cfg=cfg)
    validate_public(**public, attacker_id=attacker_id)
    public = copy.deepcopy(public)
    candidates, self_labels, background, self_challenge, targets, schema, cfg = (public[k] for k in public)
    prepared = _prepare(model_id)
    before = prepared["gate"]()
    started = time.monotonic()
    with _inference_guard(prepared["assets"], schema) as audit, threadpool_limits(limits=1):
        if model_id == "pl1":
            answer = prepared["model"].predict_world(**public)
        else:
            private, proof = prepared["derive_mask"](candidates.astype(str), self_labels.astype(str), self_challenge.astype(str), schema["aia_qi_columns"], schema["submitted_c_columns"])
            mia, aia_parts, routes, raw_t = None, [], {}, []
            if model_id == "Rv1":
                keys, matrix = prepared["numeric1"].predict_mia_all(targets={k: targets[k]["release"] for k in TARGETS}, background=background, candidates=candidates, private_mask=private, strength=1.0)
                mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy(), **{str(k): matrix[:, j] for j, k in enumerate(keys)}})
            attack = prepared["attack_class"]() if model_id in ("Rv1", "Rv2") else None
            if model_id == "Tv105":
                mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy()})
            for index, k in enumerate(TARGETS, 1):
                item = targets[k]
                args = dict(c=item["release"], background=background, candidates=candidates, challenge=item["challenge"], cfg=cfg)
                if model_id == "Rv1":
                    times, route = prepared["numeric1"].predict_aia_one(attack, **args)
                elif model_id == "Rv2":
                    answers, route = attack.predict(**args)
                    chosen = answers["combined"]
                    raw_t.append(prepared["numeric2"].ranks(chosen["mia"]))
                    times = chosen["aia"]
                else:
                    variants, route = prepared["model"].predict_all(**args, schema=schema)
                    chosen = prepared["synthesis"].predict(variants, prepared["selection"])
                    values = np.asarray(chosen["mia"], dtype=float).copy()
                    values[private] = 0.0
                    mia[str(k)] = values
                    times = chosen["aia"]
                routes[str(k)] = route
                aia_parts.append(pd.DataFrame({"target_k": k, "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(), "time_hat": np.asarray(times, dtype=float)}))
                print(f"{model_id}: public target {index}/22 complete ({k})", flush=True)
            if model_id == "Rv2":
                matrix = prepared["numeric2"].finish_mia(np.column_stack(raw_t), private)
                mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy(), **{str(k): matrix[:, j] for j, k in enumerate(TARGETS)}})
            answer = {"F_mia": mia, "F_aia": pd.concat(aia_parts, ignore_index=True),
                      "diagnostics": {"known_negative_proof": proof, "routes": routes}}
    if audit["blocked_reads"]:
        raise RuntimeError("A model suppressed a forbidden data-read error")
    after = prepared["gate"]()
    if before != after:
        raise ValueError("Frozen model changed during inference")
    from sugiyama_v2.scoring import validate_predictions
    validation = validate_predictions(answer["F_mia"], answer["F_aia"], {"team_id": 10, **public})
    answer["diagnostics"] = {**answer.get("diagnostics", {}), "model_id": model_id,
        "adapter_version": 1, "algorithm_changed": False,
        "selected_components_only": model_id == "Rv2", "private_mask_matches_production": True,
        "public_only_arguments": list(public), "frozen_gates": before, "validation": validation,
        "verified_asset_reads": sorted(audit["asset_reads"]), "canonical_public_schema_equal_to_supplied": True, "blocked_reads": audit["blocked_reads"],
        "seconds": time.monotonic() - started,
        "limitations": ["Historical fitted assets retain their original training exposure.",
                        "Trusted Python API and read audit are not an operating-system sandbox.",
                        "Numeric production parity must be established on the archived production inputs."]}
    return answer