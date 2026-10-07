"""Apply frozen R v1 to public Sugiyama world inputs without reading real answers.

This adapter changes input plumbing only: its predictions use R v1's original
M v2 MIA and T v88 AIA functions, selections and fitted assets. It accepts no
case paths, generation seeds, release-style labels or private evaluation truth.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ATTACK = HERE.parent
FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
            "smoking", "FPG", "time", "onset", "death"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
TARGETS = tuple(list(range(10)) + list(range(11, 18)) + [22, 25, 26, 27, 28])
MODEL_NAME = "Attack_R/v1"


def _file_module(name, path):
    """Load an exact original source without occupying generic common/model names."""
    path = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ImportError(f"Conflicting frozen module origin: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import frozen source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_frozen():
    for path in (ATTACK, ATTACK / "Attak_T"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # T initializes the shared local_eval before the frozen M compatibility
    # package is imported; reversing this order is explicitly unsupported.
    import attack_research
    if Path(attack_research.__file__).resolve() != (ATTACK / "Attak_T/attack_research/__init__.py").resolve():
        raise ImportError("R v1 adapter requires the canonical T package")
    common = _file_module("_r_v1_sugiyama_common", HERE / "v1/common.py")
    model = _file_module("_r_v1_sugiyama_model", HERE / "v1/model.py")
    from attack_research.published_v88.check import verify_package
    from attack_research.v88.predict import CombinedAttack
    from M.v1.actual import derive_private_mask
    from M.v2.artifacts import load_selection, check_v1_preserved

    return common, model, verify_package, CombinedAttack, derive_private_mask, load_selection, check_v1_preserved


def _frame(frame, columns, rows, name):
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != columns:
        raise ValueError(f"{name} must contain exactly the declared public columns")
    if (rows is not None and len(frame) != rows) or frame.empty:
        raise ValueError(f"Unexpected public input row count: {name}")
    return frame.copy(deep=True)


def predict_world(*, candidates, self_labels, background, self_challenge, targets, schema, cfg):
    """Return submission-shaped F_mia/F_aia using only supplied public inputs."""
    started = perf_counter()
    candidates = _frame(candidates, ["record_id"] + FEATURES, 1049, "B")
    self_labels = _frame(self_labels, ["record_id"] + FEATURES + ["is_rare"], 1049, "B_self")
    background = _frame(background, FEATURES, None, "A_bg")
    self_challenge = _frame(self_challenge, ["challenge_row_id"] + QI, 326, "self challenge")
    if not isinstance(targets, dict) or set(targets) != set(TARGETS):
        raise ValueError("R v1 requires all 22 nonself targets in a single world")
    releases, challenges = {}, {}
    for target in TARGETS:
        item = targets[target]
        if not isinstance(item, dict) or set(item) != {"release", "challenge"}:
            raise ValueError("Each target accepts only its public release and challenge")
        releases[target] = _frame(item["release"], FEATURES, 1049, f"C_{target}")
        challenges[target] = _frame(item["challenge"], ["challenge_row_id"] + QI, 326,
                                    f"challenge_{target}")
    schema, cfg = copy.deepcopy(schema), copy.deepcopy(cfg)
    if schema.get("submitted_c_columns") != FEATURES or schema.get("aia_qi_columns") != QI:
        raise ValueError("R v1 requires the unchanged official public feature schema")

    common, model, verify_t, attack_class, derive_mask, selection_loader, preserve_m = _load_frozen()
    dependencies = common.verify_dependencies()
    published = verify_t()
    selection = selection_loader()
    preservation = preserve_m()
    if selection["mia_strength"] != 1.0:
        raise ValueError("R v1 requires frozen M v2 MIA strength 1.0")
    try:
        # Decimal equality uses representations of these supplied tables only.
        # is_rare is permitted self information, not any target's hidden truth.
        private, proof = derive_mask(candidates.astype(str), self_labels.astype(str),
            self_challenge.astype(str), schema["aia_qi_columns"], schema["submitted_c_columns"])
        keys, values = model.predict_mia_all(targets=releases, background=background,
            candidates=candidates, private_mask=private, strength=selection["mia_strength"])
        values = np.asarray(values, dtype=float)
        if keys != sorted(TARGETS) or values.shape != (1049, 22) or not np.isfinite(values).all():
            raise ValueError("Frozen R v1 MIA returned invalid dimensions or values")
        if ((values < 0) | (values > 1)).any() or not (values[private] == 0).all():
            raise ValueError("Frozen R v1 MIA violated its output/private-mask constraints")
        f_mia = pd.DataFrame({"record_id": candidates.record_id.to_numpy(copy=True),
                             **{str(k): values[:, j] for j, k in enumerate(keys)}})
        print("R v1 Sugiyama: MIA completed for all 22 targets", flush=True)

        attack = attack_class()
        answers, routes = [], {}
        for position, target in enumerate(keys, 1):
            times, route = model.predict_aia_one(attack, c=releases[target], background=background,
                candidates=candidates, challenge=challenges[target], cfg=cfg)
            answers.append(pd.DataFrame({"target_k": target,
                "challenge_row_id": challenges[target].challenge_row_id.to_numpy(copy=True),
                "time_hat": times}))
            routes[str(target)] = str(route)
            print(f"R v1 Sugiyama: AIA {position}/22 completed (target {target}; inferred route {route})", flush=True)
        f_aia = pd.concat(answers, ignore_index=True)
    finally:
        # Preserve both frozen runtime gates even when inference fails.
        if common.verify_dependencies() != dependencies or verify_t() != published:
            raise ValueError("Frozen R/T dependencies changed during Sugiyama inference")
        selection_loader()
        if preserve_m() != preservation:
            raise ValueError("Frozen M v1 preservation changed during Sugiyama inference")

    return {"F_mia": f_mia, "F_aia": f_aia, "diagnostics": {
        "model": "Attack_R/v1", "algorithm_modified": False,
        "input_scope": "Supplied public tables only; no actual-data or stored-prediction loader",
        "R_v1_dependencies": dependencies, "T_v88": published, "M_v1_preservation": preservation,
        "mia_strength": selection["mia_strength"], "private_inference": proof,
        "inferred_routes": routes, "target_count": len(keys),
        "elapsed_seconds": perf_counter() - started,
    }}
