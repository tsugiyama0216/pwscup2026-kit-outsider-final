"""Frozen R v2 selected T/T components for a single-target mock_TS diagnostic.

This does not claim fresh training on the reduced mock background. The frozen
T models, including donor-supported fitted priors, retain their training history.
Self labels/challenge are unavailable, so no known-negative mask is invented.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RV2 = HERE.parent
ATTACK = HERE.parents[2]
ROOT = ATTACK.parents[1]
TS = ATTACK / "Attak_T"
SCHEMA = ROOT / "main_round/anonymization/team10/participant_data/schema.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _input_guard(allowed_assets):
    """Catch accidental local data reads; this is not an OS security boundary."""
    state = {"active": True, "blocked": [], "asset_reads": set()}
    allowed = {p.resolve() for p in allowed_assets} | {SCHEMA.resolve()}
    public_schema = (Path.cwd() / "schema.json").resolve()
    allowed.add(public_schema)

    def audit(event, args):
        if event != "open" or not state["active"] or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        if (path.is_relative_to(ATTACK / ".tools/python")
                or path.is_relative_to(ATTACK / ".venv")):
            return  # Interpreter archives and installed Python packages.
        suffix = path.suffix.lower()
        if path in allowed:
            state["asset_reads"].add(str(path.relative_to(ROOT)))
            return
        # No CSV, private problem files, prediction caches, or unlisted model
        # assets are needed by the selected frozen inference components.
        private_output = ROOT / "main_round/mock_test/local-output"
        participant = ROOT / "main_round/anonymization/team10/participant_data"
        targets = ATTACK / "targets"
        forbidden = (suffix in {".csv", ".parquet", ".joblib", ".pkl", ".pickle", ".zip"}
                     or path.is_relative_to(private_output)
                     or path.is_relative_to(participant)
                     or path.is_relative_to(targets))
        if forbidden:
            state["blocked"].append(str(path.relative_to(ROOT)))
            raise RuntimeError("Frozen R v2 prediction attempted an unapproved data read: " + str(path))

    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


def predict(*, c, background, candidates, challenge, cfg):
    selection = json.loads((RV2 / "selection.json").read_text(encoding="utf-8"))
    if (selection["mia"], selection["aia"]) != ("T", "T"):
        raise ValueError("This adapter is only for frozen R v2 T/T")
    for name, expected in selection["source_sha256"].items():
        if _sha(RV2 / name) != expected:
            raise ValueError("Frozen R v2 source changed: " + name)
    manifest = json.loads((RV2 / "production_dependencies.json").read_text(encoding="utf-8"))
    assets = {ROOT / name: expected for name, expected in manifest["files_sha256"].items()
              if name.startswith("main_round/attack/Attak_T/local-output/")
              and Path(name).suffix in {".json", ".joblib"}}
    for path, expected in assets.items():
        if _sha(path) != expected:
            raise ValueError("Frozen T fitted asset changed: " + str(path))
    if _sha(Path.cwd() / "schema.json") != _sha(SCHEMA):
        raise ValueError("Public schema differs from the schema required by frozen T")
    for path in (ATTACK, TS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    with _input_guard(assets) as audit:
        import attack_research  # canonical workspace/import initialization
        from attack_research.v88.predict import CombinedAttack
        spec = importlib.util.spec_from_file_location("_rv2_frozen_numeric", RV2 / "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        answers, route = CombinedAttack().predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        chosen = answers["combined"]
        # R v2 ranks T, then applies finish_mia. Its chosen cross-target
        # correction strength is zero, so a one-column matrix is sufficient.
        mia = model.finish_mia(model.ranks(chosen["mia"])[:, None],
                               np.zeros(len(candidates), dtype=bool))[:, 0]
        aia = np.asarray(chosen["aia"], dtype=float)

    if audit["blocked"]:
        raise RuntimeError("A dependency suppressed a forbidden data-read exception")

    diagnostics = {"adapter": "frozen_R_v2_single_target_diagnostic", "route": str(route),
        "background_rows": len(background), "known_negative_mask_applied": False,
        "reason": "mock_TS supplies no self_labels or self_challenge",
        "fresh_training_protocol_compliant": False,
        "frozen_training_exposure_retained": True, "blocked_data_reads": audit["blocked"],
        "fitted_assets_verified": len(assets), "asset_reads": sorted(audit["asset_reads"])}
    print("RV2_MOCK_TS_DIAGNOSTICS " + json.dumps(diagnostics, ensure_ascii=False), flush=True)
    return {"mia": mia, "aia": aia}
