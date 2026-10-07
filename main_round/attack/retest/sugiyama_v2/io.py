"""Versioned public/private case storage; attack plugins receive public tables only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import attack_workspace as workspace
from local_eval.data import FEATURES, QI

TEAM_ID = 10
TARGETS = tuple(k for k in (*range(18), 22, 25, 26, 27, 28) if k != TEAM_ID)
FORMAT = "sugiyama-official-shape-v2"


def sha(path):
    if Path(path).name.lower() == "token.txt":
        raise ValueError("No credentials belong in mock evaluation")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def output_path(path):
    root = (workspace.MOCK_OUTPUT).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Use a new child folder under mock_test/local-output")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Preserve previous output; choose an empty folder")
    return path


def contained(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Manifest paths must be relative POSIX paths")
    rel = PurePosixPath(name)
    path = (root / name).resolve()
    if (rel.is_absolute() or rel.as_posix() != name or ".." in rel.parts
            or not path.is_relative_to(root) or path == root):
        raise ValueError("Case path escapes its root")
    return path


def source_manifest():
    return {p.relative_to(workspace.ROOT.parent.parent).as_posix(): sha(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def csv(path, frame):
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def export_world(world, destination):
    metadata = world["metadata"]
    if (not isinstance(metadata, dict) or metadata.get("synthetic", True) is not True
            or metadata.get("format", FORMAT) != FORMAT):
        raise ValueError("Only explicitly synthetic world metadata can be exported")
    root = output_path(destination)
    root.mkdir(parents=True, exist_ok=True)
    pub, private = root / "public", root / "private"
    (pub / "targets").mkdir(parents=True)
    private.mkdir()
    public, hidden = world["public"], world["private"]
    if public["team_id"] != TEAM_ID or set(public["targets"]) != set(TARGETS):
        raise ValueError("Expected the official 22 nonself targets")
    for name, key in (("B.csv", "candidates"), ("B_self.csv", "self_labels"),
                      ("A_bg.csv", "background"), ("aia_challenge_10.csv", "self_challenge")):
        csv(pub / name, public[key])
    save_json(pub / "schema.json", public["schema"])
    pool = hidden["pool"].set_index("pool_id")
    ids = np.asarray(hidden["candidate_pool_ids"], dtype=int)
    csv(private / "pool_ids_10.csv", pd.DataFrame({
        "record_id": public["candidates"].record_id.to_numpy(), "pool_id": ids,
        "is_rare": pool.loc[ids, "is_rare"].to_numpy(dtype=int)}))
    for target in TARGETS:
        item = public["targets"][target]
        csv(pub / "targets" / f"C_{target}.csv", item["release"])
        csv(pub / "targets" / f"aia_challenge_{target}.csv", item["challenge"])
        ids = np.asarray(hidden["cohorts"][target], dtype=int)
        csv(private / f"pool_ids_{target}.csv", pd.DataFrame({"record_id": np.arange(len(ids)),
            "pool_id": ids, "is_rare": pool.loc[ids, "is_rare"].to_numpy(dtype=int)}))
        csv(private / f"aia_truth_{target}.csv", hidden["aia_truth"][target])
    # These provenance/role fields are not included in the prediction API.
    save_json(private / "case_metadata.json", {**world["metadata"], "format": FORMAT, "synthetic": True})
    save_json(private / "roles.json", {key: hidden[key] for key in (
        "candidate_pool_ids", "cohorts", "challenge_pool_ids") if key in hidden})
    files = {p.relative_to(root).as_posix(): sha(p) for folder in (pub, private)
             for p in sorted(folder.rglob("*")) if p.is_file()}
    manifest = {"format": FORMAT, "team_id": TEAM_ID, "targets": list(TARGETS),
        "metadata": world["metadata"], "files_sha256": files, "sources_sha256": source_manifest(),
        "source_hash_policy": "Creation provenance only; current source changes do not invalidate frozen case inputs.",
        "original_input_sha256": {name: sha(workspace.ORIGINALS / name)
            for name in ("B_self.csv", "A_bg.csv", "schema.json")},
        "official_score_prediction": False, "live_credentials": False}
    save_json(root / "manifest.json", manifest)
    load_world(root)
    return manifest


def snapshot(root):
    """Validate saved input bytes, not the current generator's source revision.

    Source/original hashes document how this immutable case was created. They
    intentionally do not prevent re-evaluating saved cases after code changes.
    """
    root = Path(root).resolve()
    manifest_path = contained(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("team_id") != TEAM_ID or manifest.get("targets") != list(TARGETS):
        raise ValueError("Unsupported world contract")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing case hashes")
    if manifest.get("official_score_prediction") is not False or manifest.get("live_credentials") is not False:
        raise ValueError("Saved worlds must be credential-free local simulations")
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("Missing world metadata")
    required = {"public/B.csv", "public/B_self.csv", "public/A_bg.csv", "public/schema.json",
                "public/aia_challenge_10.csv", "private/pool_ids_10.csv", "private/case_metadata.json"}
    for target in TARGETS:
        required.update((f"public/targets/C_{target}.csv", f"public/targets/aia_challenge_{target}.csv",
                         f"private/pool_ids_{target}.csv", f"private/aia_truth_{target}.csv"))
    if not required.issubset(files):
        raise ValueError("Incomplete world manifest")
    for name, expected in files.items():
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Invalid case SHA256")
        path = contained(root, name)
        if PurePosixPath(name).parts[0] not in {"public", "private"}:
            raise ValueError("Case files must be public or private inputs")
        if not path.is_file() or sha(path) != expected:
            raise ValueError("World file changed: " + name)
    case_metadata = json.loads((root / "private" / "case_metadata.json").read_text(encoding="utf-8"))
    if case_metadata.get("synthetic") is not True or case_metadata.get("format") != FORMAT:
        raise ValueError("Private truth must identify a synthetic Sugiyama world")
    expected_metadata = {**manifest.get("metadata", {}), "format": FORMAT, "synthetic": True}
    if case_metadata != expected_metadata:
        raise ValueError("Private and manifest case metadata differ")
    return manifest, {"manifest": sha(manifest_path), **files}


def _frame(path, columns, n):
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) != list(columns) or len(frame) != n:
        raise ValueError("Unexpected table schema/rows: " + str(path))
    numeric = frame.drop(columns=["prefecture"], errors="ignore").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite table values")
    if "prefecture" in frame and (frame.prefecture.isna().any()
            or not frame.prefecture.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("Missing or invalid prefecture")
    return frame


def _ids(frame, key):
    values = frame[key].to_numpy(float)
    if (not np.isfinite(values).all() or len(np.unique(values)) != len(values)
            or (values != np.floor(values)).any() or (values < 0).any()
            or (values >= 2**53).any()):
        raise ValueError("IDs must be unique, nonnegative, exactly represented integers")


def load_world(path):
    root = Path(path).resolve()
    manifest, before = snapshot(root)
    pub = root / "public"
    candidates = _frame(pub / "B.csv", ["record_id"] + FEATURES, 1049)
    labels = _frame(pub / "B_self.csv", ["record_id"] + FEATURES + ["is_rare"], 1049)
    _ids(candidates, "record_id")
    _ids(labels, "record_id")
    if set(labels.record_id) != set(candidates.record_id):
        raise ValueError("Self-label IDs differ from candidate IDs")
    labels = labels.set_index("record_id").loc[candidates.record_id].reset_index()
    if not labels[["record_id"] + FEATURES].equals(candidates):
        raise ValueError("B_self public values differ from B")
    if not labels.is_rare.isin([0, 1]).all() or int(labels.is_rare.sum()) != 24:
        raise ValueError("Own labels must contain exactly 24 latent rare people")
    challenge = _frame(pub / "aia_challenge_10.csv", ["challenge_row_id"] + QI, 326)
    _ids(challenge, "challenge_row_id")
    own_keys = set(map(tuple, candidates[QI].to_numpy()))
    self_keys = set(map(tuple, challenge[QI].to_numpy()))
    if (len(own_keys) != len(candidates) or len(self_keys) != len(challenge)
            or len(own_keys & self_keys) != 163):
        raise ValueError("Self challenge must identify exactly 163 unique private candidates")
    targets = {}
    for target in TARGETS:
        c = _frame(pub / "targets" / f"C_{target}.csv", FEATURES, 1049)
        a = _frame(pub / "targets" / f"aia_challenge_{target}.csv", ["challenge_row_id"] + QI, 326)
        _ids(a, "challenge_row_id")
        targets[target] = {"release": c, "challenge": a}
    public = {"team_id": TEAM_ID, "candidates": candidates, "self_labels": labels,
        "background": _frame(pub / "A_bg.csv", FEATURES, 5143), "self_challenge": challenge,
        "targets": targets, "schema": json.loads((pub / "schema.json").read_text(encoding="utf-8"))}
    if snapshot(root)[1] != before:
        raise ValueError("World changed during reading")
    return public, root / "private", manifest, before


def prediction_arguments(public, cfg):
    """No paths, seeds, style labels, truth, or floors cross the API boundary."""
    return copy.deepcopy({key: public[key] for key in (
        "candidates", "self_labels", "background", "self_challenge", "targets", "schema")}) | {"cfg": copy.deepcopy(cfg)}
