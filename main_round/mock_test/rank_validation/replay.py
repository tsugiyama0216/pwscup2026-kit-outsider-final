"""Read-only verification of a score handoff; reports go to a fresh output path."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes handoff")
    return path


def verify(root):
    entries = read(root / "manifest.json")["entries"]
    for item in entries:
        content = within(root, item["path"]).read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Handoff hash mismatch: " + item["path"])
    return {"files": len(entries), "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}



MAIN_TARGETS = frozenset((set(range(18)) | {22, 25, 26, 27, 28}) - {10})


def expected_targets(attacker_team_id):
    """The handoff supports isolated target 0 or the pinned main-round roster."""
    if attacker_team_id is None:
        return frozenset({0})
    if type(attacker_team_id) is int and attacker_team_id == 10:
        return MAIN_TARGETS
    raise ValueError("Unsupported replay protocol: attacker_team_id must be null or integer 10")


def read_csv_strict(path):
    import pandas as pd
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not value for value in header) or len(header) != len(set(header)):
        raise ValueError(f"Missing or duplicate CSV columns: {Path(path).name}")
    return pd.read_csv(path, float_precision="round_trip")


def _columns(frame, expected, label):
    if not frame.columns.is_unique or set(frame.columns) != set(expected):
        raise ValueError(f"{label}: missing, duplicate or extra columns")


def _ids(values, label, *, unique=False):
    import numpy as np
    import pandas as pd
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{label}: IDs must be integers, not booleans")
    numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers != np.floor(numbers)).any() or (np.abs(numbers) >= 2**53).any():
        raise ValueError(f"{label}: IDs must be finite exact integers")
    numbers = numbers.astype(np.int64)
    if unique and len(np.unique(numbers)) != len(numbers):
        raise ValueError(f"{label}: duplicate IDs")
    return numbers


def _values(values, label, *, maximum=None, binary=False):
    import numpy as np
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{label}: values must be finite and nonnegative")
    if maximum is not None and (array > maximum).any():
        raise ValueError(f"{label}: values exceed {maximum}")
    if binary and not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label}: values must be binary")
    return array


def _pool(frame, label, *, rows=None):
    _columns(frame, {"record_id", "pool_id", "is_rare"}, label)
    ids = _ids(frame.record_id, label + " record_id", unique=True)
    _ids(frame.pool_id, label + " pool_id", unique=True)
    _values(frame.is_rare, label + " is_rare", binary=True)
    if frame.empty or (rows is not None and len(frame) != rows):
        raise ValueError(f"{label}: unexpected pool row count")
    return ids


def validate_frames(mia, aia, attacker_pool, target_pools, aia_truths, floor, attacker_team_id):
    """Fail before official inner joins and target means can discard bad IDs.

    The expected roster comes from the declared protocol, never from agreement
    between two possibly incomplete prediction files. Every truth table is
    checked too, including the floor whose absence would otherwise mean zero.
    """
    import numpy as np
    targets = expected_targets(attacker_team_id)
    if set(target_pools) != targets or set(aia_truths) != targets:
        raise ValueError("Truth target roster differs from the complete replay protocol")
    candidate_ids = _pool(attacker_pool, "attacker pool", rows=1049)
    _columns(mia, {"record_id", *(str(k) for k in targets)}, "F_mia")
    prediction_ids = _ids(mia.record_id, "F_mia record_id", unique=True)
    if len(mia) != 1049 or set(prediction_ids) != set(candidate_ids):
        raise ValueError("MIA candidate IDs differ from attacker truth")
    _values(mia.drop(columns="record_id"), "MIA confidence", maximum=1)
    _columns(aia, {"target_k", "challenge_row_id", "time_hat"}, "F_aia")
    target_ids = _ids(aia.target_k, "F_aia target_k")
    query_ids = _ids(aia.challenge_row_id, "F_aia challenge_row_id")
    if set(target_ids) != targets:
        raise ValueError("AIA target roster differs from the complete replay protocol")
    pairs = list(zip(target_ids, query_ids))
    if len(set(pairs)) != len(pairs):
        raise ValueError("F_aia: duplicate target/challenge IDs")
    _values(aia.time_hat, "AIA time_hat")
    for target in sorted(targets):
        _pool(target_pools[target], f"target pool {target}", rows=1049 if attacker_team_id is not None else None)
        truth = aia_truths[target]
        _columns(truth, {"challenge_row_id", "label", "onset", "time", "is_rare"}, f"AIA truth {target}")
        truth_ids = _ids(truth.challenge_row_id, f"AIA truth {target} IDs", unique=True)
        if len(truth) != 326 or truth.label.value_counts().to_dict() != {"member": 163, "control": 163}:
            raise ValueError(f"AIA truth {target}: expected 163 members and 163 controls")
        _values(truth.onset, f"AIA truth {target} onset", binary=True)
        _values(truth.is_rare, f"AIA truth {target} is_rare", binary=True)
        _values(truth.time, f"AIA truth {target} time")
        predicted = query_ids[target_ids == target]
        if len(predicted) != 326 or set(predicted) != set(truth_ids):
            raise ValueError(f"AIA challenge IDs differ from truth for target {target}")
    _columns(floor, {"team_id", "aia_floor"}, "AIA floor")
    floor_ids = _ids(floor.team_id, "AIA floor team_id", unique=True)
    if set(floor_ids) != targets:
        raise ValueError("AIA floor target roster differs from the complete replay protocol")
    _values(floor.aia_floor, "AIA floor", maximum=1)
    return {"candidate_rows": len(mia), "challenge_rows": len(aia), "target_count": len(targets),
            "expected_roster_verified": True, "prediction_and_truth_ids_verified": True}


def load_truth(truth, attacker_team_id):
    targets = expected_targets(attacker_team_id)
    attacker = "attacker" if attacker_team_id is None else str(attacker_team_id)
    expected_aia = {f"aia_truth_{k}.csv" for k in targets}
    expected_pools = {f"pool_ids_{k}.csv" for k in targets} | {f"pool_ids_{attacker}.csv"}
    if {p.name for p in truth.glob("aia_truth_*.csv")} != expected_aia:
        raise ValueError("AIA truth files differ from the complete replay protocol")
    if {p.name for p in truth.glob("pool_ids_*.csv")} != expected_pools:
        raise ValueError("Pool truth files differ from the complete replay protocol")
    return (read_csv_strict(truth / f"pool_ids_{attacker}.csv"),
            {k: read_csv_strict(truth / f"pool_ids_{k}.csv") for k in targets},
            {k: read_csv_strict(truth / f"aia_truth_{k}.csv") for k in targets},
            read_csv_strict(truth / "aia_floor.csv"))


def replay(root):
    import numpy as np
    import pandas as pd

    source = (root / "reference/official-kit/src").resolve()
    for name, module in list(sys.modules.items()):
        if name == "pwscup2026" or name.startswith("pwscup2026."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(source):
                raise RuntimeError("Run this pinned scorer in a fresh process")
    sys.path.insert(0, str(source))
    from pwscup2026.kit.codabench.scoring_io import score_attack

    if not Path(score_attack.__code__.co_filename).resolve().is_relative_to(source):
        raise RuntimeError("Wrong official scorer")
    results, failures = [], []
    start = time.monotonic()
    fixtures = read(root / "data/replay_fixtures.json")
    for index, fixture in enumerate(fixtures, 1):
        try:
            mia = read_csv_strict(within(root, fixture["F_mia"]))
            aia = read_csv_strict(within(root, fixture["F_aia"]))
            truth = within(root, fixture["truth_dir"])
            config = read(within(root, fixture["config"]))
            attacker_pool, target_pools, aia_truths, floor = load_truth(truth, fixture["attacker_team_id"])
            validation = validate_frames(mia, aia, attacker_pool, target_pools, aia_truths, floor,
                                         fixture["attacker_team_id"])
            score = score_attack({"F_mia.csv": mia, "F_aia.csv": aia}, truth, config, team_id=fixture["attacker_team_id"])
            errors = {key: abs(float(score[key]) - float(value)) for key, value in fixture["expected_metrics"].items()}
            if not errors or not all(math.isfinite(value) and value <= 1e-12 for value in errors.values()):
                raise ValueError("Historical score mismatch: " + str(errors))
            results.append({"fixture_id": fixture["fixture_id"], "maximum_absolute_error": max(errors.values()), "validation": validation})
        except Exception as exc:
            failures.append({"fixture_id": fixture["fixture_id"], "error": str(exc)})
        if index % 100 == 0:
            print(json.dumps({"checked": index, "total": len(fixtures), "failed": len(failures)}), flush=True)
    return {"fixture_count": len(fixtures), "passed": len(results), "failed": len(failures),
            "maximum_absolute_error": max((row["maximum_absolute_error"] for row in results), default=None),
            "failures": failures, "seconds": time.monotonic() - start,
            "official_hidden_truth_tested": False, "new_model_selection_performed": False,
            "complete_roster_and_ids_verified_for_passed_fixtures": True,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.handoff.resolve()
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parents[1] / "local-output"
    if not output.is_relative_to(allowed.resolve()) or output.exists():
        parser.error("Output must be a new file under mock_test/local-output")
    before = verify(root)
    print(json.dumps({"hash_verification": before}), flush=True)
    report = {"hash_verification": before, **replay(root)}
    if verify(root) != before:
        raise RuntimeError("Handoff changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
