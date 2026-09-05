"""Fail-closed loader for the user-supplied actual prelim team24 data.

There is deliberately no fallback to participant_data or B_practice.csv.
Callers supply --dist explicitly. Only the four enumerated data files are read;
token.txt is neither enumerated nor opened.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attack_phase_v3.generate_submission_v3 import discover_known_private_records
from attack_phase_v5.generate_submission_v5 import load_public_targets, verify_targets_manifest


EXPECTED_ACTUAL_HASHES = {
    "B.csv": "7A7AA39E16FA5C76EE9C8DC009928373AEC6CF47E7B1BD10DC4E6899B5D2AB3A",
    "B_self.csv": "297354A3C909FBA6BF4D33053CF91EFA9139892BF5269050A5607B86E0F33E35",
    "A_bg.csv": "F4817FBA6C1434D6B74221C2B5ABE90E4A53E80D8FA17E12602D43ADAE0A3126",
    "schema.json": "D0698EB89CDE0901B28DDAF72CB57270D744BDB72B45F8315F75B3199F024128",
}


@dataclass
class ActualInputs:
    candidates: pd.DataFrame
    self_frame: pd.DataFrame
    background: pd.DataFrame
    schema: dict[str, Any]
    releases: dict[str, pd.DataFrame]
    challenges: dict[str, pd.DataFrame]
    self_challenge: pd.DataFrame
    known_private: np.ndarray
    rare: np.ndarray
    provenance: dict[str, Any]


def verify_actual_hashes(dist: Path) -> dict[str, str]:
    actual = {}
    for name, expected in EXPECTED_ACTUAL_HASHES.items():
        path = Path(dist) / name
        if not path.is_file():
            raise ValueError(f"required actual prelim file missing: {name}; no practice-data fallback")
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if digest != expected:
            raise ValueError(f"actual prelim hash mismatch: {name}; do not substitute practice data")
        actual[name] = digest
    return actual


def load_actual_inputs(dist: Path, targets: Path) -> ActualInputs:
    hashes = verify_actual_hashes(dist)
    target_hashes = verify_targets_manifest(targets)
    candidates = pd.read_csv(Path(dist) / "B.csv")
    self_frame = pd.read_csv(Path(dist) / "B_self.csv")
    background = pd.read_csv(Path(dist) / "A_bg.csv")
    schema = json.loads((Path(dist) / "schema.json").read_text(encoding="utf-8"))
    releases, challenges = load_public_targets(targets)
    self_challenge = pd.read_csv(Path(targets) / "aia_challenge_24.csv")
    private, diagnostics = discover_known_private_records(candidates, self_challenge)
    rare = self_frame.is_rare.astype(bool).to_numpy()
    if (len(candidates) != 1049 or len(background) != 5143
            or not self_frame[candidates.columns].equals(candidates)
            or int(private.sum()) != 163 or int(rare.sum()) != 24
            or int((private & rare).sum()) != 13):
        raise ValueError("actual data cardinality/self-challenge invariants failed")
    provenance = {"data_phase": "actual_prelim_team24", "dist": str(Path(dist).resolve()),
        "input_hashes": hashes, "target_file_hashes": target_hashes,
        "known_private": diagnostics, "rare_rows": int(rare.sum()),
        "eligible_rare_rows": int((rare & ~private).sum()),
        "record_id_scope": "table-local pseudonym, not a cross-table identity key",
        "token_read": False, "practice_fallback_allowed": False}
    return ActualInputs(candidates, self_frame, background, schema, releases,
                        challenges, self_challenge, private, rare, provenance)
