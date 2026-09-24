"""Estimate the rulebook AIA baseline phi_i from a truth-labeled challenge.

The organizer's phi_i is the mean of 1,000 AIA scores after replacing C_i by
1,049-row samples from A_bg.  This helper follows that recipe when a local
truth-labeled challenge is available.  The public attack challenge normally
does not expose its truth columns, so a result made before the attack phase is
only a local estimate and must not be called the official phi_i.

Expected challenge columns are the eight public AIA columns plus ``is_target``,
``time`` and ``onset``.  ``is_target`` is 1 for a target row and 0 for its
matched control.  An optional ``is_rare`` column is ignored; rarity is derived
from schema.json in the same way as the scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PUBLIC_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
TRUTH_COLUMNS = (*PUBLIC_COLUMNS, "time", "onset", "is_target")


def rare_mask(frame: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    detector = schema["rare_detector"]
    values = frame[detector["columns"]].to_numpy(dtype=float)
    diff = values - np.asarray(detector["mu"], dtype=float)
    inv_sigma = np.linalg.pinv(np.asarray(detector["sigma"], dtype=float))
    distance = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv_sigma, diff), 0.0, None))
    return distance >= float(detector["band"]["lo"])


def challenge_parts(challenge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(TRUTH_COLUMNS) - set(challenge.columns))
    if missing:
        raise ValueError(f"challenge is missing columns: {missing}")
    if challenge["is_target"].isna().any() or not set(challenge["is_target"].unique()).issubset({0, 1, False, True}):
        raise ValueError("challenge.is_target must contain only 0/1")
    targets = challenge.loc[challenge["is_target"].astype(bool)].copy()
    controls = challenge.loc[~challenge["is_target"].astype(bool)].copy()
    if len(targets) == 0 or len(targets) != len(controls):
        raise ValueError("challenge must contain equal non-empty target/control groups")
    return targets, controls


def estimate_phi(
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    schema: dict[str, Any],
    cfg: dict[str, Any],
    aia_predict: Any,
    score_aia: Any,
    *,
    repetitions: int = 1000,
    sample_size: int = 1049,
    seed: int = 20260924,
) -> dict[str, Any]:
    if repetitions <= 0 or sample_size <= 0 or sample_size > len(background):
        raise ValueError("repetitions must be positive and sample_size must fit A_bg")
    targets, controls = challenge_parts(challenge)
    rare_target = rare_mask(targets, schema)
    rare_control = rare_mask(controls, schema)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        selected = rng.choice(len(background), size=sample_size, replace=False)
        released = background.iloc[selected].reset_index(drop=True)
        _, target_hat = aia_predict(released, targets, background, cfg)
        _, control_hat = aia_predict(released, controls, background, cfg)
        result = score_aia(
            None,
            target_hat,
            targets["onset"].to_numpy(),
            targets["time"].to_numpy(),
            None,
            control_hat,
            controls["onset"].to_numpy(),
            controls["time"].to_numpy(),
            cfg,
            member_rare_mask=rare_target,
            control_rare_mask=rare_control,
        )
        values[rep] = float(result.score)
    return {
        "schema_version": "team10-phi-estimate-v1",
        "scope": "local estimate of rulebook phi_i; official value requires organizer challenge construction and sampling",
        "phi_i_estimate": float(values.mean()),
        "repetitions": repetitions,
        "sample_size": sample_size,
        "seed": seed,
        "score_min": float(values.min()),
        "score_max": float(values.max()),
        "score_std": float(values.std(ddof=1)) if repetitions > 1 else 0.0,
        "challenge_rows": len(challenge),
        "target_rows": len(targets),
        "control_rows": len(controls),
        "inputs": {
            "challenge_sha256": hashlib.sha256(challenge.to_csv(index=False).encode()).hexdigest(),
            "background_rows": len(background),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="directory containing A_bg.csv and schema.json")
    parser.add_argument("--kit-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=1049)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    kit_source = args.kit_root / "starter" / "src"
    sys.path.insert(0, str(kit_source.resolve()))
    from pwscup2026.attack.aia import aia_predict
    from pwscup2026.scoring.attack_score import score_aia

    cfg = yaml.safe_load((args.kit_root / "participant_data" / "scoring_config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("aia", {}).update({"qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"], "block_keys": ["sex", "prefecture"], "k_nn_onset": 5})
    challenge = pd.read_csv(args.challenge)
    background = pd.read_csv(args.data / "A_bg.csv")
    schema = json.loads((args.data / "schema.json").read_text(encoding="utf-8"))
    result = estimate_phi(
        challenge,
        background,
        schema,
        cfg,
        aia_predict,
        score_aia,
        repetitions=args.repetitions,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
