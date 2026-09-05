from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generate_pwscup import (
    load_json,
    load_public_background,
    replace_outcomes_with_nearest_public_donors,
    validate_public_configuration,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "既存のDP C.csvに、公開A_bgだけを使う転帰ドナー後処理を再適用する。"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parent-privacy-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    schema = load_json(args.schema)
    validate_public_configuration(config, schema)

    parent = load_json(args.parent_privacy_report)
    input_hash = sha256(args.input)
    parent_output_hash = str(parent.get("output_sha256", "")).lower()
    if input_hash != parent_output_hash:
        raise ValueError(
            "--input does not match output_sha256 in --parent-privacy-report"
        )

    output = pd.read_csv(args.input, encoding="utf-8")
    original_lines = args.input.read_text(encoding="utf-8-sig").splitlines()
    if list(output.columns) != list(schema["submitted_c_columns"]):
        raise ValueError("--input columns do not match schema.submitted_c_columns")
    if len(output) != int(config["output_rows"]):
        raise ValueError("--input row count does not match config.output_rows")

    background = load_public_background(args.background, config)
    replace_outcomes_with_nearest_public_donors(output, background, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the parent's covariate text exactly and replace only the final
    # time/onset/death fields. This avoids irrelevant float round-trip changes.
    rendered_lines = [original_lines[0]]
    for original, row in zip(original_lines[1:], output.itertuples(index=False)):
        prefix = original.rsplit(",", 3)[0]
        rendered_lines.append(f"{prefix},{row.time:.12g},{int(row.onset)},{int(row.death)}")
    args.output.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "public_outcome_donor_repostprocess",
        "privacy_interpretation": (
            "The parent DP release is unchanged in its covariates. Outcomes are "
            "replaced using only public A_bg and the parent DP output. For a "
            "precommitted public config this is postprocessing with no additional rho."
        ),
        "selection_warning": (
            "The v5 config was selected with B_practice utility diagnostics; that "
            "selection experiment is outside the parent privacy report."
        ),
        "parent_input": str(args.input.resolve()),
        "parent_input_sha256": input_hash,
        "parent_privacy_report": str(args.parent_privacy_report.resolve()),
        "parent_privacy_report_sha256": sha256(args.parent_privacy_report),
        "parent_mechanism": parent.get("mechanism"),
        "parent_privacy": parent.get("privacy"),
        "parent_dp_release_count": parent.get("dp_release_count"),
        "background": str(args.background.resolve()),
        "background_sha256": sha256(args.background),
        "schema": str(args.schema.resolve()),
        "schema_sha256": sha256(args.schema),
        "config": str(args.config.resolve()),
        "config_version": config.get("config_version"),
        "config_sha256": sha256(args.config),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_rows": len(output),
        "public_outcome_donor": config["postprocess"]["public_outcome_donor"],
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote output: {args.output}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
