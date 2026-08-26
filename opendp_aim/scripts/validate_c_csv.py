from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pwscup2026.common import config as config_mod
from pwscup2026.common import submission_schema
from pwscup2026.kit.validate import KIT_CONFIG_PATH, load_dist, rare_count_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="提出前の単体C.csv検証")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    cfg = config_mod.load_config(args.config or KIT_CONFIG_PATH)
    dist = load_dist(args.dist)
    frame = pd.read_csv(args.csv)
    result = submission_schema.validate_c_submission(
        frame, cfg, n_expected=dist.get("defense_rows")
    )
    if not result.ok:
        for error in result.errors:
            print(f"NG: {error}")
        raise SystemExit(1)

    diagnostic = rare_count_diagnostic(frame, dist)
    print(f"OK: {args.csv}")
    print(f"rows={len(frame)}, columns={len(frame.columns)}")
    if diagnostic is not None:
        print(
            f"rare_detected={diagnostic['rare_detected']}, "
            f"gate={diagnostic['gate']}"
        )


if __name__ == "__main__":
    main()
