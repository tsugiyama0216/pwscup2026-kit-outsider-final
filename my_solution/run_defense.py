"""参加者用の防御ベースライン。

このファイルの ``build_defense`` を編集すると、公式の参照コードを残したまま
自分の匿名化方式を試せます。入出力整形と提出前検証はキット側に任せます。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pwscup2026.defense import synth as synth_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "starter" / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

from _common import coerce_to_submission, dist_paths, load_cfg, read_b, write_c  # noqa: E402


DefenseFunction = Callable[[pd.DataFrame, dict, np.random.Generator], pd.DataFrame]
METHODS: dict[str, DefenseFunction] = {
    "copula_survival": synth_mod.copula_survival,
    "copula_synth": synth_mod.copula_synth,
}


def build_defense(
    b_df: pd.DataFrame,
    cfg: dict,
    rng: np.random.Generator,
    method: str,
) -> pd.DataFrame:
    """匿名化データ C を作る、最初のカスタマイズ箇所。

    既定では公式参照実装を呼びます。独自方式へ置き換える場合も、入力と同じ行数を返し、
    最後に ``coerce_to_submission`` を通してください。提出 C から record_id を落とす処理は
    ``write_c`` が担当します。
    """

    synthesized = METHODS[method](b_df, cfg, rng)
    if "record_id" not in synthesized.columns:
        # record_id は提出時に削除されるが、キットの整形処理が一時的に必要とする。
        synthesized = synthesized.copy()
        synthesized.insert(0, "record_id", np.arange(len(synthesized)))
    return coerce_to_submission(synthesized, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="参加者用ベースラインで C.csv を作る")
    parser.add_argument("--dist", required=True, help="participant_data ディレクトリ")
    parser.add_argument("--out", required=True, help="出力する C.csv")
    parser.add_argument("--method", choices=sorted(METHODS), default="copula_survival")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dist_dir = dist_paths(args.dist)
    b_df = read_b(dist_dir)
    cfg = load_cfg()
    rng = np.random.default_rng(args.seed)

    print(
        f"参加者用ベースラインを実行します "
        f"(method={args.method}, rows={len(b_df)}, seed={args.seed})..."
    )
    c_df = build_defense(b_df, cfg, rng, args.method)
    if len(c_df) != len(b_df):
        raise ValueError(f"行数が変わっています: C={len(c_df)} / B={len(b_df)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_c(c_df, out_path)


if __name__ == "__main__":
    main()
