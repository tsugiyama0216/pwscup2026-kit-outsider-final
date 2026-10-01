"""Read-only diagnostics of the actual main-round attack inputs.

Run using ../../.venv/Scripts/python.exe. This module does not import other v1
modules, read previous attack candidates, produce predictions, or access tokens.
Only aggregate diagnostics are written; raw rows and record IDs stay in memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes  # noqa: E402

ORIGINALS = ROOT.parent / "anonymization" / "team10" / "participant_data"
TARGETS = ROOT / "targets"
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
PUBLIC_CONTINUOUS = ["age", "SBP", "TG", "HDL", "ALT"]
CATEGORIES = ["age", "sex", "prefecture", "smoking"]
OUTCOME = ["time", "onset", "death"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    return list(frame[columns].itertuples(index=False, name=None))


def member_mask(query: pd.DataFrame, reference: pd.DataFrame,
                columns: list[str]) -> np.ndarray:
    pool = set(rows(reference, columns))
    return np.fromiter((row in pool for row in rows(query, columns)), bool, len(query))


def rate(mask) -> float:
    mask = np.asarray(mask)
    return float(mask.mean()) if len(mask) else 0.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def transform(frame: pd.DataFrame, columns: list[str], log_labs=False) -> np.ndarray:
    result = frame[columns].to_numpy(dtype=float, copy=True)
    if log_labs:
        for column in ("TG", "ALT"):
            if column in columns:
                result[:, columns.index(column)] = np.log1p(result[:, columns.index(column)])
    return result


def distance_summaries(k: int, c: pd.DataFrame, b: pd.DataFrame,
                       background: pd.DataFrame, negative: np.ndarray,
                       rare: np.ndarray) -> list[dict]:
    output = []
    spaces = [
        ("continuous7", CONTINUOUS, False),
        ("log_continuous7", CONTINUOUS, True),
        ("public_continuous5", PUBLIC_CONTINUOUS, False),
        ("continuous7_and_outcome", CONTINUOUS + OUTCOME, False),
    ]
    for name, columns, logarithm in spaces:
        ref = transform(background, columns, logarithm)
        scale = np.std(ref, axis=0)
        scale[scale == 0] = 1.0
        model = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1).fit(
            transform(c, columns, logarithm) / scale)
        d_b = model.kneighbors(transform(b, columns, logarithm) / scale,
                               return_distance=True)[0][:, 0]
        d_bg = model.kneighbors(ref / scale, return_distance=True)[0][:, 0]
        # Descriptive only: A_bg and private/shared rows do not have identical
        # distributions, so this threshold is not an official FPR estimate.
        threshold = float(np.quantile(d_bg, 0.01))
        groups = {
            "B_all": d_b,
            "B_known_private": d_b[negative],
            "B_not_proven_private": d_b[~negative],
            "B_true_rare": d_b[rare],
            "B_true_rare_known_private": d_b[rare & negative],
            "B_true_rare_not_proven_private": d_b[rare & ~negative],
            "A_bg": d_bg,
        }
        for group, distances in groups.items():
            entry = {"target_k": k, "space": name, "group": group,
                     "count": len(distances), "background_q01_threshold": threshold}
            for label, q in (("min", 0), ("q01", .01), ("q05", .05),
                             ("q10", .10), ("median", .5), ("q90", .9)):
                entry[label] = float(np.quantile(distances, q)) if len(distances) else None
            entry["count_at_or_below_background_q01"] = int(np.sum(distances <= threshold))
            output.append(entry)
    return output


def inspect_target(k: int, c: pd.DataFrame, b: pd.DataFrame,
                   background: pd.DataFrame, challenge: pd.DataFrame,
                   negative: np.ndarray, rare: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    full = list(c.columns)
    cevent = c[(c["onset"] == 1) | (c["death"] == 1)]
    bevent = b[(b["onset"] == 1) | (b["death"] == 1)]
    bgevent = background[(background["onset"] == 1) | (background["death"] == 1)]
    self_qi = member_mask(b, c, QI)
    self_full = member_mask(b, c, full)
    bg_full = member_mask(c, background, full)
    raw = pd.read_csv(TARGETS / f"C_{k}.csv", dtype=str, keep_default_na=False)
    last = c.iloc[-1]
    prefix = c.iloc[:-1]
    donor_columns = {col: bool(last[col] in set(prefix[col])) for col in full}
    last_duplicate = bool(member_mask(c.iloc[-1:], prefix, full)[0])
    matching_profile = []
    for col in CONTINUOUS:
        values = c[col].to_numpy(dtype=float)
        b_values = b[col].to_numpy(dtype=float)
        bg_values = background[col].to_numpy(dtype=float)
        decimal = raw[col].map(lambda s: len(s.split(".", 1)[1]) if "." in s and "e" not in s.lower() else 0)
        item = {
            "target_k": k, "column": col, "unique_values": int(c[col].nunique()),
            "C_values_in_B_rate": rate(np.isin(values, b_values)),
            "B_values_in_C_rate": rate(np.isin(b_values, values)),
            "C_values_in_A_bg_rate": rate(np.isin(values, bg_values)),
            "A_bg_values_in_C_rate": rate(np.isin(bg_values, values)),
            "integer_grid_rate": rate(np.abs(values - np.rint(values)) <= 1e-9),
            "tenth_grid_rate": rate(np.abs(values * 10 - np.rint(values * 10)) <= 1e-8),
            "hundredth_grid_rate": rate(np.abs(values * 100 - np.rint(values * 100)) <= 1e-7),
            "text_scientific_notation_rate": rate(raw[col].str.contains("e", case=False)),
            "text_fractional_digits_median": float(decimal.median()),
            "text_fractional_digits_max": int(decimal.max()),
            "row_index_spearman": float(spearmanr(np.arange(len(c)), values).statistic)
                if len(set(values)) > 1 else None,
        }
        matching_profile.append(item)
    continuous_labs = [item for item in matching_profile if item["column"] != "age"]
    readout = {
        "target_k": k,
        "rows": len(c), "columns": len(full),
        "duplicate_excess_rows": int(c.duplicated(full).sum()),
        "unique_public_QI_rows": int(len(c.drop_duplicates(QI))),
        "B_full_match_count": int(self_full.sum()),
        "B_public_QI_match_count": int(self_qi.sum()),
        "B_public_QI_known_private_match_count": int((self_qi & negative).sum()),
        "B_public_QI_rare_match_count": int((self_qi & rare).sum()),
        "C_full_match_A_bg_count": int(bg_full.sum()),
        "C_all_six_labs_match_A_bg_count": int(member_mask(c, background, CONTINUOUS[1:]).sum()),
        "C_public_QI_match_A_bg_count": int(member_mask(c, background, QI).sum()),
        "challenge_public_QI_match_C_count": int(member_mask(challenge, c, QI).sum()),
        "category_tuple_unique_count": len(set(rows(c, CATEGORIES))),
        "C_category_tuples_in_B_rate": rate(member_mask(c, b, CATEGORIES)),
        "C_category_tuples_in_A_bg_rate": rate(member_mask(c, background, CATEGORIES)),
        "onset_count": int(c["onset"].sum()),
        "death_count": int(c["death"].sum()),
        "event_count": len(cevent),
        "time_horizon_count": int((c["time"] == 10).sum()),
        "time_horizon_with_official_tolerance_count": int(np.isclose(c["time"], 10, atol=.01).sum()),
        "admin_censored_count": int(((c["onset"] == 0) & (c["death"] == 0)).sum()),
        "admin_censored_time_not_10_count": int(((c["onset"] == 0) & (c["death"] == 0) & (c["time"] != 10)).sum()),
        "admin_censored_outside_official_horizon_tolerance_count": int(((c["onset"] == 0) & (c["death"] == 0) & ~np.isclose(c["time"], 10, atol=.01)).sum()),
        "event_time_tuple_in_B_count": int(member_mask(cevent, bevent, OUTCOME).sum()),
        "event_time_tuple_in_A_bg_count": int(member_mask(cevent, bgevent, OUTCOME).sum()),
        "event_time_in_B_count": int(member_mask(cevent, bevent, ["time"]).sum()),
        "event_time_in_A_bg_count": int(member_mask(cevent, bgevent, ["time"]).sum()),
        "C_lab_values_in_B_mean_rate": float(np.mean([item["C_values_in_B_rate"] for item in continuous_labs])),
        "C_lab_values_in_A_bg_mean_rate": float(np.mean([item["C_values_in_A_bg_rate"] for item in continuous_labs])),
        "C_lab_hundredth_grid_mean_rate": float(np.mean([item["hundredth_grid_rate"] for item in continuous_labs])),
        "C_lab_tenth_grid_mean_rate": float(np.mean([item["tenth_grid_rate"] for item in continuous_labs])),
        "last_row_all_values_from_previous_rows": all(donor_columns.values()),
        "last_row_duplicate_of_previous_row": last_duplicate,
        "last_row_values_present_column_count": sum(donor_columns.values()),
        "last_row_background_full_match": bool(bg_full[-1]),
        "source_sha256": digest(TARGETS / f"C_{k}.csv"),
    }
    distance_rows = distance_summaries(k, c, b, background, negative, rare)
    for entry in distance_rows:
        if entry["space"] == "continuous7" and entry["group"] in ("B_known_private", "B_not_proven_private"):
            suffix = "known_private" if entry["group"] == "B_known_private" else "not_proven_private"
            readout[f"NN7_count_below_background_q01_{suffix}"] = entry["count_at_or_below_background_q01"]
    return readout, matching_profile, distance_rows


def strategy(row: dict) -> tuple[str, str]:
    observations, hypotheses = [], []
    if row["B_public_QI_match_count"]:
        observations.append(f"自Bの7属性一致{row['B_public_QI_match_count']}件")
        hypotheses.append("公開属性の完全一致を最優先で検証")
    if row["challenge_public_QI_match_C_count"]:
        observations.append(f"challengeの7属性一致{row['challenge_public_QI_match_C_count']}件")
        hypotheses.append("AIAの一致候補のtimeを確認")
    if row["C_full_match_A_bg_count"]:
        observations.append(f"背景の全13列一致{row['C_full_match_A_bg_count']}行")
        hypotheses.append("背景由来の行を本人の証拠から区別")
    elif row["C_all_six_labs_match_A_bg_count"]:
        observations.append(f"背景の検査6列一致{row['C_all_six_labs_match_A_bg_count']}行")
        hypotheses.append("背景ドナーの検査値再利用を区別")
    if row["event_time_tuple_in_A_bg_count"]:
        observations.append(f"背景event転帰一致{row['event_time_tuple_in_A_bg_count']}/{row['event_count']}行")
        hypotheses.append("背景の転帰再利用を考慮")
    if row["event_time_tuple_in_B_count"]:
        observations.append(f"自Bのevent転帰一致{row['event_time_tuple_in_B_count']}行")
        hypotheses.append("time一致単独では本人保持と断定しない")
    if row["C_lab_hundredth_grid_mean_rate"] > .8:
        observations.append("検査値が主に0.01格子")
        hypotheses.append("列ごとの丸め幅を使う近似照合")
    if row["C_lab_values_in_B_mean_rate"] > .01:
        observations.append(f"検査値の自B一致平均{row['C_lab_values_in_B_mean_rate']:.1%}")
        hypotheses.append("列の部分保持・列別入替えを調査")
    if row["duplicate_excess_rows"]:
        observations.append(f"重複超過{row['duplicate_excess_rows']}行")
        hypotheses.append("複製を独立した証拠として数えない")
    if row["last_row_all_values_from_previous_rows"]:
        observations.append("末尾全列が既存値")
        hypotheses.append("補完候補行あり/なしを比較")
    if not observations:
        observations.append("全行/7属性一致なし、粗丸め主体ではない")
    if not hypotheses:
        hypotheses.append("連続値の近傍・密度比・複数の列の組で比較")
    observations.append(f"近傍域の未確定{row['NN7_count_below_background_q01_not_proven_private']}/886・既知非共有{row['NN7_count_below_background_q01_known_private']}/163人")
    return "、".join(observations), "／".join(dict.fromkeys(hypotheses))


def render_report(result: dict, output: Path):
    teams = result["teams"]
    lines = [
        "# 本戦v1・22チームの加工診断", "",
        f"作成日時：{result['created_at_jst']}。自己10を除く22チーム、配布原本と自チーム本戦原本だけを使用。", "",
        "## 先に確認できたこと", "",
        f"- 公式マニフェストの{result['input_integrity']['verified_files']}件が一致。入力原本は変更していない。",
        f"- 自己challengeと自Bの7属性一致は{result['self_challenge']['matched_B_rows']}人で全対応が一意。内訳は発症{result['self_challenge']['onset_count']}人、真の希少{result['self_challenge']['rare_count']}人、希少かつ発症{result['self_challenge']['rare_onset_count']}人。",
        "- この163人はルールの標的数と完全に整合する。配布仕様と同一属性衝突がないことに基づき、他チームへのMIAでは非共有の既知陰性として扱う。残り886人が共有者と確定したわけではない。",
        "- 完全一致は数値をround_tripで読み、IDを除いた指定列の値で比較した。個人の値・ID・対応表は出力していない。", "",
        "## 22チームの一覧", "",
        "観測は実測した事実。右列は次に検証する攻撃仮説であり、方式や精度の確定ではない。", "",
        "全22相手で自Bの全13列一致・公開7属性一致は0。challengeの公開7属性とCの完全一致も全22相手で0。検査6列の0.01格子率はいずれも平均0.45%未満で、粗い丸めが主体の提出は見つからなかった。", "",
        "一覧の近傍域は、7連続属性によるCへの距離が背景データの下位1%点以下の範囲。未確定は残り886人を指し、陽性人数ではない。", "",
        "|相手|観測|攻撃に使う判断|", "|---:|---|---|",
    ]
    for row in teams:
        observed, hypothesis = strategy(row)
        lines.append(f"|{row['target_k']}|{observed}|{hypothesis}|")
    lines += ["", "## 指標の意味と限界", "",
        "- `C_values_in_B_rate`はCの各列の値が自Bの同じ列にある割合。年齢など共通値の多い列では高くても所属の証拠にならない。列ごとに高くても同じ行の対応は保証しない。",
        "- eventはonset=1またはdeath=1の行。time=10の大量一致を別扱いにし、eventのtimeと(onset,death)の組の一致を数えた。timeだけ保存して他列を入れ替えた可能性もある。",
        "- time=10の厳密一致と、公式打切り採点のnp.isclose(time,10,atol=0.01)による一致を別集計した。厳密に10でないことだけでは打切りの不整合といえない。",
        "- 原文字列の異なる10進表記が同じdoubleを表す場合がある。例えばC8のevent転帰は、公式と同じ数値読込で186/186行が背景の転帰と一致する。文字列の書式差を数値の変更と取り違えない。",
        "- 7属性一致はage/SBP/TG/HDL/ALT/sex/prefectureの一致。B全13列一致とは別指標。微小ノイズ・丸めでは完全一致が消えるため、一致ゼロは安全性や完全合成の証明にならない。",
        "- 最近傍距離はA_bgの標準偏差で標準化したユークリッド距離。7連続属性、TG/ALT対数化7属性、公開5連続属性、7属性+転帰の4通り。カテゴリーはこの距離には含めず別集計した。",
        "- 最近傍の背景1%点は診断用で、公式FPRを示さない。県構成や自己非共有群と共有群の分布が異なるため、既知陰性163人への適合だけで良否を決めない。",
        "- 真の希少はB_selfのis_rareを使った。Cの外れ度による検出希少とは区別する。",
        "- 末尾の値が各列で既存値なら運営のホットデック補完条件に適合するが、その条件単独では一般に一意識別できない。全対象との比較結果はJSONに残した。",
        "- 検査値の格子・桁・行順は加工方式の仮説づくりに使う。提出バイト列の保存は運営メールで確認できるが、行順が元Bと対応する保証はない。",
        "- 公開属性一致、event転帰一致、値保持の件数は漏洩の可能性の調査指標であり、公式攻撃得点ではない。", "",
        "## v1へ渡す方針", "",
        "1. 自己challengeから特定した非共有163人を、全相手に対する既知陰性として保護する。",
        "2. 今回は公開属性の完全一致が全相手で0なので、多視点の近似距離・列ごとの尺度・背景比を基本にする。",
        "3. C1の背景検査値再利用、C3の背景行、C8/C28の背景転帰再利用を、本人の情報保持と混同しない。time単独の一致を強い陽性証拠にしない。",
        "4. 列の一部保持を想定した照合・対数距離・背景密度比を比較する。今回、粗い丸めの強い証拠はなく、丸め逆算を主力にしない。",
        "5. C0の重複行、C3の補完候補行と背景由来行について、含める/弱める感度を別比較する。独立した新患者として重複計上しない。",
        "6. 各手法と人数・共有制約の効果を分離して検証し、一度に強制しない。", "",
        "## 出力", "",
        "- `team_metrics.csv`：22チームの全体集計。",
        "- `column_metrics.csv`：7連続列ごとの値保持、丸め、書式、行順の集計。",
        "- `distance_metrics.csv`：最近傍距離を自B全体・既知非共有・未確定・希少・背景に分けた集計。",
        "- `diagnosis.json`：入力整合性、定義、全集計、来歴。", "",
        "根拠：本戦ルールブック§3–6、2026-09-27運営メール、固定公式キット。過去の攻撃候補やv88は使用していない。",
    ]
    (output / "DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "M" / "local-output" / "v1" / "diagnosis")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "M" / "local-output" / "v1").resolve()):
        raise ValueError("Diagnostic output must remain under M/local-output/v1")
    integrity = verify_hashes()
    b = read_csv(ORIGINALS / "B.csv")
    background = read_csv(ORIGINALS / "A_bg.csv")
    b_self = read_csv(ORIGINALS / "B_self.csv")
    if not b_self["record_id"].equals(b["record_id"]):
        raise ValueError("B_self row keys differ from B")
    shared_columns = [column for column in b.columns if column in b_self]
    if not b[shared_columns].equals(b_self[shared_columns]):
        raise ValueError("B_self source values differ from B")
    rare = b_self["is_rare"].to_numpy(dtype=bool)
    self_challenge = read_csv(TARGETS / "aia_challenge_10.csv")
    if b.duplicated(QI).any() or self_challenge.duplicated(QI).any():
        raise ValueError("Self challenge QI matching is not unique")
    negative = member_mask(b, self_challenge, QI)
    own = b.loc[negative]
    self_summary = {
        "matched_B_rows": int(negative.sum()),
        "challenge_rows": len(self_challenge),
        "unmatched_challenge_rows": int((~member_mask(self_challenge, b, QI)).sum()),
        "onset_count": int(own["onset"].sum()),
        "rare_count": int(rare[negative].sum()),
        "rare_onset_count": int((rare & negative & (b["onset"].to_numpy() == 1)).sum()),
        "all_matching_QI_unique": True,
    }
    assert (self_summary["matched_B_rows"], self_summary["onset_count"],
            self_summary["rare_count"], self_summary["rare_onset_count"]) == (163, 16, 13, 4)
    targets = sorted(int(x) for x in json.loads((TARGETS / "mia_columns.json").read_text(
        encoding="utf-8-sig"))["target_columns"] if int(x) != 10)
    assert len(targets) == 22
    teams, columns, distances = [], [], []
    for k in targets:
        c = read_csv(TARGETS / f"C_{k}.csv")
        challenge = read_csv(TARGETS / f"aia_challenge_{k}.csv")
        row, cols, dists = inspect_target(k, c, b, background, challenge, negative, rare)
        teams.append(row)
        columns.extend(cols)
        distances.extend(dists)
    result = clean({
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "input_integrity": integrity,
        "source_inputs": {name: digest(ORIGINALS / name) for name in ("B.csv", "B_self.csv", "A_bg.csv", "schema.json")},
        "code_sha256": digest(Path(__file__)),
        "scope": "actual main-round inputs; no previous attack candidates; no tokens; no submissions",
        "self_challenge": self_summary,
        "teams": teams, "columns": columns, "distances": distances,
        "definitions": {
            "exact": "numeric round-trip CSV parsing; no record_id in value matching",
            "C_lab_mean": "unweighted mean over BMI,SBP,TG,HDL,ALT,FPG",
            "event": "onset=1 or death=1; administrative censoring summarized separately",
            "known_private": "B rows uniquely QI-matched to own challenge, agreeing with official target quotas",
            "NN": "Euclidean distance divided by A_bg column standard deviation; descriptive, not official score",
        },
    })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(teams).to_csv(output / "team_metrics.csv", index=False)
    pd.DataFrame(columns).to_csv(output / "column_metrics.csv", index=False)
    pd.DataFrame(distances).to_csv(output / "distance_metrics.csv", index=False)
    (output / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    render_report(result, output)
    print(json.dumps({"status": "complete", "targets": len(teams),
                      "verified_source_files": integrity["verified_files"],
                      "known_private_count": int(negative.sum()),
                      "full_background_matches": {str(row["target_k"]): row["C_full_match_A_bg_count"]
                                                   for row in teams if row["C_full_match_A_bg_count"]},
                      "hotdeck_compatible_last_rows": [row["target_k"] for row in teams
                                                        if row["last_row_all_values_from_previous_rows"]],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
