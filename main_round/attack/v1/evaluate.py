"""Evaluate the independent v1 on fixed shadow cases; never load actual masks.

Run from main_round/attack: python -m v1.evaluate --stage dev --output ...
The holdout stage calls only model.predict(..., variant=selected).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attack_workspace as workspace


SEEDS = {
    "dev": (2026093000, 2026093001, 2026093002),
    "holdout": (2026101000, 2026101001, 2026101002),
}
METHOD_REFERENCE = "official"
NULL_REPETITIONS = 100
HERE = Path(__file__).resolve().parent


def json_value(value):
    """Convert aggregate results to strict JSON without exposing record values."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def object_hash(value):
    text = json.dumps(json_value(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_hash(frame):
    # The serialization stays in memory. No rows, IDs, or individual values are saved.
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_hash(values):
    import numpy as np
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def dependency_hashes():
    """Freeze v1 sources/settings and the general evaluator's direct dependencies."""
    paths = {
        workspace.ROOT / "attack_workspace.py",
        workspace.ROOT / "local_eval" / "data.py",
        workspace.ROOT / "local_eval" / "scoring.py",
        workspace.ROOT / "source_manifest.json",
    }
    suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
    paths.update(p for p in HERE.rglob("*")
                 if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    return {p.relative_to(workspace.ROOT).as_posix(): workspace.sha256(p)
            for p in sorted(paths)}


def public_inputs(release, case, cfg):
    from local_eval.data import FEATURES, QI
    return {
        "c": release[FEATURES].copy(deep=True),
        "background": case.background[FEATURES].copy(deep=True),
        "candidates": case.candidates[["record_id"] + FEATURES].copy(deep=True),
        "challenge": case.challenge[["challenge_row_id"] + QI].copy(deep=True),
        "cfg": copy.deepcopy(cfg),
    }


def official_predict(modules, **inputs):
    confidence = modules["pwscup2026.attack.mia"].overlap_mia(
        inputs["c"], inputs["background"], inputs["candidates"], inputs["cfg"])
    _, time_hat = modules["pwscup2026.attack.aia"].aia_predict(
        inputs["c"], inputs["challenge"], inputs["background"], inputs["cfg"])
    return {"mia": confidence, "aia": time_hat}


def aggregate(frame, keys):
    """Summaries are paired within seed/style before any averaging."""
    import pandas as pd
    rows = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        delta = group["delta_vs_official"]
        rows.append({
            **dict(zip(keys, key)),
            "conditions": len(group),
            "mia_mean": group["mia"].mean(),
            "mia_all_mean": group["mia_all"].mean(),
            "mia_rare_mean": group["mia_rare"].mean(),
            "aia_no_phi_mean": group["aia_no_phi"].mean(),
            "aia_all_mean": group["aia_all"].mean(),
            "aia_rare_mean": group["aia_rare"].mean(),
            "combined_no_phi_mean": group["combined_no_phi"].mean(),
            "combined_no_phi_min": group["combined_no_phi"].min(),
            "combined_no_phi_max": group["combined_no_phi"].max(),
            "paired_delta_mean": delta.mean(),
            "paired_delta_min": delta.min(),
            "paired_wins": int((delta > 1e-12).sum()),
            "paired_ties": int((delta.abs() <= 1e-12).sum()),
            "paired_losses": int((delta < -1e-12).sum()),
            "aia_member_success_mean": group["aia_member_success"].mean(),
            "aia_control_success_mean": group["aia_control_success"].mean(),
        })
    return pd.DataFrame(rows)


def write_outputs(output, rows, run):
    import pandas as pd
    from local_eval.data import PRIMARY_STYLES
    frame = pd.DataFrame(rows)
    reference = frame.loc[frame.method.eq(METHOD_REFERENCE),
                          ["stage", "seed", "style", "combined_no_phi", "mia", "aia_no_phi"]]
    reference = reference.rename(columns={"combined_no_phi": "official_combined_no_phi",
                                          "mia": "official_mia", "aia_no_phi": "official_aia_no_phi"})
    paired = frame.merge(reference, on=["stage", "seed", "style"], how="left",
                         validate="many_to_one")
    paired["delta_vs_official"] = paired.combined_no_phi - paired.official_combined_no_phi
    paired["mia_delta_vs_official"] = paired.mia - paired.official_mia
    paired["aia_delta_vs_official"] = paired.aia_no_phi - paired.official_aia_no_phi
    paired["primary_condition"] = paired["style"].isin(PRIMARY_STYLES)
    primary = paired.loc[paired.primary_condition]
    tables = {
        "results.csv": frame,
        "paired_comparison.csv": paired,
        "summary.csv": aggregate(primary, ["method"]),
        "by_style.csv": aggregate(paired, ["style", "method"]),
        "by_seed.csv": aggregate(primary, ["seed", "method"]),
        "by_rare_shared.csv": aggregate(primary, ["mia_rare_members", "method"]),
        "sanity.csv": aggregate(paired.loc[~paired.primary_condition], ["style", "method"]),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    save_json(output / "summary.json", tables["summary.csv"].to_dict("records"))
    save_json(output / "case_summary.json", paired.to_dict("records"))
    save_json(output / "run.json", run)
    summary = tables["summary.csv"].sort_values("combined_no_phi_mean", ascending=False)
    lines = [
        "# 自分の攻撃 v1：模擬データでの比較", "",
        f"区分：{run['stage']}。選択名：`{run['selected']}`。",
        "公式参照と独立実装を、同じ人物抽出・加工条件で比較しました。",
        "**本戦22標的の得点予測ではありません。主値はMIA＋AIAのφ控除前代理値です。**", "",
        "| 方法 | 主条件数 | MIA | AIA・φ未控除 | 合計 | 公式との差 | 勝／同／負 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        method = str(row["method"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {method} | {row['conditions']} | {row['mia_mean']:.6f} | "
                     f"{row['aia_no_phi_mean']:.6f} | {row['combined_no_phi_mean']:.6f} | "
                     f"{row['paired_delta_mean']:+.6f} | {row['paired_wins']}／"
                     f"{row['paired_ties']}／{row['paired_losses']} |")
    lines += [
        "", "- 主比較は5加工方式×3抽出の15条件です。copy・independentの6条件は動作確認として別集計しました。",
        "- 3抽出は希少共有0・1・2人を各1回含みます。`by_seed.csv` と `by_rare_shared.csv` は同じ3抽出の別表示で、独立した追加試験ではありません。",
        "- `by_style.csv` は全7方式、`paired_comparison.csv` は同じseed・方式での差です。得意な方式だけを選んだ平均ではありません。",
        "- 相手ごとのφは不明なので控除していません。自チームφの転用、MIAの匿名性用K変換や置換控除は行っていません。",
        "- 置換100回の平均・95%点は診断値です。正式な偶然補正、p値、信頼区間として使いません。",
        "- 人物分割seedは2026092701でdev/holdoutは非交差ですが、この分割のholdoutは既存研究で採点済みです。新seedは既使用人物群からの新しい抽出で、完全に未知の人物による検証ではありません。",
        "- 元人物はA_bg、背景は自Bで、本戦とは役割が逆です。希少は公開検出器による代理区分で、真のis_rareではありません。",
        "- 7加工方式は模擬方式です。本戦22標的の加工法や各方式の割合、県ごとの偏りを再現していません。",
        "- 自己challengeの163行、実データのマスクや個別校正は使用していません。その実戦用処理の効果は、この比較には含みません。",
        "- 予測関数へ渡すのはC・背景・候補・公開QI・公開設定のみです。所属、time正解、真希少、元人物番号、抽出seedを渡しません。個人値・ID・個別予測は出力していません。",
        "- devの予測時間は全variantsを返す1呼出しの合計です。同じ値を各variant行へ記録しており、各variantの専有時間ではありません。",
        "", ("候補の調整はdevで行い、選択と設定を固定してからholdoutを実行します。今回の実行はholdoutを採点していません。"
               if run["stage"] == "dev" else
               "holdout開始前に選択名・モデルと設定のハッシュを `freeze.json` へ保存し、選択した1候補だけを公式参照と比較しました。この結果を見て再調整した場合は最終確認の役割を終えます。"),
        "", f"所要時間：{run['elapsed_seconds']:.2f}秒。CPU計算スレッド数：1。原本94件と評価コードの実行前後の一致を確認済みです。",
        "", "[主比較](summary.csv) / [方式別](by_style.csv) / [抽出別](by_seed.csv) / [希少共有人数別](by_rare_shared.csv) / [動作確認](sanity.csv) / [実行記録](run.json)",
        "", "CodaBench提出、提出ZIP作成、GitHub送信は行っていません。", "",
    ]
    (output / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selected", default="ensemble_nearest")
    args = parser.parse_args(argv)
    if not args.selected.strip():
        parser.error("selected must be a nonempty candidate name")
    started = perf_counter()
    verified = workspace.verify_hashes()
    np, pd, modules, cfg, versions = workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    from local_eval.data import (FEATURES, QI, PARTITION_SEED, PRIMARY_STYLES,
                                 STYLES, build_case, make_release, partition_source)
    from local_eval.scoring import null_diagnostics, score_arrays, validate_predictions

    source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
    background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
    schema = json.loads((workspace.ORIGINALS / "schema.json").read_text(encoding="utf-8"))
    if source.duplicated(QI).any() or background.duplicated(QI).any():
        raise ValueError("Duplicate QI would invalidate the person-disjoint protocol")
    if len(source[QI].merge(background[QI], on=QI, how="inner")):
        raise ValueError("Evaluation people overlap the independent background")
    partitions = partition_source(source, schema)
    dependencies = dependency_hashes()
    model = importlib.import_module("v1.model")
    function = getattr(model, "predict_variants" if args.stage == "dev" else "predict", None)
    if not callable(function):
        raise ValueError("v1.model lacks the prediction API required by this stage")
    output = (args.output or workspace.OUTPUT / "v1" /
              (args.stage + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    if not output.is_relative_to(workspace.OUTPUT.resolve()):
        parser.error("output must remain inside main_round/attack/local-output")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty; previous evaluations are preserved")
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "format": "independent-v1-shadow-evaluation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "selected": args.selected, "seeds": list(SEEDS[args.stage]),
        "styles": list(STYLES), "primary_styles": list(PRIMARY_STYLES),
        "partition_seed": PARTITION_SEED,
        "partition_sizes": {key: len(indices) for key, indices in partitions.items()},
        "partition_sha256": {key: hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
                             for key, indices in partitions.items()},
        "person_partition_previously_used": True,
        "fresh_sampling_is_independent_person_holdout": False,
        "input_sha256": {name: workspace.sha256(workspace.ORIGINALS / name)
                         for name in ("A_bg.csv", "B.csv", "schema.json")},
        "code_and_settings_sha256": dependencies,
        "official_config_sha256": object_hash(cfg), "schema_sha256": object_hash(schema),
        "dependency_versions": versions, "source_checks_before": verified,
        "null_repetitions": NULL_REPETITIONS, "compute_threads": 1,
        "official_score_estimated": False, "aia_phi": None,
        "actual_private_masks_used": False, "individual_values_saved": False,
        "official_submission": False,
        "scoring": {"low_fpr": cfg["scoring"]["low_fpr"], "tau_time": cfg["aia"]["tau_time"],
                    "rare_se_z": cfg["scoring"]["aia_rare_se_z"],
                    "aggregate": "equal-weight mean MIA + mean AIA before phi"},
        "cases": [], "methods": [], "output": str(output),
    }
    # Written before any holdout case is generated or any holdout prediction is made.
    save_json(output / ("freeze.json" if args.stage == "holdout" else "protocol.json"), run)
    rows, expected_variants = [], None
    try:
        with threadpool_limits(limits=1):
            for seed in SEEDS[args.stage]:
                case = build_case(source, background, schema, args.stage, seed)
                case_record = {**case.metadata, "public_input_sha256": {
                    "background": frame_hash(case.background), "candidates": frame_hash(case.candidates),
                    "challenge": frame_hash(case.challenge)}, "styles": {}}
                for style in STYLES:
                    release = make_release(case.target, case.background, style, seed, schema)
                    release_hash = frame_hash(release)
                    case_record["styles"][style] = {"release_sha256": release_hash}
                    t = perf_counter()
                    official = official_predict(modules, **public_inputs(release, case, cfg))
                    official_seconds = perf_counter() - t
                    t = perf_counter()
                    if args.stage == "dev":
                        predictions = function(**public_inputs(release, case, cfg))
                        if not isinstance(predictions, dict) or not predictions:
                            raise ValueError("predict_variants must return a nonempty dict of variants")
                        if any(not isinstance(name, str) or not name.strip()
                               for name in predictions):
                            raise ValueError("Variant names must be nonempty strings")
                        if METHOD_REFERENCE in predictions:
                            claimed = validate_predictions(predictions[METHOD_REFERENCE], case)
                            actual = validate_predictions(official, case)
                            if any(not np.allclose(left, right, rtol=0, atol=1e-12)
                                   for left, right in zip(claimed, actual)):
                                raise ValueError("The model's official variant differs from the pinned official reference")
                            predictions = {name: value for name, value in predictions.items()
                                           if name != METHOD_REFERENCE}
                        if not predictions:
                            raise ValueError("predict_variants must include a candidate in addition to official")
                        names = tuple(sorted(predictions))
                        if expected_variants is None:
                            expected_variants = names
                        elif names != expected_variants:
                            raise ValueError("Variant names changed between conditions")
                    else:
                        predictions = {args.selected: function(**public_inputs(release, case, cfg),
                                                              variant=args.selected)}
                    batch_seconds = perf_counter() - t
                    predictions = {METHOD_REFERENCE: official, **predictions}
                    for method, prediction in predictions.items():
                        mia, aia = validate_predictions(prediction, case)
                        row = {
                            "stage": args.stage, "seed": seed, "style": style, "method": method,
                            "mia_rare_members": case.metadata["mia_rare_members"],
                            "case_role_sha256": case.metadata["role_assignment_sha256"],
                            "release_sha256": release_hash,
                            "mia_predictions_sha256": array_hash(mia), "aia_predictions_sha256": array_hash(aia),
                            "predict_call_seconds": official_seconds if method == METHOD_REFERENCE else batch_seconds,
                            "predict_timing_scope": ("one_method" if method == METHOD_REFERENCE or args.stage == "holdout"
                                                     else "all_variants_batch"),
                            **score_arrays(mia, aia, case, cfg),
                            **null_diagnostics(mia, aia, case, cfg, NULL_REPETITIONS),
                        }
                        rows.append(row)
                    save_json(output / "progress.json", {"stage": args.stage, "completed_conditions": len(rows) // len(predictions),
                                                           "total_conditions": len(SEEDS[args.stage]) * len(STYLES),
                                                           "elapsed_seconds": perf_counter() - started})
                    print(f"Completed {args.stage} seed={seed} style={style}", flush=True)
                run["cases"].append(case_record)
        run["source_checks_after"] = workspace.verify_hashes()
        if dependency_hashes() != dependencies:
            raise ValueError("Evaluation code or settings changed during execution")
        run["methods"] = sorted({row["method"] for row in rows})
        run["elapsed_seconds"] = perf_counter() - started
        run["conditions"] = len(SEEDS[args.stage]) * len(STYLES)
        run["rows"] = len(rows)
        run["completed"] = True
        write_outputs(output, rows, run)
        print(f"Evaluation complete: {output / 'EVALUATION.md'}", flush=True)
    except Exception as error:
        save_json(output / "failure.json", {"stage": args.stage, "error_type": type(error).__name__,
                                            "completed_result_rows": len(rows),
                                            "elapsed_seconds": perf_counter() - started,
                                            "completed": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
