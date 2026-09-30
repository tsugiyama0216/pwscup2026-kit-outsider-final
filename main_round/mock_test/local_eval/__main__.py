"""CLI: python -m local_eval --split dev [--plugin candidate.py]."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import attack_workspace as workspace

SEEDS = {"dev": [20260927, 20260928, 20260929], "holdout": [20261001, 20261002, 20261003]}


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def predict_safely(function, release, case, cfg):
    from .data import FEATURES, QI
    from .scoring import validate_predictions
    predictions = function(c=release[FEATURES].copy(deep=True),
        background=case.background[FEATURES].copy(deep=True),
        candidates=case.candidates[["record_id"] + FEATURES].copy(deep=True),
        challenge=case.challenge[["challenge_row_id"] + QI].copy(deep=True), cfg=copy.deepcopy(cfg))
    return validate_predictions(predictions, case)


def summarize(frame):
    from .data import PRIMARY_STYLES
    import pandas as pd
    primary = frame.loc[frame["style"].isin(PRIMARY_STYLES)].copy()
    baseline = primary.loc[primary.method.eq("official"), ["seed", "style", "combined_no_phi"]]
    baseline = baseline.rename(columns={"combined_no_phi": "official_combined"})
    paired = primary.merge(baseline, on=["seed", "style"], how="left", validate="many_to_one")
    paired["delta_vs_official"] = paired.combined_no_phi - paired.official_combined
    rows = []
    for name, group in paired.groupby("method", sort=False):
        delta = group.delta_vs_official
        rows.append({"method": name, "cases": len(group), "mia_mean": group.mia.mean(),
            "aia_no_phi_mean": group.aia_no_phi.mean(), "combined_no_phi_mean": group.combined_no_phi.mean(),
            "combined_no_phi_min": group.combined_no_phi.min(), "combined_no_phi_max": group.combined_no_phi.max(),
            "paired_delta_vs_official_mean": delta.mean(), "paired_delta_vs_official_min": delta.min(),
            "paired_wins": int((delta > 1e-12).sum()), "paired_ties": int((delta.abs() <= 1e-12).sum()),
            "paired_losses": int((delta < -1e-12).sum())})
    return pd.DataFrame(rows).sort_values("combined_no_phi_mean", ascending=False), paired


def write_report(out, summary, frame, metadata):
    labels = {"official": "公式参照手法", "multiview": "多視点の比較用手法",
              "constant": "定数回答の対照", "background_only": "背景だけの対照"}
    lines = ["# 本戦攻撃・ローカル代理評価", "",
        f"実行区分：`{metadata['split']}`。既知ラベルを持つ模擬データでの手法比較です。", "",
        "**これは公式得点の予測ではありません。主指標はMIA＋AIAの補正値φを引く前の代理値（0〜2、高いほど攻撃が強い）です。**", "",
        "## 比較結果", "",
        "5加工方式（丸め・弱いノイズ・強いノイズ・復元抽出・コピュラ合成）×3回の固定抽出を同じ条件で比較しました。コピー・独立生成は動作確認用で、主平均から除外しています。", "",
        "| 手法 | MIA平均 | AIA平均・φ未控除 | 合計平均 | 同じ条件の公式参照との差 | 勝/同/負 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {labels.get(row['method'], row['method'])} | {row['mia_mean']:.4f} | {row['aia_no_phi_mean']:.4f} | {row['combined_no_phi_mean']:.4f} | {row['paired_delta_vs_official_mean']:+.4f} | {row['paired_wins']}/{row['paired_ties']}/{row['paired_losses']} |")
    lines += ["", "上表の差は同じseed・加工方式での差を平均した値です。方式の等重み平均であり、本戦22チームの構成比を推定した値ではありません。抽出間の幅は信頼区間ではありません。", "",
              "## 加工方式による違い", "", "| 方式 | 手法 | MIA平均 | AIA平均・φ未控除 | 合計平均 |", "|---|---|---:|---:|---:|"]
    for (style, method), group in frame.loc[~frame.method.isin(["constant", "background_only"])].groupby(["style", "method"]):
        lines.append(f"| {style} | {labels.get(method, method)} | {group.mia.mean():.4f} | {group.aia_no_phi.mean():.4f} | {group.combined_no_phi.mean():.4f} |")
    lines += ["",
              "## 動作確認と偶然の水準", "", "| 手法 | コピーCのMIA平均 | 独立CのMIA平均 | 独立CのAIA平均 |", "|---|---:|---:|---:|"]
    for method in frame.method.unique():
        copy_rows = frame.loc[frame.method.eq(method) & frame["style"].eq("copy")]
        independent = frame.loc[frame.method.eq(method) & frame["style"].eq("independent")]
        lines.append(f"| {labels.get(method, method)} | {copy_rows.mia.mean():.4f} | {independent.mia.mean():.4f} | {independent.aia_no_phi.mean():.4f} |")
    lines += ["", "定数MIAの基準値は0.01です。独立Cも、少人数の希少層・max処理・有限標本の影響で点数が正になることがあります。",
        (f"各回答につき{metadata['null_repetitions']}回、MIAは希少層内の所属ラベルを入れ替え、AIAは対応する2人のmember/controlを入れ替えました。その平均・95%点を results.csv に記録しています。これは診断値で、公式の補正値として差し引いていません。"
         if metadata["null_repetitions"] else "この実行ではラベル入替による偶然の水準の診断を省略しています。"), ""]
    if metadata["null_repetitions"]:
        lines += ["| 手法 | MIA・ラベル入替時95%点の平均 | AIA・ラベル入替時95%点の平均 |", "|---|---:|---:|"]
        for method, group in frame.groupby("method", sort=False):
            lines.append(f"| {labels.get(method, method)} | {group.mia_null_p95.mean():.4f} | {group.aia_null_p95.mean():.4f} |")
        lines += ["", "各ケースの95%点を平均した診断値で、平均攻撃力の95%点や信頼区間ではありません。ケースごとの実測値と対照を併せて確認します。", ""]
    lines += [
        "## 再現条件と限界", "",
        "- 公式のscore_mia/score_aiaをそのまま使用。低FPR=1%、AIA許容誤差±0.5年、希少SE補正z=1.64。",
        "- 模擬コホート1,049人、MIA候補1,049人中member26人・希少候補24人。希少memberは0/1/2人の3条件。",
        "- AIAはmember163人・control163人。各側の希少13・発症16・希少発症4を固定し、県・性・5歳帯・発症・希少区分を対応させています。",
        "- AIA標的はMIAで共有する26人と非重複。AIA対照は模擬コホートとMIA候補のどちらにも含めません。",
        "- A_bgを人物単位で調整用/最終確認用に分割。自Bは背景専用に使うため、本戦の役割と入れ替わっています。希少区分は公開検出器による代理で、真の希少ラベルではありません。",
        "- 正解や元の行番号を予測関数に渡しません。これは信頼できるローカル候補コード用のAPIであり、悪意あるコードから正解を隠す実行環境ではありません。",
        "- 加工方式は単純な模擬方式です。他チームの未知の生成手法・県偏りを再現したものではありません。",
        "- 同じ分割内の3回は同じ人物群からの再抽出で独立試験ではありません。人物が非交差の最終確認用で、候補固定後に確認してください。",
        "- 本戦の予測CSVだけを渡して公式得点を推定する機能はありません。候補アルゴリズムを模擬データへ適用して比較します。",
        "- この版の予測APIは1標的ずつです。複数標的の同時最適化や自己challengeを使う校正は、別の評価構成が必要です。", "",
        ("**最終確認用の採点は、この実行では行っていません。**" if metadata["split"] == "dev" else "**最終確認用を採点しました。結果を見て調整した場合、この分割は以後、未使用の検証用ではなくなります。**"), "",
        f"全体所要時間：{metadata['elapsed_seconds']:.2f}秒。原本94件の実行前後の整合性確認に成功。公式提出なし。", "",
        "[ケース別結果](results.csv) / [主比較](summary.csv) / [同条件での差](paired_comparison.csv) / [実行記録](run.json)", "",
        "使い方と候補の追加方法は [評価器README](../../../local_eval/README.md) を参照してください。"]
    # Resolve the documentation link for arbitrary output nesting.
    import os
    relative = Path(os.path.relpath(Path(__file__).parent / "README.md", out)).as_posix()
    lines[-1] = f"使い方と候補の追加方法は [評価器README]({relative}) を参照してください。"
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Known-label local attack comparison; not official scoring")
    parser.add_argument("--split", choices=SEEDS, default="dev")
    parser.add_argument("--methods", default="official,multiview,constant,background_only")
    parser.add_argument("--plugin", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--null-reps", type=int, default=100)
    args = parser.parse_args(argv)
    if args.null_reps < 0 or args.null_reps > 10000:
        parser.error("null-reps must be between 0 and 10000")
    start = perf_counter()
    hashes = workspace.verify_hashes()
    np, pd, _, cfg, versions = workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    from .data import FEATURES, QI, STYLES, PRIMARY_STYLES, partition_source, build_case, make_release
    from .methods import BUILTINS
    from .scoring import score_arrays, null_diagnostics
    selected = {}
    for name in args.methods.split(","):
        if name not in BUILTINS:
            parser.error(f"Unknown method: {name}")
        selected[name] = BUILTINS[name]
    if "official" not in selected:
        parser.error("Include official to retain a fixed paired reference")
    plugins = {}
    for index, path in enumerate(args.plugin):
        path = path.resolve()
        name = "plugin_" + path.stem
        if name in selected:
            parser.error(f"Duplicate candidate name: {name}")
        spec = importlib.util.spec_from_file_location(f"local_candidate_{index}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "predict", None)):
            parser.error(f"{path.name} must export predict")
        selected[name] = mod.predict
        plugins[name] = {"path": str(path), "sha256": workspace.sha256(path)}
    out = (args.output or workspace.MOCK_OUTPUT / "evaluations" /
           (args.split + "-" + datetime.now().strftime("%Y%m%dT%H%M%S"))).resolve()
    if not out.is_relative_to(workspace.MOCK_OUTPUT.resolve()):
        parser.error("output must be inside main_round/mock_test/local-output")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty; keep previous evaluations")
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
    background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
    schema = json.loads((workspace.ORIGINALS / "schema.json").read_text(encoding="utf-8"))
    if source.duplicated(QI).any() or background.duplicated(QI).any():
        raise ValueError("Source QI duplicates would invalidate person-disjoint splitting")
    if len(source[QI].merge(background[QI], on=QI, how="inner")):
        raise ValueError("Background overlaps the evaluation source")
    partitions = partition_source(source, schema)
    rows, cases = [], []
    with threadpool_limits(limits=1):
        for seed in SEEDS[args.split]:
            case = build_case(source, background, schema, args.split, seed)
            cases.append(case.metadata)
            for style in STYLES:
                release = make_release(case.target, case.background, style, seed, schema)
                for name, function in selected.items():
                    t = perf_counter()
                    mia, aia = predict_safely(function, release, case, cfg)
                    prediction_seconds = perf_counter() - t
                    row = {"split": args.split, "seed": seed, "style": style, "method": name,
                        **score_arrays(mia, aia, case, cfg),
                        **null_diagnostics(mia, aia, case, cfg, args.null_reps),
                        "prediction_seconds": prediction_seconds}
                    rows.append(row)
                print(f"Completed {args.split} seed={seed} style={style}", flush=True)
    frame = pd.DataFrame(rows)
    summary, paired = summarize(frame)
    frame.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_comparison.csv", index=False)
    frame.groupby(["method", "style"])[["mia", "aia_no_phi", "combined_no_phi"]].mean().to_csv(out / "by_style.csv")
    after = workspace.verify_hashes()
    module_dir = Path(__file__).parent
    metadata = {"format": "local-attack-shadow-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "seeds": SEEDS[args.split], "methods": list(selected), "styles": list(STYLES),
        "primary_styles": list(PRIMARY_STYLES), "null_repetitions": args.null_reps,
        "scoring": {"official_commit": "a490490e11a0ec460a5df9fdefb17c72ad459c1e",
            "low_fpr": cfg["scoring"]["low_fpr"], "tau_time": cfg["aia"]["tau_time"],
            "rare_se_z": cfg["scoring"]["aia_rare_se_z"], "aia_phi": None,
            "official_score_estimated": False, "aggregate": "equal-weight mean MIA + mean AIA before phi"},
        "partition_sizes": {key: len(indices) for key, indices in partitions.items()},
        "partition_sha256": {key: hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest() for key, indices in partitions.items()},
        "cases": cases, "plugins": plugins, "source_checks_before": hashes, "source_checks_after": after,
        "input_sha256": {name: workspace.sha256(workspace.ORIGINALS / name) for name in ("B.csv", "A_bg.csv", "schema.json")},
        "evaluator_sha256": {p.name: workspace.sha256(p) for p in sorted(module_dir.glob("*.py"))},
        "dependency_versions": versions, "compute_threads": 1, "output": str(out),
        "elapsed_seconds": perf_counter() - start, "official_submission": False,
        "holdout_scored_in_this_run": args.split == "holdout",
        "person_partition_previously_used": True,
        "fresh_sampling_is_independent_person_holdout": False}
    save(out / "run.json", metadata)
    write_report(out, summary, frame, metadata)
    print(f"Local proxy evaluation complete: {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
