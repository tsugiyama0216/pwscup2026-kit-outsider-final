"""Evaluate stored shadow cases without regenerating people, releases, or truth.

python -m local_eval.evaluate_case --case saved-case-or-suite --plugin candidate.py --output results
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter

import numpy as np
import pandas as pd

import attack_workspace as workspace
from .__main__ import predict_safely, save
from .data import FEATURES, PRIMARY_STYLES, QI, QUOTAS, STYLES, ShadowCase


@dataclass
class SavedCase:
    root: Path
    case: ShadowCase
    releases: dict
    manifest: dict
    integrity: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _contained(root, name):
    _require(isinstance(name, str) and name and "\\" not in name,
             "Manifest paths must be relative POSIX paths")
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name,
             f"Unsafe manifest path: {name}")
    path = (root / name).resolve()
    _require(path.is_relative_to(root) and path != root, f"Manifest path escapes case: {name}")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ensure_empty_output(path):
    _require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
             "Output directory must be empty")


def resolve_output(path=None):
    if path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = workspace.MOCK_OUTPUT / "evaluations" / f"saved-{timestamp}"
    out = Path(path).resolve()
    _require(out.is_relative_to(workspace.MOCK_OUTPUT.resolve()),
             "Output must stay under main_round/mock_test/local-output")
    _ensure_empty_output(out)
    return out


def _case_snapshot(root, manifest):
    manifest_path = _contained(root, "manifest.json")
    files = manifest.get("files_sha256")
    _require(isinstance(files, dict) and files, "Missing case file hashes")
    actual = {}
    for name, expected in files.items():
        _require(name != "manifest.json" and _digest(expected), f"Invalid file hash: {name}")
        path = _contained(root, name)
        _require(path.is_file(), f"Missing case file: {name}")
        actual[name] = workspace.sha256(path)
        _require(actual[name] == expected, f"Case SHA256 mismatch: {name}")
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _require(present == set(files) | {"manifest.json"}, "Case has unlisted or missing files")
    return {"manifest_sha256": workspace.sha256(manifest_path), "files_sha256": actual}


def _table(root, name, columns, rows):
    frame = pd.read_csv(_contained(root, name), float_precision="round_trip")
    _require(list(frame.columns) == columns, f"Unexpected columns: {name}")
    _require(len(frame) == rows and not frame.isna().any().any(), f"Invalid rows or missing values: {name}")
    return frame


def _numeric(frame, columns, name):
    values = frame[columns].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"Nonfinite numeric values: {name}")
    return values


def _integers(series, name):
    values = series.to_numpy(dtype=float)
    _require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(),
             f"Expected finite integer values: {name}")
    _require(((values >= 0) & (values < 2**53)).all(), f"Invalid integer range: {name}")
    return values.astype(np.int64)


def _binary(series, name):
    # CSV bool columns are normally parsed as bool; reject textual or fractional labels.
    _require(pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series),
             f"Expected binary labels: {name}")
    values = series.to_numpy(dtype=float)
    _require(np.isfinite(values).all() and np.isin(values, [0, 1]).all(), f"Invalid binary labels: {name}")
    return values.astype(bool)


def load_case(path):
    """Verify and reload a v1 export. No source-data or generation calls occur here."""
    root = Path(path).resolve()
    if root.is_file() and root.name == "manifest.json":
        root = root.parent
    _require(root.is_dir(), f"Case directory missing: {root}")
    manifest = json.loads(_contained(root, "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("format") == "pwscup2026-local-shadow-v1", "Unsupported case format")
    split, seed, styles = manifest.get("split"), manifest.get("seed"), manifest.get("styles")
    _require(split in ("dev", "holdout") and type(seed) is int and 0 <= seed < 2**32,
             "Invalid split or seed")
    _require(isinstance(styles, list) and styles and all(isinstance(s, str) and s in STYLES for s in styles)
             and len(set(styles)) == len(styles), "Invalid case styles")
    before = _case_snapshot(root, manifest)
    required = {"public/B_reference.csv", "public/mia_candidates.csv", "public/aia_challenge.csv",
                "public/schema.json", "private/mia_truth.csv", "private/aia_truth.csv"}
    required.update(f"public/C_{style}.csv" for style in styles)
    _require(required <= set(before["files_sha256"]), "Missing required public/private case files")
    schema = json.loads(_contained(root, "public/schema.json").read_text(encoding="utf-8"))
    _require(isinstance(schema, dict), "Invalid saved schema")
    metadata = manifest.get("metadata")
    _require(isinstance(metadata, dict) and metadata.get("seed") == seed and metadata.get("split") == split,
             "Case metadata disagrees with manifest")
    fixed = {"target_rows": 1049, "mia_candidates": 1049, "mia_members": 26,
             "mia_rare_candidates": 24, "mia_rare_members": seed % 3,
             "aia_members": 163, "aia_controls": 163, "aia_rare_per_side": 13,
             "aia_onset_per_side": 16, "aia_rare_onset_per_side": 4}
    _require(all(metadata.get(key) == value for key, value in fixed.items()), "Unexpected v1 case counts")
    candidates = _table(root, "public/mia_candidates.csv", ["record_id"] + FEATURES, 1049)
    challenge = _table(root, "public/aia_challenge.csv", ["challenge_row_id"] + QI, 326)
    background = _table(root, "public/B_reference.csv", FEATURES, 1049)
    mia = _table(root, "private/mia_truth.csv", ["record_id", "member", "detected_rare"], 1049)
    aia = _table(root, "private/aia_truth.csv",
                 ["challenge_row_id", "member", "detected_rare", "time", "onset", "pair_id"], 326)
    for public, truth, column in ((candidates, mia, "record_id"), (challenge, aia, "challenge_row_id")):
        ids = _integers(public[column], column)
        truth_ids = _integers(truth[column], f"truth {column}")
        _require(len(np.unique(ids)) == len(ids) and np.array_equal(ids, truth_ids),
                 f"Truth ID order mismatch or duplicate: {column}")
    for name, frame in (("candidates", candidates), ("challenge", challenge), ("background", background)):
        _numeric(frame, [c for c in frame.columns if c != "prefecture"], name)
    mia_truth = _binary(mia.member, "MIA member").astype(int)
    mia_rare = _binary(mia.detected_rare, "MIA rare")
    aia_member = _binary(aia.member, "AIA member")
    aia_rare = _binary(aia.detected_rare, "AIA rare")
    aia_onset = _binary(aia.onset, "AIA onset").astype(int)
    aia_time = _numeric(aia, ["time"], "AIA time").ravel()
    aia_pair = _integers(aia.pair_id, "AIA pair_id")
    _require((aia_time >= 0).all(), "Negative AIA truth time")
    _require(mia_truth.sum() == 26 and mia_rare.sum() == 24
             and mia_rare[mia_truth == 1].sum() == seed % 3, "MIA labels disagree with protocol")
    _require(set(aia_pair) == set(range(163)), "AIA pair IDs must cover 0..162")
    for side in (aia_member, ~aia_member):
        for (rare, onset), count in QUOTAS.items():
            _require(int((side & (aia_rare == rare) & (aia_onset == onset)).sum()) == count,
                     "AIA label quotas disagree with protocol")
    for pair in range(163):
        positions = np.flatnonzero(aia_pair == pair)
        _require(len(positions) == 2 and aia_member[positions].sum() == 1, "Invalid AIA matched pair")
        a, b = positions
        _require(aia_rare[a] == aia_rare[b] and aia_onset[a] == aia_onset[b]
                 and challenge.sex.iloc[a] == challenge.sex.iloc[b]
                 and challenge.prefecture.iloc[a] == challenge.prefecture.iloc[b]
                 and (challenge.age.iloc[a] - 40) // 5 == (challenge.age.iloc[b] - 40) // 5,
                 "AIA matching conditions disagree within pair")
    releases = {}
    for style in styles:
        release = _table(root, f"public/C_{style}.csv", FEATURES, 1049)
        _numeric(release, [c for c in FEATURES if c != "prefecture"], f"C_{style}")
        releases[style] = release
    case = ShadowCase(pd.DataFrame(columns=FEATURES), candidates, challenge, background,
                      mia_truth, mia_rare, aia_member, aia_rare, aia_time, aia_onset, aia_pair,
                      metadata.copy(), {})
    _require(_case_snapshot(root, manifest) == before, "Case changed while loading")
    return SavedCase(root, case, releases, manifest, before)


def load_cases(paths):
    """Expand saved suites and deduplicate canonical case paths, preserving order."""
    cases, suites, seen = [], [], set()
    for supplied in paths:
        path = Path(supplied).resolve()
        if path.is_file() and path.name in ("manifest.json", "suite.json"):
            path = path.parent
        suite_path = _contained(path, "suite.json")
        entries = [(path, None, None)]
        if suite_path.is_file():
            digest = workspace.sha256(suite_path)
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            _require(suite.get("format") == "pwscup2026-local-shadow-suite-v1", "Unsupported suite format")
            records, seeds = suite.get("cases"), suite.get("seeds")
            _require(isinstance(records, list) and records and isinstance(seeds, list)
                     and len(seeds) == len(records) and suite.get("split") in ("dev", "holdout"),
                     "Invalid suite cases/seeds/split")
            entries = []
            for record, seed in zip(records, seeds):
                _require(isinstance(record, dict) and _digest(record.get("manifest_sha256")), "Invalid suite case hash")
                child = _contained(path, record.get("path"))
                _require(child.is_dir(), "Suite case is not a directory")
                entries.append((child, record["manifest_sha256"], (suite["split"], seed)))
            _require(workspace.sha256(suite_path) == digest, "Suite changed while loading")
            suites.append({"path": str(suite_path), "sha256": digest})
        for child, expected_hash, expected_case in entries:
            # Even repeated cases must match every suite's pinned hash and metadata.
            loaded = load_case(child)
            if expected_hash is not None:
                _require(loaded.integrity["manifest_sha256"] == expected_hash, "Suite case manifest SHA256 mismatch")
                _require((loaded.manifest["split"], loaded.manifest["seed"]) == expected_case,
                         "Suite case split/seed mismatch")
            if loaded.root not in seen:
                cases.append(loaded)
                seen.add(loaded.root)
    _require(cases, "At least one saved case is required")
    return cases, suites


def summarize_saved(frame):
    """Pair by case identity, never by seed alone; omit sanity styles from the main mean."""
    baseline = frame.loc[frame.method.eq("official"), ["case_id", "style", "combined_no_phi"]]
    baseline = baseline.rename(columns={"combined_no_phi": "official_combined"})
    paired = frame.merge(baseline, on=["case_id", "style"], how="left", validate="many_to_one")
    _require(paired.official_combined.notna().all(), "Missing official result for a case/style")
    paired["delta_vs_official"] = paired.combined_no_phi - paired.official_combined
    paired["primary"] = paired["style"].isin(PRIMARY_STYLES)
    columns = ["method", "cases", "mia_mean", "aia_no_phi_mean", "combined_no_phi_mean",
               "combined_no_phi_min", "combined_no_phi_max", "paired_delta_vs_official_mean",
               "paired_delta_vs_official_min", "paired_wins", "paired_ties", "paired_losses"]
    rows = []
    for name, group in paired.loc[paired.primary].groupby("method", sort=False):
        delta = group.delta_vs_official
        rows.append([name, len(group), group.mia.mean(), group.aia_no_phi.mean(), group.combined_no_phi.mean(),
                     group.combined_no_phi.min(), group.combined_no_phi.max(), delta.mean(), delta.min(),
                     int((delta > 1e-12).sum()), int((delta.abs() <= 1e-12).sum()), int((delta < -1e-12).sum())])
    return pd.DataFrame(rows, columns=columns).sort_values("combined_no_phi_mean", ascending=False), paired


def _report(out, frame, summary, metadata):
    units = frame[["case_id", "style"]].drop_duplicates()
    primary = units["style"].isin(PRIMARY_STYLES).sum()
    lines = ["# 保存済み模擬問題の攻撃比較", "",
             f"保存済み{frame.case_id.nunique()}ケース・計{len(units)}個のケース×加工方式を、同じ入力で比較しました。データの再生成はしていません。", "",
             "**公式得点の予測ではありません。主指標はMIA＋AIAのφ控除前代理値（0〜2、高いほど攻撃が強い）です。**", "",
             f"主集計は{primary}条件です。copy・independentは動作確認用として主集計から除外します。", ""]
    if summary.empty:
        lines += ["今回は動作確認用の方式のみです。主比較の平均は算出せず、下記の方式別結果を参照してください。", ""]
    else:
        lines += ["| 手法 | 条件数 | MIA平均 | AIA平均・φ未控除 | 合計平均 | 最小〜最大 | 公式との差 | 勝/同/負 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary.itertuples():
            lines.append(f"| {row.method} | {row.cases} | {row.mia_mean:.4f} | {row.aia_no_phi_mean:.4f} | {row.combined_no_phi_mean:.4f} | {row.combined_no_phi_min:.4f}〜{row.combined_no_phi_max:.4f} | {row.paired_delta_vs_official_mean:+.4f} | {row.paired_wins}/{row.paired_ties}/{row.paired_losses} |")
        lines.append("")
        lines += ["ケースごとの主集計です。同じ分割でも抽出を変えた場合の変動を確認できます。", "",
                  "| ケース | 分割 | seed | 手法 | 主条件数 | 合計平均 | 公式との差 |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for (case_id, split, seed, method), group in frame.loc[frame.primary].groupby(
                ["case_id", "split", "seed", "method"], sort=False):
            lines.append(f"| {case_id} | {split} | {seed} | {method} | {len(group)} | {group.combined_no_phi.mean():.4f} | {group.delta_vs_official.mean():+.4f} |")
        lines.append("")
    lines += ["| 加工方式 | 手法 | 条件数 | MIA平均 | AIA平均・φ未控除 | 合計平均 |", "|---|---|---:|---:|---:|---:|"]
    for (style, method), group in frame.groupby(["style", "method"], sort=False):
        lines.append(f"| {style} | {method} | {len(group)} | {group.mia.mean():.4f} | {group.aia_no_phi.mean():.4f} | {group.combined_no_phi.mean():.4f} |")
    lines += ["", "集計は保存ケース×加工方式を等重みで平均します。加工方式の実戦割合や信頼区間を表しません。公式参照との差は同一ケース・同一方式で比較します。", "",
              "- 同じ人物分割内のseed違いは同じ人物群からの再抽出であり、独立した試験集団ではありません。",
              "- devとholdoutは人物を分けていますが、holdoutは過去の研究ですでに採点されています。未使用の最終検証用とは扱えません。",
              "- 公開A_bgを模擬人物、自Bを背景に使用します。希少区分は公開検出器による代理です。他チームの未知の加工法や私的コホートを再現していません。",
              "- 公式score_mia/score_aiaを使用し、未知のφを補わず、採点式は変更しません。",
              f"- ラベル入替診断は各回答{metadata['null_repetitions']}回。診断値を公式補正として得点から差し引きません。",
              "- 予測関数へ渡すのは公開入力のみです。候補Pythonの任意ファイルアクセスを隔離するsandboxではありません。",
              "- 候補ファイル本体のハッシュを保存します。外部モデル・追加依存コードは候補の作者が別途固定・記録してください。", "",
              "入力ファイルとマニフェストは実行前後に照合済みです。公式提出は行いません。", "",
              "[ケース×方式別結果](results.csv) / [主集計](summary.csv) / [主条件での同条件比較](paired_comparison.csv) / [ケース別](by_case.csv) / [方式別](by_style.csv) / [実行記録](run.json)"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--plugin", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path,
                        help="Empty output directory; default: local-output/evaluations/saved-<UTC timestamp>")
    parser.add_argument("--null-reps", type=int, default=100)
    args = parser.parse_args(argv)
    _require(0 <= args.null_reps <= 10000, "null-reps must be between 0 and 10000")
    start = perf_counter()
    out = resolve_output(args.output)
    cases, suites = load_cases(args.case)
    for loaded in cases:
        _require(not out.is_relative_to(loaded.root) and not loaded.root.is_relative_to(out), "Output must be separate from saved cases")
    source_before = workspace.verify_hashes()
    _, _, _, cfg, versions = workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    from .methods import official
    from .scoring import null_diagnostics, score_arrays
    selected, plugins = {"official": official}, {}
    for index, supplied in enumerate(args.plugin, 1):
        path = supplied.resolve()
        _require(path.is_file(), f"Missing candidate: {path}")
        digest = workspace.sha256(path)
        name = f"plugin_{index}_{re.sub(r'[^A-Za-z0-9_]+', '_', path.stem)}"
        spec = importlib.util.spec_from_file_location(f"saved_case_{name}_{digest[:12]}", path)
        _require(spec is not None and spec.loader is not None, f"Cannot load candidate: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _require(callable(getattr(module, "predict", None)), f"Candidate must export predict(): {path}")
        selected[name] = module.predict
        plugins[name] = {"path": str(path), "sha256": digest}
    rows = []
    with threadpool_limits(limits=1):
        for index, loaded in enumerate(cases, 1):
            for style, release in loaded.releases.items():
                for name, function in selected.items():
                    started = perf_counter()
                    mia, aia = predict_safely(function, release, loaded.case, cfg)
                    seconds = perf_counter() - started
                    rows.append({"case_id": f"case_{index:03d}", "case_path": str(loaded.root),
                                 "split": loaded.manifest["split"], "seed": loaded.manifest["seed"],
                                 "style": style, "method": name, **score_arrays(mia, aia, loaded.case, cfg),
                                 **null_diagnostics(mia, aia, loaded.case, cfg, args.null_reps),
                                 "prediction_seconds": seconds})
                print(f"Completed case_{index:03d} style={style}", flush=True)
    case_records = []
    for index, loaded in enumerate(cases, 1):
        after = _case_snapshot(loaded.root, loaded.manifest)
        _require(after == loaded.integrity, "Saved case changed during evaluation")
        case_records.append({"case_id": f"case_{index:03d}", "path": str(loaded.root),
                             "metadata": loaded.case.metadata, "styles": list(loaded.releases),
                             "integrity_before": loaded.integrity, "integrity_after": after})
    for suite in suites:
        _require(workspace.sha256(Path(suite["path"])) == suite["sha256"], "Suite changed during evaluation")
    for plugin in plugins.values():
        _require(workspace.sha256(Path(plugin["path"])) == plugin["sha256"], "Candidate changed during evaluation")
    source_after = workspace.verify_hashes()
    summary, frame = summarize_saved(pd.DataFrame(rows))
    metadata = {"format": "pwscup2026-saved-case-evaluation-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "cases": case_records, "suites": suites,
                "plugins": plugins, "methods": list(selected), "null_repetitions": args.null_reps,
                "primary_styles": list(PRIMARY_STYLES), "config": cfg, "dependency_versions": versions,
                "official_runtime": {"package": str(workspace.PACKAGE),
                                     "source_manifest_sha256": workspace.sha256(workspace.ROOT / "source_manifest.json")},
                "source_checks_before": source_before, "source_checks_after": source_after,
                "evaluator_sha256": {p.name: workspace.sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "aggregate": "equal-weight saved case/style mean; copy and independent excluded from primary",
                "holdout_previously_used": True, "holdout_scored_in_this_run": any(c.manifest["split"] == "holdout" for c in cases),
                "official_score_estimated": False, "aia_phi": None, "official_submission": False,
                "compute_threads": 1, "elapsed_seconds": perf_counter() - start, "output": str(out)}
    # Prediction code and other runs may have created files since the initial check.
    _ensure_empty_output(out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    frame.loc[frame.primary].to_csv(out / "paired_comparison.csv", index=False)
    measures = ["mia", "aia_no_phi", "combined_no_phi", "delta_vs_official"]
    for filename, groups in (("by_case.csv", ["case_id", "split", "seed", "method", "primary"]),
                             ("by_style.csv", ["style", "method", "primary"])):
        frame.groupby(groups, sort=False)[measures].agg(["count", "mean", "min", "max"]).to_csv(out / filename)
    save(out / "run.json", metadata)
    _report(out, frame, summary, metadata)
    print(f"Saved-case comparison complete: {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
