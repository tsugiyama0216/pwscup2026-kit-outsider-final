"""Compare the unchanged four explorations using the existing main mock scorer.

Generate four matched worlds and estimate each world's floor once. Attacks run
in separate public-input-only processes; scoring uses Sugiyama v2 unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONUTF8"] = "1"

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
REPO = MAIN.parent
sys.path.insert(0, str(MAIN / "mock_test"))
import sugiyama_v2
import attack_workspace as workspace
from sugiyama_v2.generate import build_world
from sugiyama_v2.io import (csv, export_world, load_world, output_path, save_json,
                           sha, snapshot, source_manifest)
from sugiyama_v2.runner import per_target_diagnostics
from sugiyama_v2.scoring import estimate_floors, score_world
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

METHODS = {
    "baseline": ("公式参照 DOMIAS + 1-NN", HERE / "main_mock/official_baseline.py"),
    "explore_1_distance": ("① QI最近傍距離 + 1-NN", HERE / "explore_1_distance/plugin.py"),
    "explore_2_density": ("② DOMIAS + 近傍時間区間", HERE / "explore_2_density/plugin.py"),
    "explore_3_gen_lra": ("③ 固定カーネルGen-LRA + 近傍時間区間", HERE / "explore_3_gen_lra/plugin.py"),
    "explore_4_predictor": ("④ 予測残差差 + time回帰", HERE / "explore_4_predictor/plugin.py"),
}
WORLDS = [{"id": f"{profile}_holdout_{seed}", "profile": profile,
           "split": "holdout", "seed": seed, "floor_seed": seed + 1000003}
          for seed in (2026100801, 2026100802) for profile in ("mixed", "stress")]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = [Path(__file__)]
    for folder in (*HERE.glob("explore_*"), HERE / "main_mock"):
        if folder.is_dir():
            paths += [p for p in folder.rglob("*") if p.is_file()
                      and p.suffix in (".py", ".json") and "__pycache__" not in p.parts]
    return {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(paths)}


def inputs_hashes():
    root = MAIN / "anonymization/team10/participant_data"
    return {name: sha(root / name) for name in ("B_self.csv", "A_bg.csv", "schema.json")}


def load_predictions(destination):
    return tuple(pd.read_csv(destination / name, float_precision="round_trip")
                 for name in ("F_mia.csv", "F_aia.csv"))


def prediction_checks(root):
    predictions = {key: load_predictions(root / key) for key in METHODS}
    aia = lambda key: predictions[key][1].set_index(["target_k", "challenge_row_id"]).sort_index()
    checks = {
        "distance_aia_equals_official": aia("baseline").equals(aia("explore_1_distance")),
        "density_mia_bytes_equal_official": sha(root / "baseline/F_mia.csv") == sha(root / "explore_2_density/F_mia.csv"),
        "density_genlra_aia_equal": aia("explore_2_density").equals(aia("explore_3_gen_lra")),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def run_predictions(case, config, key, destination):
    started = perf_counter()
    destination.mkdir(parents=True)
    command = [sys.executable, "-I", "-B", str(HERE / "main_mock/worker.py"),
               "--public", str(case / "public"), "--config", str(config),
               "--plugin", str(METHODS[key][1]), "--output", str(destination)]
    with (destination / "attack.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=case / "public", check=True, stdout=log,
                       stderr=subprocess.STDOUT, env=os.environ.copy())
    return perf_counter() - started


def evaluate_world(definition, output, config, cfg, workers):
    root = output / definition["id"]
    root.mkdir()
    case = root / "case"
    print("Generate " + definition["id"], flush=True)
    with threadpool_limits(limits=1):
        export_world(build_world(definition["seed"], definition["profile"], definition["split"]), case)
    public, private, manifest, original = load_world(case)
    truth = root / "scoring_truth"
    shutil.copytree(private, truth)
    print("Estimate shared floors: 22 targets x 1000 draws", flush=True)
    started = perf_counter()
    with threadpool_limits(limits=1):
        floors, floor_metadata = estimate_floors(public, truth, cfg, definition["floor_seed"], 1000)
    csv(truth / "aia_floor.csv", floors)
    save_json(truth / "aia_floor_metadata.json", floor_metadata)
    truth_hashes = {p.name: sha(p) for p in sorted(truth.iterdir()) if p.is_file()}
    floor_seconds = perf_counter() - started
    print(f"Shared floors ready ({floor_seconds:.1f}s)", flush=True)
    timings = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(run_predictions, case, config, key, root / key): key for key in METHODS}
        for future in as_completed(jobs):
            key = jobs[future]
            timings[key] = future.result()
            print("Predictions complete: " + key, flush=True)
    checks = prediction_checks(root)
    rows, details = [], []
    for key in METHODS:
        destination = root / key
        f_mia, f_aia = load_predictions(destination)
        with threadpool_limits(limits=1):
            result = score_world(f_mia, f_aia, public, truth, cfg)
            detail = per_target_diagnostics(f_mia, f_aia, public, truth, cfg, manifest["metadata"], result)
        csv(destination / "by_target.csv", detail)
        csv(destination / "by_style.csv", detail.groupby("style", sort=True).agg(
            targets=("target_k", "count"), mia_power=("mia_power", "mean"),
            aia_power=("aia_power", "mean"), attack_power=("attack_power", "mean"),
            aia_before_floor=("aia_before_floor", "mean")).reset_index())
        for metric in ("mia_power", "aia_power", "attack_power"):
            if not np.isclose(result[metric], detail[metric].mean(), rtol=0, atol=1e-12):
                raise AssertionError("Per-target aggregation mismatch: " + metric)
        result.update({"completed": True, "created_utc": utc_now(), "method": key,
            "world": definition, "plugin_sha256": sha(METHODS[key][1]),
            "case_manifest_sha256": original["manifest"],
            "shared_scoring_truth": "../scoring_truth", "scoring_truth_sha256": truth_hashes,
            "prediction_seconds": timings[key], "shared_floor_seconds": floor_seconds,
            "output_sha256": {name: sha(destination / name) for name in ("F_mia.csv", "F_aia.csv")}})
        save_json(destination / "result.json", result)
        rows.append({"world": definition["id"], "method": key, "label": METHODS[key][0],
            "profile": definition["profile"], "split": definition["split"], "seed": definition["seed"],
            **{metric: result[metric] for metric in ("mia_power", "aia_power", "attack_power")},
            "aia_before_floor": result["pre_floor"]["aia_power"],
            "mean_floor": float(floors.aia_floor.mean()), "prediction_seconds": timings[key]})
        details.append(detail.assign(world=definition["id"], method=key))
        print(key + " score: " + str(result["leaderboard"]), flush=True)
    if snapshot(case)[1] != original:
        raise AssertionError("Saved case changed")
    if {p.name: sha(p) for p in sorted(truth.iterdir()) if p.is_file()} != truth_hashes:
        raise AssertionError("Shared scoring truth changed")
    save_json(root / "verification.json", {"case_unchanged": True, "shared_truth_unchanged": True,
        "controlled_comparison_checks": checks, "case_manifest_sha256": original["manifest"],
        "scoring_truth_sha256": truth_hashes, "floor_repetitions": 1000})
    return rows, details


def write_report(output, rows, details):
    frame = pd.DataFrame(rows)
    baseline = frame[frame.method == "baseline"].set_index("world")
    for metric in ("mia_power", "aia_power", "attack_power"):
        frame["delta_" + metric] = frame[metric] - frame.world.map(baseline[metric])
    csv(output / "by_world.csv", frame)
    csv(output / "by_target.csv", pd.concat(details, ignore_index=True))
    metrics = ("mia_power", "aia_power", "attack_power", "aia_before_floor", "delta_attack_power")
    summary = frame.groupby("method", sort=False)[list(metrics)].mean().reset_index()
    summary["attack_power_min"] = summary.method.map(frame.groupby("method").attack_power.min())
    summary["attack_power_max"] = summary.method.map(frame.groupby("method").attack_power.max())
    csv(output / "comparison.csv", summary)
    save_json(output / "comparison.json", {"world_count": len(WORLDS), "target_count_per_world": 22,
        "floor_repetitions": 1000, "worlds": WORLDS, "summary": summary.to_dict("records"),
        "by_world": frame.to_dict("records"), "official_score_prediction": False})
    lines = ["# 本戦疑似環境での4方式比較", "", "設定は予備戦評価時から変更せず、4世界を同じ重みで平均しました。AIAは模擬φ控除後です。", "",
        "| 手法 | MIA力 | AIA力（φ控除後） | 合計 | 公式参照との差 |", "|---|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {METHODS[row['method']][0]} | {row['mia_power']:.6f} | {row['aia_power']:.6f} | {row['attack_power']:.6f} | {row['delta_attack_power']:+.6f} |")
    lines += ["", "## 世界別の合計", "", "| 条件 | 公式参照 | ① | ② | ③ | ④ |", "|---|---:|---:|---:|---:|---:|"]
    for definition in WORLDS:
        selected = frame[frame.world == definition["id"]].set_index("method")
        lines.append("| " + definition["id"] + " | " + " | ".join(f"{selected.loc[key, 'attack_power']:.6f}" for key in METHODS) + " |")
    lines += ["", "## 条件と限界", "",
        "- Sugiyama v2の既存生成器・本戦公式採点を変更せず使用。自己10を除く22標的、MIA 1,049人、AIA 7,172行／世界。",
        "- 2seed × mixed/stress。生成元のsplitは全世界holdoutに固定し、手法や設定の選択は行っていません。",
        "- 各世界の模擬φは背景から1,049行を非復元抽出する公式参照AIAの1,000回平均。全5方式で同じ保存φを共用。",
        "- 攻撃の別プロセスには公開表・公式設定・公開定数team10のみを渡し、正解・加工名・seed・φは渡していません。OSレベルのファイル隔離ではありません。",
        "- ①のAIA＝公式参照、②のMIA＝公式参照、②③のAIA一致を全世界で確認。③は固定カーネルGen-LRA変種。",
        "- holdoutは既存の生成元分割名で、既使用のデータ由来です。seedは同じ有限の生成元を再利用し、独立した実人口サンプルではありません。",
        "- 本戦の非公開正解・φ・相手加工過程は再現できません。公式得点の予測や、予備戦との差を純粋な改善幅として扱うことはできません。", "",
        "[固定条件・ソースハッシュ](protocol.json) / [集約CSV](comparison.csv) / [世界別CSV](by_world.csv) / [標的別CSV](by_target.csv)"]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>本戦疑似環境：4方式比較</title><style>body{font:16px system-ui;max-width:1200px;margin:32px;color:#172338}table{border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}p{line-height:1.8}</style><h1>本戦疑似環境：4方式比較</h1><p>2seed × mixed/stress、同じholdout生成元、22標的／世界。AIAは共通の模擬φ（1,000反復）控除後。4世界等重み平均。</p>'
    display = summary[["method", "mia_power", "aia_power", "attack_power", "delta_attack_power"]].copy()
    display["method"] = display.method.map(lambda key: METHODS[key][0])
    display.columns = ["手法", "MIA力", "AIA力（φ控除後）", "合計", "公式参照との差"]
    page += display.to_html(index=False, float_format=lambda x: f"{x:.6f}", border=0)
    page += '<p>③は固定カーネルGen-LRA変種。疑似データでの結果であり、公式点の予測値ではありません。</p><p><a href="SUMMARY.md">世界別結果・条件</a> · <a href="by_target.csv">標的別CSV</a> · <a href="protocol.json">固定条件</a></p></html>'
    (output / "report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="main_mock_fixed_20261008_v1")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name must be a simple directory name")
    output = output_path(MAIN / "mock_test/local-output/exploration" / args.name)
    official = workspace.verify_hashes()
    _, _, _, cfg, versions = workspace.load_runtime()
    sources, evaluators, inputs = source_hashes(), source_manifest(), inputs_hashes()
    output.mkdir(parents=True)
    save_json(output / "protocol.json", {"created_utc": utc_now(), "worlds": WORLDS,
        "methods": list(METHODS), "floor_repetitions": 1000, "source_sha256": sources,
        "evaluator_sha256": evaluators, "donor_sha256": inputs, "official_sources": official,
        "runtime_versions": versions, "cfg": cfg, "workers": args.workers,
        "configs": {key: json.loads((HERE / key / "config.json").read_text(encoding="utf-8"))
                    for key in METHODS if key != "baseline"},
        "selection": "Same attack settings as prelim; worlds fixed before main mock scores; no tuning",
        "aggregation": "equal weight across four worlds, official equal weight across 22 targets each",
        "official_score_prediction": False})
    config = output / "public_config.json"
    save_json(config, cfg)
    rows, details = [], []
    try:
        for definition in WORLDS:
            world_rows, world_details = evaluate_world(definition, output, config, cfg, args.workers)
            rows.extend(world_rows)
            details.extend(world_details)
        if source_hashes() != sources or source_manifest() != evaluators or inputs_hashes() != inputs:
            raise AssertionError("Source or donor data changed during evaluation")
        if workspace.verify_hashes() != official:
            raise AssertionError("Official source verification changed")
        write_report(output, rows, details)
        save_json(output / "verification.json", {"completed": True, "completed_utc": utc_now(),
            "source_hashes_unchanged": True, "donor_hashes_unchanged": True,
            "all_world_controlled_checks_passed": True, "world_count": 4, "method_world_count": 20})
    except BaseException as error:
        save_json(output / "failure.json", {"error": repr(error), "utc": utc_now()})
        raise
    print("Report: " + str(output / "report.html"), flush=True)


if __name__ == "__main__":
    main()
