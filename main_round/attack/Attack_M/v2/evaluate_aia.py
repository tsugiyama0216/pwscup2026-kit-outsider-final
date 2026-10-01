"""Predeclared regional AIA development and frozen-selection holdout.

No actual target C, actual challenge mask, old attack prediction, token or team
number is used. Original source people are repartitioned by the existing fixed
person partition, then sampled using new seeds. Only aggregate results/hashes
are saved. Holdout is a separate explicit invocation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import attack_workspace as workspace
from local_eval.data import build_case, make_release, partition_source, FEATURES, QI, STYLES, PRIMARY_STYLES, PARTITION_SEED
from pwscup2026.scoring.attack_score import score_aia
from M.v2 import aia

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "local-output" / "v2" / "aia-dev"
SELECTION = HERE / "aia_selection.json"
PROTOCOL_PATH = HERE / "aia_protocol.json"
SEEDS = {"dev": [2026093061, 2026093062, 2026093063],
         "holdout": [2026101061, 2026101062, 2026101063]}
REGIONS = ("rare", "medium", "common")
REGION_PROCESSING = ("preserve", "background_mix", "shuffle")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def object_hash(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def dependency_hashes():
    paths = [HERE / name for name in ("aia.py", "evaluate_aia.py", "test_aia.py")]
    paths += [ROOT / "local_eval/data.py", ROOT / "local_eval/scoring.py", ROOT / "M/v1/model.py"]
    return {p.relative_to(ROOT).as_posix(): workspace.sha256(p) for p in paths}


def region_names(background):
    distribution = background.prefecture.value_counts(normalize=True).to_dict()
    rare = min(distribution, key=lambda p: (abs(distribution[p] - .003), str(p)))
    middle = min((p for p in distribution if p != rare),
                 key=lambda p: (abs(distribution[p] - .04), str(p)))
    common = max((p for p in distribution if p not in (rare, middle)),
                 key=lambda p: (distribution[p], str(p)))
    return {"rare": rare, "medium": middle, "common": common}


def regional_case(source, background, schema, split, seed, home):
    """Assign one consistent source-person -> prefecture mapping to all roles.

    The newly chosen private block contains the 163 AIA members plus 150 other
    target people, excluding the MIA shared people. Controls are assigned the
    same home county. All clinical/outcome values and exact match quotas remain
    unchanged. C-only regional corruption is a subsequent defense operation.
    """
    case = build_case(source, background, schema, split, seed)
    rows = list(source[FEATURES].itertuples(index=False, name=None))
    source_by_full = {row: i for i, row in enumerate(rows)}
    source_by_qi = {row: i for i, row in enumerate(source[QI].itertuples(index=False, name=None))}
    if len(source_by_full) != len(source) or len(source_by_qi) != len(source):
        raise ValueError("Synthetic role mapping requires unique original rows/QI")
    members = set(case.private_roles["aia_member"])
    controls = set(case.private_roles["aia_control"])
    eligible = sorted(set(case.private_roles["target"]) - members - set(case.private_roles["mia_shared"]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 313, 24601]))
    private = members | set(rng.choice(eligible, 150, replace=False).tolist())
    moved = private | controls
    role_map = {idx: home if idx in moved else source.iloc[idx].prefecture for idx in range(len(source))}
    # Each role obtains county from the same original-person mapping, never
    # from its member/control label independently.
    for frame in (case.target, case.candidates):
        indices = [source_by_full[row] for row in frame[FEATURES].itertuples(index=False, name=None)]
        frame.loc[:, "prefecture"] = [role_map[idx] for idx in indices]
    q_indices = [source_by_qi[row] for row in case.challenge[QI].itertuples(index=False, name=None)]
    case.challenge.loc[:, "prefecture"] = [role_map[idx] for idx in q_indices]
    if set(case.challenge.prefecture) != {home}:
        raise AssertionError("Challenge concentration was not applied consistently")
    target_qi = set(case.target[QI].itertuples(index=False, name=None))
    membership = np.asarray([row in target_qi for row in case.challenge[QI].itertuples(index=False, name=None)])
    np.testing.assert_array_equal(membership, case.aia_member)
    for side in (True, False):
        subset = case.aia_member == side
        assert int(subset.sum()) == 163
        assert int(case.aia_rare[subset].sum()) == 13
        assert int(case.aia_onset[subset].sum()) == 16
        assert int((case.aia_rare[subset] & (case.aia_onset[subset] == 1)).sum()) == 4
    case.metadata.update({"regional_private_rows": len(private), "regional_home": home,
                          "regional_target_home_count": int(case.target.prefecture.eq(home).sum()),
                          "regional_challenge_home_count": int(case.challenge.prefecture.eq(home).sum()),
                          "regional_all_roles_consistent": True,
                          "regional_mapping_sha256": object_hash(sorted(role_map.items()))})
    return case


def process_region(release, background, mode, seed, region, style):
    release = release.copy(deep=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, REGIONS.index(region),
                                                       STYLES.index(style), REGION_PROCESSING.index(mode), 9801]))
    if mode == "background_mix":
        release.loc[:, "prefecture"] = rng.choice(background.prefecture.to_numpy(), len(release), replace=True)
    elif mode == "shuffle":
        release.loc[:, "prefecture"] = rng.permutation(release.prefecture.to_numpy())
    elif mode != "preserve":
        raise ValueError("Unknown regional defense")
    return release


def score(prediction, case, cfg):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (326,) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid AIA prediction")
    m = case.aia_member
    result = score_aia(None, prediction[m], case.aia_onset[m], case.aia_time[m],
                       None, prediction[~m], case.aia_onset[~m], case.aia_time[~m], cfg,
                       member_rare_mask=case.aia_rare[m], control_rare_mask=case.aia_rare[~m])
    return {"aia_no_phi": float(result.score),
            "aia_all": result.breakdown["r_time_all"], "aia_rare": result.breakdown["r_time_rare"],
            "member_success": result.breakdown["succ_time_member"],
            "control_success": result.breakdown["succ_time_control"]}


def aggregate(frame, grouping):
    result = []
    groups = grouping[0] if len(grouping) == 1 else grouping
    for key, group in frame.groupby(groups, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        result.append({**dict(zip(grouping, key)), "conditions": len(group),
                       "aia_no_phi_mean": group.aia_no_phi.mean(), "paired_delta_mean": group.delta.mean(),
                       "paired_delta_min": group.delta.min(), "wins": int((group.delta > 1e-12).sum()),
                       "ties": int((group.delta.abs() <= 1e-12).sum()), "losses": int((group.delta < -1e-12).sum()),
                       "member_success_mean": group.member_success.mean(),
                       "control_success_mean": group.control_success.mean()})
    return pd.DataFrame(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = (args.output or (DEFAULT_OUTPUT if args.stage == "dev" else DEFAULT_OUTPUT.with_name("aia-holdout"))).resolve()
    if not output.is_relative_to((HERE.parent / "local-output/v2").resolve()):
        raise ValueError("Output must remain under M/local-output/v2")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output is nonempty; preserve the existing run")
    output.mkdir(parents=True, exist_ok=True)
    before = workspace.verify_hashes()
    _, _, modules, cfg, versions = workspace.load_runtime()
    source = pd.read_csv(workspace.ORIGINALS / "A_bg.csv", float_precision="round_trip")
    background = pd.read_csv(workspace.ORIGINALS / "B.csv", float_precision="round_trip")
    schema = json.loads((workspace.ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))
    names = region_names(background)
    partitions = partition_source(source, schema)
    deps = dependency_hashes()
    protocol = {
        "format": "M-v2-aia-regional-protocol-v1", "seeds": SEEDS, "regions": list(REGIONS),
        "region_selection": "background proportions closest to 0.003, 0.04, and maximum; three distinct counties",
        "region_names_from_public_background": names, "region_processing": list(REGION_PROCESSING),
        "styles": list(STYLES), "primary_styles": list(PRIMARY_STYLES), "variants": list(aia.VARIANTS),
        "settings": {"support_thresholds": aia.SETTINGS, "distance_ratio": aia.DISTANCE_RATIO,
                     "soft_prefecture_penalty": aia.SOFT_PREFECTURE_PENALTY},
        "person_partition_seed": PARTITION_SEED,
        "partition_sha256": {name: hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
                             for name, indices in partitions.items()},
        "selection_rule": "Among candidates with preserve-primary mean delta >= -0.005, maximize equal-weight primary dev AIA before phi; tie by variant order. Require >1e-6 gain over reference, otherwise keep reference.",
        "scoring": {"tau": cfg["aia"]["tau_time"], "rare_z": cfg["scoring"]["aia_rare_se_z"], "phi": None},
        "quota": {"members": 163, "controls": 163, "rare_each": 13, "onset_each": 16, "rare_onset_each": 4},
        "input_hashes": {name: workspace.sha256(workspace.ORIGINALS / name) for name in ("B.csv", "A_bg.csv", "schema.json")},
        "dependencies": deps,
        "limitations": ["Target phi unknown; scores are uncorrected local proxies, not official attack scores.",
                        "The dev/holdout person partition has already been used in prior local work; seeds are new samples, not entirely unseen people.",
                        "Source people are A_bg and background is own B, reversing their real-world roles.",
                        "Rare labels are public-detector proxies, not true is_rare labels.",
                        "County rewriting creates controlled counterfactual regional cohorts rather than recreating private generation.",
                        "Region mapping is applied to all same-person roles before C-only anonymization.",
                        "Seven stylized defenses and equal weighting do not estimate their prevalence among actual targets."],
    }
    protocol_hash = object_hash(protocol)
    if args.stage == "dev":
        if SELECTION.exists():
            raise ValueError("AIA selection already exists; preserve it rather than silently reselect")
        if PROTOCOL_PATH.exists() and json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != plain(protocol):
            raise ValueError("Predeclared AIA protocol differs; do not overwrite")
        save(PROTOCOL_PATH, protocol)
        selected = None
    else:
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        if selection["protocol_sha256"] != protocol_hash or selection["dependencies"] != deps:
            raise ValueError("Frozen AIA protocol/code does not match development selection")
        selected = selection["selected_variant"]
    run = {"stage": args.stage, "started_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
           "protocol_sha256": protocol_hash, "protocol": protocol, "selected_variant": selected,
           "source_checks_before": before, "versions": versions, "compute_threads": 1,
           "holdout_executed": args.stage == "holdout", "actual_target_data_used": False, "cases": []}
    save(output / ("protocol.json" if args.stage == "dev" else "freeze.json"), run)
    started = perf_counter()
    results, completed = [], 0
    with threadpool_limits(limits=1):
        for seed in SEEDS[args.stage]:
            for region in REGIONS:
                case = regional_case(source, background, schema, args.stage, seed, names[region])
                run["cases"].append({**case.metadata, "region": region, "challenge_hash": frame_hash(case.challenge),
                                      "target_hash": frame_hash(case.target)})
                for style in STYLES:
                    base_c = make_release(case.target, case.background, style, seed, schema)
                    for mode in REGION_PROCESSING:
                        release = process_region(base_c, case.background, mode, seed, region, style)
                        inputs = {"c": release[FEATURES].copy(deep=True), "background": case.background[FEATURES].copy(deep=True),
                                  "challenge": case.challenge[QI].copy(deep=True), "cfg": copy.deepcopy(cfg)}
                        if args.stage == "dev":
                            predictions = aia.predict_variants(**inputs)
                        else:
                            predictions = {name: aia.predict(**inputs, variant=name)
                                           for name in dict.fromkeys(("v1_reference", selected))}
                        # v1 selected AIA is exactly the pinned official predictor.
                        canonical_c = release[FEATURES].sort_values(FEATURES, kind="mergesort").reset_index(drop=True)
                        canonical_bg = case.background[FEATURES].sort_values(FEATURES, kind="mergesort").reset_index(drop=True)
                        _, reference = modules["pwscup2026.attack.aia"].aia_predict(canonical_c, inputs["challenge"], canonical_bg, cfg)
                        np.testing.assert_allclose(predictions["v1_reference"], reference, atol=1e-12, rtol=0)
                        diagnostics = aia.diagnose(**inputs) if args.stage == "dev" else None
                        for variant, predicted in predictions.items():
                            results.append({"seed": seed, "region": region, "style": style, "region_processing": mode,
                                            "variant": variant, "primary": style in PRIMARY_STYLES,
                                            "release_sha256": frame_hash(release),
                                            "predictions_sha256": hashlib.sha256(np.asarray(predicted, dtype="<f8").tobytes()).hexdigest(),
                                            "routed_queries": (diagnostics["routes"][variant]["sex_only"] + diagnostics["routes"][variant]["soft_prefecture"])
                                            if diagnostics else None,
                                            **score(predicted, case, cfg)})
                        completed += 1
                print(json.dumps({"completed_seed_region": [seed, region], "conditions": completed}), flush=True)
    frame = pd.DataFrame(results)
    ref = frame.loc[frame.variant.eq("v1_reference"), ["seed", "region", "style", "region_processing", "aia_no_phi"]].rename(columns={"aia_no_phi": "reference_aia"})
    frame = frame.merge(ref, on=["seed", "region", "style", "region_processing"], validate="many_to_one")
    frame["delta"] = frame.aia_no_phi - frame.reference_aia
    primary = frame.loc[frame.primary]
    summary = aggregate(primary, ["variant"])
    preserved = aggregate(primary.loc[primary.region_processing.eq("preserve")], ["variant"])
    for filename, data in [("results.csv", frame), ("summary.csv", summary),
                           ("by_region.csv", aggregate(primary, ["region", "variant"])),
                           ("by_processing.csv", aggregate(primary, ["region_processing", "variant"])),
                           ("by_style.csv", aggregate(frame, ["style", "variant"])),
                           ("sanity.csv", aggregate(frame.loc[~frame.primary], ["style", "variant"]))]:
        data.to_csv(output / filename, index=False)
    run.update({"source_checks_after": workspace.verify_hashes(), "conditions": completed,
                "elapsed_seconds": perf_counter() - started, "completed": True})
    if dependency_hashes() != deps:
        raise ValueError("Evaluation source changed while running")
    if args.stage == "dev":
        allowed = set(preserved.loc[preserved.paired_delta_mean.ge(-.005), "variant"])
        table = summary.set_index("variant")
        choice = max((name for name in aia.VARIANTS if name in allowed),
                     key=lambda name: (table.loc[name, "aia_no_phi_mean"], -aia.VARIANTS.index(name)))
        if table.loc[choice, "paired_delta_mean"] <= 1e-6:
            choice = "v1_reference"
        selection = {"component": "aia", "selected_variant": choice, "protocol_sha256": protocol_hash,
                     "dependencies": deps, "holdout_executed": False,
                     "dev_evidence": {"selected": table.loc[choice].to_dict(),
                                      "reference": table.loc["v1_reference"].to_dict(),
                                      "preserved_region_mean_deltas": preserved.to_dict("records")},
                     "selection_rule": protocol["selection_rule"], "dev_output": str(output)}
        save(SELECTION, selection)
        run["selected_variant"] = choice
    save(output / "run.json", run)
    lines = ["# v2 AIA：地域支持による切替の開発評価", "",
             f"区分：{args.stage}。選択：{run['selected_variant']}。",
             "元人物と背景は本戦配布の範囲内だが、この値は公式攻撃得点ではなくφ未控除のローカル代理値。", "",
             "|方式|主条件|AIA平均|v1差|勝/同/負|", "|---|---:|---:|---:|---|"]
    for row in summary.to_dict("records"):
        lines.append(f"|{row['variant']}|{row['conditions']}|{row['aia_no_phi_mean']:.6f}|{row['paired_delta_mean']:+.6f}|{row['wins']}/{row['ties']}/{row['losses']}|")
    lines += ["", "3つの新抽出×3地域×7加工×3県処理＝189条件。主比較はcopy・背景独立を除く135条件。",
              "地域集中の前処理は人物単位の一つの写像を全役割へ適用。標的・対照各163、希少各13、発症各16、交差各4を保持。",
              "県保持、背景比率へ再割当、県の列内シャッフルを別比較。シャッフルはCの県人数を保存するため、支持数だけでは改善しない条件も含む。",
              "基準AIAは全189条件でv1が採用した固定公式AIAと1e-12以内で一致。",
              "候補へ渡すのはC、独立背景、公開7QI、公開設定のみ。隠れたラベルや元人物番号は渡さない。",
              "新seedでも人物分割自体は既存評価で使用済み。真に未知の人物による評価ではない。背景と元人物の役割は本戦と逆で、希少も検出代理。",
              "背景独立条件の正のRは偶然や集団差でも生じるため、そこで高得点になるようには選択していない。",
              "県の条件を緩めると対照の的中率も変わる。単なる標的的中率ではなく公式の標的対照差で比較した。",
              "原本94件の実行前後一致、CPU計算1スレッド、個人値/ID/予測CSVの保存なし。",
              "holdoutはこのdev実行では行っていない。採用名・候補・コードハッシュを固定し、別の明示実行を待つ。"
              if args.stage == "dev" else "holdoutは固定された採用候補とv1基準だけを評価。再選択はしていない。"]
    (output / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "stage": args.stage, "selected_variant": run["selected_variant"],
                      "conditions": completed, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
