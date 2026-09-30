"""Additional development stress test with independently synthesized fillers.

This preserves the selected MIA and its original evaluation. It constructs a
new underlying target B before applying a separate defense transform. Only
aggregate observations and metrics are written; synthetic rows and hidden
membership labels remain in memory.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes, load_runtime
from local_eval.data import FEATURES, QI, make_release, partition_source, rare_mask
from M.v2.mia import raw_scores, adjust_scores
from M.v2.shadow_joint import build_world
from M.v2.evaluate_mia import metric, frozen_hashes

PROTOCOL = {
    "version": 1, "stage": "dev", "purpose": "Additional development robustness check; not holdout",
    "world_seeds": [2026093061, 2026093062], "visible_rare_totals": [0, 5, 11],
    "layout": "mixed", "strengths": [0.0, 1.0], "fixed_selected_strength": 1.0,
    "excluded_sanity_styles": ["copy", "independent"], "thread_limit": 1,
    "underlying_B_generator": {
        "family": "Gaussian empirical copula",
        "fit_pool": "A_bg development partition, excluding all 1049 candidate persons",
        "numeric_columns": [c for c in FEATURES if c != "prefecture"],
        "covariance": "0.95 covariance + 0.05 diagonal, diagonal floor 1e-6",
        "prefecture": "Independent draw conditional on generated sex, from fit pool",
        "time": "Copula time quantile within generated onset/death state; non-event rows set to 10",
        "freshness": "Reject QI collisions with every A_bg/B original and previously accepted fillers in this world",
        "rare_quota": "Rejection sample exactly 24 minus shared rare per target, using the public detector",
        "batch_rows": 4096, "maximum_draws_per_target": 250000,
    },
    "defense_generator": "Original registered mixed style per target, applied after new underlying B is assembled",
    "comparison": "Fixed strength 1 versus strength 0, both with the same correct proxy-private mask",
    "selection_action": "None: preserve mia_selection.json and all existing v2 code/results",
    "limitations": [
        "Fresh filler QI overlap is zero; original reused-filler worlds had at least about 673 common donors per pair. Neither matches the full real 25-27-person pairwise topology.",
        "Only the 1049 candidate persons have exact real-style exclusive membership; the other original target overlaps are absent.",
        "The background is B and person source is A_bg; actual contest roles are reversed.",
        "Rare labels use the public detector surrogate, not true is_rare.",
        "Development person partitions were used in prior work; new seeds are not new held-out people.",
        "Synthetic population assumptions and separate defense assumptions both affect results.",
    ],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def keys(frame, columns=QI):
    return set(frame[columns].itertuples(index=False, name=None))


class FreshCopula:
    def __init__(self, pool, schema):
        self.pool = pool[FEATURES].copy(deep=True).reset_index(drop=True)
        self.schema = schema
        self.numeric = PROTOCOL["underlying_B_generator"]["numeric_columns"]
        values = self.pool[self.numeric].to_numpy(float)
        ranks = np.column_stack([rankdata(values[:, j], method="average") for j in range(values.shape[1])])
        z = norm.ppf((ranks - .5) / len(values))
        covariance = np.cov(z, rowvar=False)
        self.covariance = .95 * covariance + .05 * np.diag(np.maximum(np.diag(covariance), 1e-6))
        self.scale = np.sqrt(np.maximum(np.diag(self.covariance), 1e-6))
        self.sorted_values = np.sort(values, axis=0)
        self.positions = np.linspace(0., 1., len(values))
        self.time_position = self.numeric.index("time")

    def draw(self, rng, n):
        z = rng.multivariate_normal(np.zeros(len(self.numeric)), self.covariance, n, check_valid="raise")
        u = norm.cdf(z / self.scale)
        data = {column: np.interp(u[:, j], self.positions, self.sorted_values[:, j])
                for j, column in enumerate(self.numeric)}
        result = pd.DataFrame(data)
        for column in ("sex", "smoking", "onset", "death"):
            result[column] = np.rint(result[column]).astype(int)
        result["death"] = ((result.onset == 0) & (result.death == 1)).astype(int)
        result["prefecture"] = ""
        for sex in (0, 1):
            positions = np.flatnonzero(result.sex.to_numpy() == sex)
            donors = self.pool.loc[self.pool.sex.eq(sex), "prefecture"]
            if donors.empty:
                donors = self.pool.prefecture
            result.loc[positions, "prefecture"] = rng.choice(donors.to_numpy(), len(positions), replace=True)
        for event, death in ((0, 0), (0, 1), (1, 0)):
            positions = np.flatnonzero((result.onset.to_numpy() == event) & (result.death.to_numpy() == death))
            if event == death == 0:
                result.loc[positions, "time"] = 10.
            else:
                donors = self.pool.loc[self.pool.onset.eq(event) & self.pool.death.eq(death), "time"].to_numpy(float)
                if len(donors) == 0:
                    raise ValueError("Fitted outcome state has no source time distribution")
                result.loc[positions, "time"] = np.interp(u[positions, self.time_position],
                    np.linspace(0., 1., len(donors)), np.sort(donors))
        for column, bounds in self.schema["ranges"].items():
            result[column] = result[column].clip(*bounds)
        result["age"] = np.rint(result.age.clip(*self.schema["age_range"])).astype(int)
        result["time"] = result.time.clip(1 / 365.25, 10.)
        return result[FEATURES]

    def fill(self, *, rng, count, rare_count, forbidden):
        accepted = []
        needed = {False: count - rare_count, True: rare_count}
        local_seen = set()
        draws = collision_rejections = 0
        while any(needed.values()):
            if draws >= PROTOCOL["underlying_B_generator"]["maximum_draws_per_target"]:
                raise ValueError("Fresh filler rejection sampler exhausted; quotas not relaxed")
            batch = self.draw(rng, PROTOCOL["underlying_B_generator"]["batch_rows"])
            draws += len(batch)
            detected = rare_mask(batch, self.schema)
            selected = []
            for index, row in enumerate(batch[QI].itertuples(index=False, name=None)):
                group = bool(detected[index])
                if needed[group] == 0:
                    continue
                if row in forbidden or row in local_seen:
                    collision_rejections += 1
                    continue
                local_seen.add(row)
                selected.append(index)
                needed[group] -= 1
                if not any(needed.values()):
                    break
            accepted.append(batch.iloc[selected].copy())
        filler = pd.concat(accepted, ignore_index=True)[FEATURES]
        if len(filler) != count or int(rare_mask(filler, self.schema).sum()) != rare_count:
            raise ValueError("Synthetic filler quota verification failed")
        if len(local_seen) != count or local_seen & forbidden:
            raise ValueError("Fresh filler uniqueness verification failed")
        return filler, {"generated_draws": draws, "accepted_rows": count,
                        "accepted_rare": rare_count, "collision_rejections": collision_rejections}


def defend(target, background, schema, *, style, seed, target_slot):
    release_seed = int(np.random.SeedSequence([seed, target_slot, 50]).generate_state(1)[0])
    base_style = "jitter" if style in ("category_perturb", "background_mix") else style
    c = make_release(target, background, base_style, release_seed, schema)
    rng = np.random.default_rng(np.random.SeedSequence([seed, target_slot, 60]))
    if style == "category_perturb":
        for column, probability in (("sex", .15), ("smoking", .25)):
            change = rng.random(len(c)) < probability
            c.loc[change, column] = 1 - c.loc[change, column]
        change = rng.random(len(c)) < .30
        c.loc[change, "prefecture"] = rng.choice(background.prefecture.to_numpy(), int(change.sum()))
    elif style == "background_mix":
        change = rng.choice(len(c), round(.35 * len(c)), replace=False)
        donors = rng.choice(len(background), len(change), replace=False)
        c.loc[change, FEATURES] = background.iloc[donors][FEATURES].to_numpy()
    return c[FEATURES].reset_index(drop=True)


def dependency_hashes():
    paths = ["M/v2/mia.py", "M/v2/shadow_joint.py", "M/v2/evaluate_mia.py", "M/v2/stress_mia.py",
             "M/v2/mia_protocol.json", "M/v1/model.py", "M/v1/actual.py", "local_eval/data.py"]
    return {path: digest(ROOT / path) for path in paths}


def run(output):
    started = time.monotonic()
    output = Path(output).resolve()
    allowed = (ROOT / "M/local-output/v2").resolve()
    if not output.is_relative_to(allowed) or output == allowed or (output.exists() and any(output.iterdir())):
        raise ValueError("Use an empty subdirectory of M/local-output/v2 for the new stress results")
    selected_path = HERE / "mia_selection.json"
    selected_sha = digest(selected_path)
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    if float(selection["strength"]) != PROTOCOL["fixed_selected_strength"]:
        raise ValueError("The registered stress comparison requires fixed strength 1.0")
    if frozen_hashes() != selection["frozen_hashes"]:
        raise ValueError("The MIA dependencies no longer match the fixed development selection")
    before = verify_hashes()
    _, _, _, cfg, versions = load_runtime()
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    dependencies = dependency_hashes()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", PROTOCOL)
    protocol_sha = digest(output / "protocol.json")
    write_json(output / "freeze.json", {"stage": "dev", "fixed_strength": 1.,
        "mia_selection_sha256": selected_sha, "model_dependency_sha256": dependencies,
        "protocol_sha256": protocol_sha})
    print(json.dumps({"registered_before_scoring": True, "world_seeds": PROTOCOL["world_seeds"],
                      "rare_totals": PROTOCOL["visible_rare_totals"], "strengths": PROTOCOL["strengths"]}), flush=True)
    originals = ROOT.parent / "anonymization/team10/participant_data"
    source = pd.read_csv(originals / "A_bg.csv", float_precision="round_trip")
    background = pd.read_csv(originals / "B.csv", float_precision="round_trip")
    schema = json.loads((originals / "schema.json").read_text(encoding="utf-8-sig"))
    input_paths = ["source_manifest.json", "../anonymization/team10/participant_data/A_bg.csv",
                   "../anonymization/team10/participant_data/B.csv", "../anonymization/team10/participant_data/schema.json"]
    input_hashes = {path: digest(ROOT / path) for path in input_paths}
    source_keys, background_keys = keys(source), keys(background)
    background_full_keys = keys(background, FEATURES)
    if len(source_keys) != len(source) or len(background_keys) != len(background) or source_keys & background_keys:
        raise ValueError("Source/reference QI uniqueness or separation check failed")
    source_lookup = {row: index for index, row in enumerate(source[QI].itertuples(index=False, name=None))}
    split_indices = partition_source(source, schema)["dev"]
    rows, worlds = [], []
    for seed in PROTOCOL["world_seeds"]:
        generator = None
        original_candidates = None
        for rare_total in PROTOCOL["visible_rare_totals"]:
            world = build_world(source, background, schema, split="dev", seed=seed,
                                visible_rare_total=rare_total, layout="mixed")
            candidate_keys = keys(world.candidates)
            positions = {source_lookup[row] for row in candidate_keys}
            if len(positions) != 1049:
                raise ValueError("Candidate source identity check failed")
            pool_indices = sorted(set(split_indices) - positions)
            if generator is None:
                generator = FreshCopula(source.iloc[pool_indices], schema)
                original_candidates = candidate_keys
            elif candidate_keys != original_candidates:
                raise ValueError("World candidate pool changed across rare-count configurations")
            if keys(generator.pool) & candidate_keys:
                raise ValueError("Synthetic filler fit pool contains candidate persons")
            forbidden = source_keys | background_keys
            filler_sets, release_sets, raw_columns, target_reports = [], [], [], []
            for target_slot in range(22):
                membership = world.truth[:, target_slot]
                shared_count = int(membership.sum())
                shared_rare = int(world.rare[membership].sum())
                rng = np.random.default_rng(np.random.SeedSequence([seed, target_slot, 710]))
                filler, generated = generator.fill(rng=rng, count=1049 - shared_count,
                                                   rare_count=24 - shared_rare, forbidden=forbidden)
                filler_keys = keys(filler)
                if filler_keys & forbidden or filler_keys & candidate_keys:
                    raise ValueError("A generated noncandidate overlaps source, candidates, or other filler")
                forbidden |= filler_keys
                filler_sets.append(filler_keys)
                target = pd.concat([world.candidates.loc[membership, FEATURES], filler], ignore_index=True)
                if len(target) != 1049 or int(rare_mask(target, schema).sum()) != 24:
                    raise ValueError("Underlying B count/rare validation failed")
                observed_candidate_keys = keys(target) & candidate_keys
                intended_keys = keys(world.candidates.loc[membership])
                if observed_candidate_keys != intended_keys:
                    raise ValueError("Underlying B contains an unintended candidate or misses a shared one")
                style = world.metadata["styles"][target_slot]
                c = defend(target, background, schema, style=style, seed=seed, target_slot=target_slot)
                release_sets.append(keys(c, FEATURES))
                raw_columns.append(raw_scores(c=c, background=world.background, candidates=world.candidates))
                target_reports.append({"target_slot": target_slot, "defense_style": style,
                    "shared_candidates": shared_count, "shared_rare_candidates": shared_rare,
                    "filler_rows": len(filler), "filler_rare": 24 - shared_rare,
                    "filler_source_qi_overlap": 0, "filler_candidate_qi_overlap": 0,
                    "filler_prior_target_qi_overlap": 0,
                    "underlying_onset_count": int(target.onset.sum()),
                    "underlying_death_count": int(target.death.sum()),
                    "final_C_exact_background_rows": sum(row in background_full_keys for row in c[FEATURES].itertuples(index=False, name=None)),
                    **generated})
            raw = np.column_stack(raw_columns)
            for strength in PROTOCOL["strengths"]:
                scored = adjust_scores(raw, world.private_mask, strength)
                for j in range(22):
                    record = {"stage": "dev", "seed": seed, "visible_rare_total": rare_total,
                        "layout": "fresh_filler_mixed", "target_slot": j, "style": world.metadata["styles"][j],
                        "method": f"median_{strength:g}", "strength": strength,
                        "mia_members": int(world.truth[:, j].sum()),
                        "mia_rare_members": int(world.truth[world.rare, j].sum())}
                    record.update(metric(scored[:, j], world.truth[:, j], world.rare,
                                         float(cfg["scoring"]["low_fpr"]), _tpr_at_fpr))
                    rows.append(record)
            filler_overlap = [len(filler_sets[i] & filler_sets[j]) for i in range(22) for j in range(i)]
            release_overlap = [len(release_sets[i] & release_sets[j]) for i in range(22) for j in range(i)]
            if any(filler_overlap):
                raise ValueError("Underlying filler persons repeat across targets")
            worlds.append({"seed": seed, "visible_rare_total": rare_total,
                "topology": {key: value for key, value in world.metadata.items()
                             if key not in ("distinct_noncandidate_donors", "noncandidate_donor_uses", "limitations")},
                "topology_note": "Only candidate allocation and defense styles are reused; original donor fillers are discarded.",
                "fitted_noncandidate_source_rows": len(pool_indices),
                "source_background_qi_overlap": 0, "fit_pool_candidate_overlap": 0,
                "filler_pairwise_qi_overlap_max": max(filler_overlap),
                "filler_pairwise_qi_overlap_mean": float(np.mean(filler_overlap)),
                "final_release_pairwise_exact_row_overlap_mean": float(np.mean(release_overlap)),
                "release_overlap_note": "Final defense may deliberately reuse B background rows; this is separate from fresh underlying filler membership.",
                "targets": target_reports})
            write_json(output / "progress.json", {"stage": "dev", "worlds_completed": len(worlds),
                "worlds_total": 6, "elapsed_seconds": time.monotonic() - started})
            print(json.dumps({"stress_dev_worlds_completed": len(worlds), "worlds_total": 6}), flush=True)
    frame = pd.DataFrame(rows)
    primary = frame[~frame["style"].isin(PROTOCOL["excluded_sanity_styles"])]
    per_world = primary.groupby(["seed", "visible_rare_total", "method", "strength"])[["mia", "mia_all", "rare_max_uplift"]].mean().reset_index()
    baseline = per_world[per_world.method.eq("median_0")].set_index(["seed", "visible_rare_total"]).mia
    per_world["paired_delta"] = [row.mia - baseline.loc[(row.seed, row.visible_rare_total)] for row in per_world.itertuples(index=False)]
    summary = []
    for method, group in per_world.groupby("method"):
        delta = group.paired_delta.to_numpy(float)
        summary.append({"method": method, "strength": float(group.strength.iloc[0]), "primary_worlds": len(group),
            "mean_mia": float(group.mia.mean()), "mean_all_population_tpr": float(group.mia_all.mean()),
            "mean_rare_max_uplift": float(group.rare_max_uplift.mean()),
            "paired_delta_mean": float(delta.mean()), "paired_delta_min": float(delta.min()),
            "wins": int(np.sum(delta > 1e-12)), "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "losses": int(np.sum(delta < -1e-12))})
    frame.to_csv(output / "results.csv", index=False, lineterminator="\n")
    per_world.to_csv(output / "world_scores.csv", index=False, lineterminator="\n")
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, lineterminator="\n")
    primary.groupby(["method", "style"])[["mia", "mia_all", "rare_max_uplift"]].mean().to_csv(output / "by_style.csv", lineterminator="\n")
    primary.groupby(["method", "mia_rare_members"])[["mia", "mia_all", "rare_max_uplift"]].agg(["mean", "count"]).to_csv(output / "by_rare_members.csv", lineterminator="\n")
    write_json(output / "world_metadata.json", worlds)
    after = verify_hashes()
    if before != after or dependency_hashes() != dependencies or digest(selected_path) != selected_sha:
        raise ValueError("Registered dependencies, sources, or fixed selection changed during stress run")
    if any(digest(ROOT / path) != value for path, value in input_hashes.items()):
        raise ValueError("Synthetic-generator inputs changed during stress run")
    report = {"completed": True, "stage": "dev", "purpose": "Additional fixed-model development robustness check",
        "fixed_strength": 1., "mia_selection_sha256": selected_sha,
        "model_dependency_sha256": dependencies, "input_sha256": input_hashes,
        "protocol_sha256": protocol_sha, "protocol_path": str((output / "protocol.json").relative_to(ROOT)).replace("\\", "/"),
        "input_verification_before": before, "input_verification_after": after,
        "official_runtime_versions": versions, "all_target_conditions": 132, "primary_worlds": 6,
        "summary": summary, "underlying_filler_pairwise_qi_overlap_max": 0,
        "elapsed_seconds": time.monotonic() - started, "selection_changed": False,
        "official_submission": False, "limitations": PROTOCOL["limitations"]}
    write_json(output / "run.json", report)
    print(json.dumps({"completed": True, "summary": summary, "selection_changed": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)


if __name__ == "__main__":
    main()
