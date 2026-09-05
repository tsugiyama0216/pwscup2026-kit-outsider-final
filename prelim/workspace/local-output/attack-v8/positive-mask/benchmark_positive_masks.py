"""Weak-release, paired AIA mask study. No production truth or submissions.

Unlike full-source shadows, mixtures include only a fraction of the cohort.
Discovery selects a single rule; fresh seeds then evaluate that locked rule.
All inference uses public release/challenge/background values. Truth is used
only for scoring and whole-target baseline choice (as in the official blend).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "local-output/attack-v6/shadow-benchmark"):
    sys.path.insert(0, str(path))
import benchmark as bench
from attack_phase_v5.aia_structured_ensemble import build_aia_v5_variants

OUT = Path(__file__).resolve().parent


def build_experts(world, releases, cfg):
    baseline, _, _ = bench.v4.build_aia_v4(
        world.attack_background, releases, world.challenges,
        main_fraction=0.5, variant_fractions=[0.5], abstain_time=11.,
        horizon_time=10., link_trust_distance=bench.v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
        baseline_abstain_time=bench.shadow.choose_low_hit_time(
            world.attack_background, horizon=10., tau=.5),
        anomalous_event_rate=bench.v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        anomalous_link_trust_distance=bench.v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        integrity_auc_threshold=bench.v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    independent, _, _ = bench.build_aia_independent(
        world.attack_background, releases, world.challenges,
        member_count=163, horizon=10., abstain_time=11.)
    variants, _ = build_aia_v5_variants(
        baseline, independent, releases, world.challenges,
        member_count=163, abstain_time=11., horizon_time=10.)
    return {"v4": baseline, "independent": independent,
            "routed": variants["consensus_v4_copula__v4_event_margin"]}


def run_seed(seed, candidates, b_self, background, detector, cfg, fractions):
    spec = bench.shadow.ShadowSpec(n_targets=4, candidate_rows=1049,
        target_rows=1049, shared_total=100, minimum_shared=25, maximum_shared=26)
    world = bench.shadow.build_shadow_world(candidates, b_self, background, detector, spec, seed)
    bench.exactify_visible_control_matching(world)
    bench.shadow.validate_world(world, spec)
    results = []
    for fraction in fractions:
        releases = {}
        for target, source in world.aia_sources.items():
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(target), int(fraction*100), 8181]))
            n = int(round(len(source)*fraction))
            selected = source.iloc[rng.choice(len(source), n, replace=False)]
            other = world.attack_background.iloc[rng.choice(len(world.attack_background), len(source)-n, replace=False)]
            release = pd.concat([selected, other], ignore_index=True)
            releases[target] = bench.shadow.scenario_release(release, world.attack_background, "noise_0p03", cfg, rng)
        experts = build_experts(world, releases, cfg)
        for target, challenge in world.challenges.items():
            truth = world.aia_truth[target]
            times = {name: bench.aligned_time(frame, target, challenge) for name, frame in experts.items()}
            # Only the full original target score is used to choose an expert.
            source_scores = {name: bench.score_block(time, truth, cfg)["power"] for name, time in times.items()}
            winner = max(source_scores, key=source_scores.get)
            base = times[winner].copy()
            base[np.isclose(base,10.)] = 9.5
            active = {name: ~np.isclose(time,11.) for name,time in times.items()}
            votes = sum(mask.astype(int) for mask in active.values())
            masks = {"union": votes >= 1, "majority": votes >= 2, "intersection": votes == 3,
                     "two_expert_union": active["v4"] | active["independent"],
                     "all_rows": np.ones(len(base), bool)}
            predictions = {"baseline": base}
            for name, mask in masks.items():
                # Preserve the winning expert's active event decoder, changing
                # only coverage. New rows use a fixed, not truth-derived edge.
                values = base.copy()
                values[~mask] = 11.
                values[mask & np.isclose(values,11.)] = 9.5
                predictions[name] = values
                # Separate event-preserving negative control for union.
                if name in ("union", "two_expert_union"):
                    event_values = values.copy()
                    newly = mask & np.isclose(base,11.)
                    for expert in ("independent", "v4", "routed"):
                        use = newly & active[expert]
                        event_values[use] = times[expert][use]
                        event_values[use & np.isclose(event_values,10.)] = 9.5
                    predictions[name+"_expert_time"] = event_values
            base_score = bench.score_block(base,truth,cfg)["power"]
            for name, values in predictions.items():
                metrics = bench.score_block(values,truth,cfg)
                results.append({"seed":seed,"fraction":fraction,"target":target,
                    "strategy":name,"winner":winner,"base_power":base_score,
                    "base_positive":base_score>0.,"weak_positive":.02 <= base_score <= .25,
                    "original_winner_power":source_scores[winner],
                    "delta":metrics["power"]-base_score,"active_rows":int((values!=11.).sum()), **metrics})
        print(f"seed={seed} fraction={fraction} complete",flush=True)
    return results


def summarize(frame):
    results=[]
    for condition in ("all","base_positive","weak_positive"):
        chosen = frame if condition=="all" else frame.loc[frame[condition]]
        for name, group in chosen.groupby("strategy"):
            delta=group["delta"].to_numpy()
            seed_delta=group.groupby("seed")["delta"].mean().to_numpy()
            results.append({"condition":condition,"strategy":name,"cases":len(group),
                "seeds":len(seed_delta),"base_mean":float(group.base_power.mean()),
                "delta_mean":float(delta.mean()),"delta_p10":float(np.quantile(delta,.1)),
                "delta_min":float(delta.min()),"win_rate":float((delta>1e-12).mean()),
                "loss_rate":float((delta< -1e-12).mean()),
                "seed_delta_min":float(seed_delta.min()),"active_mean":float(group.active_rows.mean())})
    return pd.DataFrame(results)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=("discovery","validation"),default="discovery")
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--fractions",default="0,.05,.15,.35")
    parser.add_argument("--world",choices=("original","bg_cohort"),default="original")
    args=parser.parse_args()
    first=20280901 if args.split=="discovery" else 20281001
    candidates=pd.read_csv(bench.DIST/"B_practice.csv")
    b_self=pd.read_csv(bench.DIST/"B_self.csv")
    background=pd.read_csv(bench.DIST/"A_bg.csv")
    detector=bench.shadow.load_rare_detector(bench.DIST)
    cfg=bench.load_config(bench.KIT_CONFIG_PATH)
    records=[]
    for seed in range(first,first+args.count):
        if args.world == "bg_cohort":
            rng=np.random.default_rng(np.random.SeedSequence([seed,842]))
            indices=rng.permutation(len(background))
            members=background.iloc[indices[:1049]].reset_index(drop=True)
            remaining=background.iloc[indices[1049:]].reset_index(drop=True)
            self_frame=members.copy()
            self_frame["is_rare"]=bench.shadow.public_rare_mask(members,detector).astype(int)
        else:
            members,self_frame,remaining=candidates,b_self,background
        records.extend(run_seed(seed,members,self_frame,remaining,detector,cfg,
            tuple(map(float,args.fractions.split(",")))))
        suffix="" if args.world=="original" else "_"+args.world
        pd.DataFrame(records).to_csv(OUT/f"{args.split}{suffix}_scores.csv",index=False)
    summary=summarize(pd.DataFrame(records))
    summary.to_csv(OUT/f"{args.split}{suffix}_summary.csv",index=False)
    print(summary.to_string(index=False))
    if args.split=="discovery" and args.world=="original":
        chosen=summary.loc[summary.condition.eq("weak_positive") & summary.strategy.ne("baseline")].sort_values("delta_mean",ascending=False).iloc[0]
        (OUT/"locked_rule.json").write_text(json.dumps({"strategy":chosen.strategy,
            "seed_start":first,"seed_count":args.count,"selection":"weak positive discovery mean delta",
            "caveat":"mixture shadows, not a leaderboard estimate"},indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
