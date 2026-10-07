"""Select on T-style development mocks, freeze, then confirm on T's fixed 25."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import runtime as rt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from attack_research.v88.predict import CombinedAttack
from attack_research.run import read_data
from local_eval.data import build_case, make_release
from local_eval.scoring import score_arrays, validate_predictions
from model import compute_candidates, MIA_OPTIONS, AIA_OPTIONS


def source_hashes():
    return {name: rt.sha(rt.HERE/name) for name in ("runtime.py", "model.py", "evaluate.py", "protocol.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "confirm"))
    args = parser.parse_args()
    protocol = rt.read_json(rt.HERE/"protocol.json")
    out = rt.MOCK/"local-output"/"R-v2f"/args.stage
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preserve existing evaluation output")
    if args.stage == "dev" and (rt.HERE/"selection.json").exists():
        raise ValueError("A frozen selection already exists")
    before = rt.upstream_check()
    own = source_hashes()
    selection = None if args.stage == "dev" else rt.selected_check()
    out.mkdir(parents=True, exist_ok=True)
    rt.save_json(out/"started.json", {"utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                                    "upstream": before, "source_sha256": own})
    _, _, _, cfg, versions = rt.w.load_runtime()
    source, background, schema = read_data(pd)
    stage_key = "development" if args.stage == "dev" else "confirmation"
    styles = protocol["primary_styles"] + (protocol["development_diagnostics"] if args.stage == "dev" else [])
    t = CombinedAttack()
    rows = []
    cases = []
    with threadpool_limits(limits=1):
        for seed in protocol[stage_key+"_seeds"]:
            case = build_case(source, background, schema, protocol[stage_key+"_split"], seed)
            cases.append(case.metadata)
            for style in styles:
                c = make_release(case.target, case.background, style, seed, schema)
                mia, aia, route = compute_candidates(t, c=c, background=case.background,
                    candidates=case.candidates, challenge=case.challenge, cfg=cfg)
                mi = MIA_OPTIONS if args.stage == "dev" else tuple(dict.fromkeys(["T", "M", selection["mia"]]))
                ai = AIA_OPTIONS if args.stage == "dev" else tuple(dict.fromkeys(["T", selection["aia"]]))
                for component, choices in (("mia", mi), ("aia", ai)):
                    for option in choices:
                        answers = {"mia": mia[option] if component == "mia" else mia["T"],
                                   "aia": aia[option] if component == "aia" else aia["T"]}
                        m, a = validate_predictions(answers, case)
                        scored = score_arrays(m, a, case, cfg)
                        rows.append({"seed": seed, "style": style, "route": route,
                            "component": component, "option": option,
                            "score": scored["mia" if component == "mia" else "aia_no_phi"],
                            "whole": scored["mia_all" if component == "mia" else "aia_all"],
                            "rare": scored["mia_rare" if component == "mia" else "aia_rare"]})
                pd.DataFrame(rows).to_csv(out/"scores.csv", index=False)
                print(f"{args.stage}: {seed} {style} done", flush=True)
    frame = pd.DataFrame(rows)
    primary = frame[frame["style"].isin(protocol["primary_styles"])]
    summary = primary.groupby(["component", "option"]).score.mean()
    summary.to_csv(out/"summary.csv")
    primary.groupby(["component", "option", "style"]).score.mean().to_csv(out/"by_style.csv")
    if args.stage == "dev":
        selected, audits = {}, []
        for component, options in (("mia", MIA_OPTIONS), ("aia", AIA_OPTIONS)):
            part = frame[frame.component.eq(component)]
            means = part.groupby(["option", "style"]).score.mean().unstack("option")
            candidates = []
            for option in options:
                diffs = means.loc[protocol["primary_styles"], option]-means.loc[protocol["primary_styles"], "T"]
                null_delta = float(means.loc["independent", option]-means.loc["independent", "T"])
                eligible = bool(diffs.min() >= -.02-1e-12 and null_delta <= .02+1e-12)
                avg = float(summary.loc[(component, option)])
                audits.append({"component": component, "option": option, "mean": avg,
                    "minimum_style_delta_vs_T": float(diffs.min()), "independent_delta_vs_T": null_delta,
                    "eligible": eligible})
                if eligible:
                    candidates.append((option, avg))
            best = max(x[1] for x in candidates)
            selected[component] = next(name for name, value in candidates if value >= best-1e-12)
        selection = {**selected, "selected_before_confirmation": True,
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": own,
            "development_scores_sha256": rt.sha(out/"scores.csv"), "audit": audits,
            "official_result_predicted": False}
        rt.save_json(rt.HERE/"selection.json", selection)
    else:
        np.testing.assert_allclose([summary.loc[("mia","T")], summary.loc[("aia","T")]],
                                   [.7307692307692308, .8473837794327597], rtol=0, atol=1e-10)
    if source_hashes() != own or rt.upstream_check() != before:
        raise ValueError("Inputs or code changed during evaluation")
    rt.save_json(out/"report.json", {"completed": True, "stage": args.stage, "selection": selection,
        "summary": [{"component": c, "option": o, "mean": float(v)} for (c,o),v in summary.items()],
        "cases": cases, "versions": versions, "upstream": before,
        "source_sha256": own, "official_submission": False, "limitations": protocol["limitations"]})
    print(summary.to_string(), flush=True)
    print("Selected:", selection["mia"], selection["aia"], flush=True)


if __name__ == "__main__":
    main()
