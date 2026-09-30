"""Freeze protocol, learn on development people, compare on fixed shadow cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import attack_workspace as workspace

ROOT = Path(__file__).resolve().parent
OUTPUT = workspace.TS_OUTPUT / "attack-research" / "round02"


def sources():
    paths = list(ROOT.glob("*.py")) + list(ROOT.parent.glob("*.py")) + [ROOT.parent / "protocol.json"]
    paths += list((workspace.TS_ROOT / "local_eval").glob("*.py"))
    return {str(p.relative_to(workspace.TS_ROOT)).replace("\\", "/"): workspace.sha256(p) for p in paths}


def read_data(pd):
    read = lambda name: pd.read_csv(workspace.ORIGINALS / name, float_precision="round_trip")
    return read("A_bg.csv"), read("B.csv"), json.loads((workspace.ORIGINALS / "schema.json").read_text())


def prepare(np, pd, protocol, out):
    import joblib
    from .features import extract_mia, extract_aia, proposal_features
    from local_eval.data import build_case, make_release
    from local_eval.__main__ import save
    path = out / "training.joblib"
    if path.exists():
        raise ValueError("Training cache exists; do not overwrite a research round")
    source, bg, schema = read_data(pd)
    mx, my, ax, ay, hx, hy = [], [], [], [], [], []
    cases = []
    for seed in protocol["training_seeds"]:
        case = build_case(source, bg, schema, "dev", seed)
        cases.append(case.metadata)
        for style in protocol["primary_styles"] + protocol["diagnostic_styles"]:
            release = make_release(case.target, case.background, style, seed, schema)
            mia_x, _ = extract_mia(release, case.background, case.candidates)
            aia_x, proposals, _, _ = extract_aia(release, case.background, case.challenge)
            mx.append(mia_x); my.append(case.mia_truth)
            ax.append(aia_x); ay.append(case.aia_member.astype(int))
            hit = np.abs(proposals.astype(float) - case.aia_time[:, None]) <= .5
            hx.append(proposal_features(aia_x, proposals))
            hy.append((2*case.aia_member[:, None].astype(int) + hit.astype(int)).reshape(-1))
        print(f"Training shadows prepared {seed}", flush=True)
    arrays = {"mia_x": np.concatenate(mx), "mia_y": np.concatenate(my),
        "aia_x": np.concatenate(ax), "aia_y": np.concatenate(ay),
        "hit_x": np.concatenate(hx), "hit_y": np.concatenate(hy)}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise AssertionError("Nonfinite training values")
    joblib.dump(arrays, path, compress=0)
    save(out / "training_manifest.json", {"protocol": protocol, "cases": cases,
        "source_sha256": sources(), "training_sha256": workspace.sha256(path),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "label_access": "Development-person shadow labels only; prediction feature functions receive no labels"})


def fit(np, out):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from local_eval.__main__ import save
    manifest = json.loads((out / "training_manifest.json").read_text())
    if manifest["source_sha256"] != sources():
        raise ValueError("Research source changed after preparing training data")
    path = out / "models.joblib"
    if path.exists():
        raise ValueError("Model already exists")
    data = joblib.load(out / "training.joblib", mmap_mode="r")
    models, details = {}, {}
    for task in ("mia", "aia", "hit"):
        start = perf_counter()
        model = HistGradientBoostingClassifier(learning_rate=.08, max_iter=160,
            max_leaf_nodes=15, l2_regularization=5., min_samples_leaf=35,
            early_stopping=False, random_state=20260927)
        model.fit(data[task+"_x"], data[task+"_y"])
        models[task] = model
        details[task] = {"seconds": perf_counter()-start, "classes": model.classes_.tolist()}
        print(f"Fitted {task}: {details[task]['seconds']:.2f}s", flush=True)
    joblib.dump(models, path)
    save(out / "model_manifest.json", {"source_sha256": sources(), "models_sha256": workspace.sha256(path),
        "training_manifest_sha256": workspace.sha256(out/"training_manifest.json"), "fit": details})


def learned_predictions(np, models, release, case):
    from .features import extract_mia, extract_aia, proposal_features
    mx, rule_mia = extract_mia(release, case.background, case.candidates)
    ax, proposals, rule_aia, simple_time = extract_aia(release, case.background, case.challenge)
    m = models["mia"].predict_proba(mx)[:, 1]
    ap = models["aia"].predict_proba(ax)[:, 1]
    hp = models["hit"].predict_proba(proposal_features(ax, proposals)).reshape(
        len(ax), proposals.shape[1], -1)
    index = {int(c): i for i, c in enumerate(models["hit"].classes_)}
    member_hit, control_hit = hp[:, :, index[3]], hp[:, :, index[1]]
    mia = {"likelihood_rule": rule_mia, "learned": m, "learned_rule": .8*m + .2*rule_mia}
    aia = {"rule_top_half": np.where(np.argsort(np.argsort(-rule_aia, kind="stable"), kind="stable") < 163, simple_time, 11.)}
    for label, membership in (("learned", ap), ("rule", rule_aia)):
        selected = np.argsort(np.argsort(-membership, kind="stable"), kind="stable") < 163
        aia[label+"_top_half"] = np.where(selected, simple_time, 11.)
    for penalty in (.25, .5, .75, 1., 1.5, 2., 3.):
        utility = member_hit - penalty*control_hit
        best = utility.argmax(axis=1)
        response = proposals[np.arange(len(ax)), best].astype(float)
        gain = utility[np.arange(len(ax)), best]
        label = f"joint_{penalty:g}"
        aia[label] = np.where(gain > 0., response, 11.)
        selected = np.argsort(np.argsort(-gain, kind="stable"), kind="stable") < 163
        aia[label+"_half"] = np.where(selected, response, 11.)
    return mia, aia


def evaluate(np, pd, cfg, protocol, out, confirmation=False):
    import joblib
    from local_eval.data import build_case, make_release
    from local_eval.methods import official, multiview
    from local_eval.scoring import score_arrays
    from local_eval.__main__ import save
    from prelim_transfer.methods import load_legacy, build_aia
    models = joblib.load(out / "models.joblib")
    frozen = json.loads((out / "model_manifest.json").read_text())
    if frozen["source_sha256"] != sources():
        raise ValueError("Model/source hashes differ")
    source, bg, schema = read_data(pd)
    directory = out / ("confirmation" if confirmation else "development")
    if directory.exists():
        raise ValueError("Do not overwrite evaluation results")
    directory.mkdir()
    selected = json.loads((out / "selection.json").read_text()) if confirmation else None
    modules = load_legacy()
    rows, metadata = [], []
    key = "confirmation" if confirmation else "development"
    for seed in protocol[key+"_seeds"]:
        case = build_case(source, bg, schema, protocol[key+"_split"], seed)
        metadata.append(case.metadata)
        for style in protocol["primary_styles"] + protocol["diagnostic_styles"]:
            release = make_release(case.target, case.background, style, seed, schema)
            mias, aias = learned_predictions(np, models, release, case)
            # Historical baselines receive exactly the same public view.
            ref = official(c=release, background=case.background, candidates=case.candidates, challenge=case.challenge, cfg=cfg)
            mv = multiview(c=release, background=case.background, candidates=case.candidates, challenge=case.challenge, cfg=cfg)
            old, _ = build_aia(modules, case.background, {"0": release}, {"0": case.challenge})
            structured = old["structured"].set_index("challenge_row_id").loc[case.challenge.challenge_row_id, "time_hat"].to_numpy()
            mias.update(official=ref["mia"], multiview=mv["mia"])
            aias.update(official=ref["aia"], structured=structured)
            if selected:
                mias = {k: mias[k] for k in (selected["mia"], "official", "multiview")}
                aias = {k: aias[k] for k in (selected["aia"], "official", "structured")}
            for task, variants in (("mia", mias), ("aia", aias)):
                for name, prediction in variants.items():
                    m, a = (prediction, ref["aia"]) if task == "mia" else (ref["mia"], prediction)
                    scores = score_arrays(m, a, case, cfg)
                    relevant = {k: v for k, v in scores.items() if k.startswith(task)}
                    if task == "mia":
                        top = np.argsort(-prediction, kind="stable")[:26]
                        relevant["top26_recall"] = float(case.mia_truth[top].sum()/26)
                    rows.append({"task": task, "method": name, "seed": seed, "style": style, **relevant})
            pd.DataFrame(rows).to_csv(directory/"results.csv", index=False)
            print(f"Evaluated {key} seed={seed} style={style}", flush=True)
    frame = pd.DataFrame(rows)
    primary = frame[frame["style"].isin(protocol["primary_styles"])]
    summaries = []
    for task, metric in (("mia", "mia"), ("aia", "aia_no_phi")):
        for name, group in primary[primary.task.eq(task)].groupby("method", sort=False):
            summaries.append({"task":task, "method":name, "cases":len(group), "mean":group[metric].mean(),
                "minimum":group[metric].min(), "perfect_cases":int((group[metric] >= 1-1e-12).sum())})
    summary = pd.DataFrame(summaries).sort_values(["task", "mean"], ascending=[True,False], kind="stable")
    summary.to_csv(directory/"summary.csv", index=False)
    frame.groupby(["task", "method", "style"])[["mia", "aia_no_phi"]].mean().to_csv(directory/"by_style.csv")
    if not confirmation:
        picked = {task: summary[summary.task.eq(task)].iloc[0].method for task in ("mia", "aia")}
        save(out/"selection.json", {**picked, "source_sha256":sources(), "model_sha256":workspace.sha256(out/"models.joblib"),
            "basis":"Maximum mean on fixed development cases; ties preserve declared method order", "confirmation_used":False})
    save(directory/"verification.json", {"cases":metadata, "source_sha256":sources(),
        "input_verification":workspace.verify_hashes(), "selected":selected,
        "complete_goal": all(float(summary[summary.task.eq(t)].iloc[0]["minimum"]) >= 1-1e-12 for t in ("mia","aia"))})
    print(summary.to_string(index=False), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "fit", "evaluate", "confirm"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(workspace.TS_OUTPUT.resolve()):
        parser.error("Output must be in local-output")
    workspace.verify_hashes()
    np,pd,_,cfg,_=workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    protocol=json.loads((ROOT.parent/"protocol.json").read_text())
    out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.action=="prepare": prepare(np,pd,protocol,out)
        elif args.action=="fit": fit(np,out)
        else: evaluate(np,pd,cfg,protocol,out,confirmation=args.action=="confirm")
    workspace.verify_hashes()


if __name__=="__main__": main()
