"""Frozen four-model panel on released preliminary team24 data.

Prediction uses the pinned MAIN algorithm runtime. Scoring runs in a separate
process with the pinned PRELIMINARY scorer and organizer-released truth. These
previously exposed preliminary people are not an independent population.
No official submission, model selection, or coefficient fitting is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MOCK = HERE.parent
ROOT = MOCK.parents[1]
ATTACK = ROOT / "main_round/attack"
PRELIM = MOCK / "mock_TS_using_prelim"
OUTPUT_ROOT = MOCK / "local-output"
MODELS = ("Rv1", "Rv2", "pl1", "Tv105")


def sha(path):
    path = Path(path)
    if path.name.lower() == "token.txt":
        raise ValueError("Credentials are not panel inputs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean_json(value):
    """Preserve undefined legacy diagnostics as null, never as measured zero."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if type(value).__module__.startswith("numpy"):
        return clean_json(value.tolist() if hasattr(value, "tolist") else value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def output_path(path):
    path = Path(path).resolve()
    if path == OUTPUT_ROOT.resolve() or not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Panel output must be a new directory under main_round/mock_test/local-output")
    return path


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _launch(action, output, *, group=None):
    args = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), action, "--output", str(output)]
    if group:
        args += ["--group", group]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    subprocess.run(args, check=True, env=env, cwd=ROOT)


def prepare(output):
    if output.exists():
        raise ValueError("Preserve existing panel output; choose a new directory")
    output.mkdir(parents=True)
    # This child process may import PRELIMINARY code. Prediction gets a fresh process.
    sys.path.insert(0, str(MOCK))
    from mock_TS_using_prelim.common import verify_bundle
    from mock_TS_using_prelim.runner import export_public
    verified = verify_bundle()
    public = export_public(output / "public")
    plan = {
        "format_version": 1, "evaluation_round": "preliminary", "attacker_id": 24,
        "target_ids": public["target_ids"], "models": list(MODELS),
        "candidate_rows": 1049, "aia_rows_per_model": 20 * 326,
        "official_data_verification": verified, "public_manifest_sha256": sha(output / "public/bundle_manifest.json"),
        "adapter_sha256": sha(__file__), "selection_fixed": True, "additional_tuning": False,
        "historical_training_exposure": True, "independent_population": False,
        "main_round_ranking_validation": False,
        "inference_runtime": "main_round/attack/official-kit/starter/src",
        "scoring_runtime": "main_round/mock_test/mock_TS_using_prelim/official-kit/starter/src",
        "scoring_truth_source": "Organizer-released preliminary truth, preserved in existing provenance/data_manifest.json",
        "runtime_separation": "Fresh main prediction and preliminary scoring processes",
        "shared_components": "Rv2 and Rv1 AIA are the old v88 answer from frozen Tv105.predict_all. Rv2 uses its exact double-rank baseline; Rv1 MIA uses frozen M v2 selected strength.",
        "tv105_training_note": "Retains its original fixed per-input shadow_router procedure; no new hyperparameters, saved model fitting, or selection.",
        "official_submission_performed": False,
    }
    write_json(output / "panel_manifest.json", plan)
    print(json.dumps({"prepared": str(output), "targets": 20, "models": list(MODELS)}, ensure_ascii=False), flush=True)


def verify_public(output):
    manifest = read_json(output / "public/bundle_manifest.json")
    if manifest.get("round") != "prelim" or manifest.get("attacker_id") != 24 or len(manifest.get("target_ids", [])) != 20:
        raise ValueError("Expected actual preliminary team24 and 20 other targets")
    public = (output / "public").resolve()
    for name, digest in manifest["files"].items():
        path = (public / name).resolve()
        if not path.is_relative_to(public) or sha(path) != digest:
            raise ValueError("Public bundle missing or changed: " + name)
    actual = {p.relative_to(public).as_posix() for p in public.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"bundle_manifest.json"}:
        raise ValueError("Unexpected file in public input bundle")
    plan = read_json(output / "panel_manifest.json")
    if plan["public_manifest_sha256"] != sha(public / "bundle_manifest.json"):
        raise ValueError("Public manifest changed")
    return manifest


def initialize_main():
    if any(name == "pwscup2026" or name.startswith("pwscup2026.") for name in sys.modules):
        raise RuntimeError("Prediction requires a fresh process before any official import")
    source = ATTACK / "official-kit/starter/src"
    for path in (MOCK, ATTACK, ATTACK / "Attak_T", source):
        sys.path.insert(0, str(path))
    # The primary venv has an editable installation from another checkout.
    # Pin the worktree source before importing any algorithm that uses it.
    import pwscup2026
    if not Path(pwscup2026.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Wrong main official package origin")
    import attack_research  # Unchanged canonical compatibility bootstrap.
    import attack_workspace
    attack_workspace.load_runtime()
    return str(Path(pwscup2026.__file__).resolve())


@contextmanager
def no_target_truth_reads(public_directory):
    """Trusted-code audit, not an OS security boundary; leaves self B_self usable."""
    state = {"active": True, "blocked": []}
    forbidden = ((PRELIM / "data/private").resolve(), (PRELIM / "data/reference").resolve())
    public_directory = Path(public_directory).resolve()
    def audit(event, args):
        if event != "open" or not state["active"] or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        outside_public_data = (path.is_relative_to(ROOT) and path.suffix.lower() in {".csv", ".parquet", ".npz", ".zip"}
                               and not path.is_relative_to(public_directory))
        if outside_public_data or any(path.is_relative_to(base) for base in forbidden) or path.name.lower() == "token.txt":
            state["blocked"].append(path.name)
            raise RuntimeError("Prediction attempted a forbidden truth/reference/credential read")
    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


def _public_tables(output, manifest):
    import pandas as pd
    public = output / "public"
    own, target = public / "team24_processing", public / "attack_targets"
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    return {"candidates": read(own / "B.csv"), "self_labels": read(own / "B_self.csv"),
            "background": read(own / "A_bg.csv"), "self_challenge": read(target / "aia_challenge_24.csv"),
            "schema": read_json(own / "schema.json"), "cfg": read_json(public / "cfg.json"),
            "targets": {k: {"release": read(target / f"C_{k}.csv"), "challenge": read(target / f"aia_challenge_{k}.csv")}
                        for k in manifest["target_ids"]}}


def _validate_frames(mia, aia, tables):
    import numpy as np
    expected = set(tables["targets"])
    if list(mia.columns) != ["record_id", *map(str, sorted(expected))] or len(mia) != 1049:
        raise ValueError("Invalid MIA dimensions or target columns")
    if mia.record_id.duplicated().any() or not np.array_equal(mia.record_id, tables["candidates"].record_id):
        raise ValueError("MIA IDs/order differ from supplied public candidates")
    if list(aia.columns) != ["target_k", "challenge_row_id", "time_hat"] or aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("Invalid AIA columns or duplicate IDs")
    if set(aia.target_k) != expected or len(aia) != 326 * len(expected):
        raise ValueError("Invalid AIA target dimensions")
    for k in expected:
        if set(aia.loc[aia.target_k == k, "challenge_row_id"]) != set(tables["targets"][k]["challenge"].challenge_row_id):
            raise ValueError("AIA IDs differ from public challenge")
    m, a = mia.iloc[:, 1:].to_numpy(float), aia.time_hat.to_numpy(float)
    if not np.isfinite(m).all() or not np.isfinite(a).all() or (m < 0).any() or (m > 1).any() or (a < 0).any():
        raise ValueError("Prediction values outside declared bounds")


def _save_prediction(output, model, mia, aia, tables, provenance):
    _validate_frames(mia, aia, tables)
    destination = output / "predictions" / model
    destination.mkdir(parents=True, exist_ok=False)
    for name, frame in (("F_mia.csv", mia), ("F_aia.csv", aia)):
        frame.to_csv(destination / name, index=False, float_format="%.17g", lineterminator="\n")
    write_json(destination / "prediction_provenance.json", {**provenance, "model_id": model,
                "prediction_sha256": {n: sha(destination / n) for n in ("F_mia.csv", "F_aia.csv")}})


def predict(output, group, *, pilot=False):
    started = time.monotonic()
    manifest = verify_public(output)
    origin = initialize_main()
    import numpy as np
    import pandas as pd
    tables = _public_tables(output, manifest)
    expected_models = ("Rv1", "Rv2", "Tv105") if group == "shared" else ("pl1",)
    if not pilot and any((output / "predictions" / name).exists() for name in expected_models):
        raise ValueError("Preserve existing model predictions; choose a new panel output")
    work = output / ("pilot" if pilot else "work") / group
    if work.exists():
        work = work.with_name(group + "-attempt-" + str(time.time_ns()))
    work.mkdir(parents=True, exist_ok=False)
    hashes = {}
    frozen = read_json(ATTACK / "Attack_R/v2/production_dependencies.json")["files_sha256"]
    for name, digest in frozen.items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen dependency changed: " + name)
    maskmod = module_from_file("_panel_private_mask", ATTACK / "Attack_M/v1/actual.py")
    public = output / "public"
    read_str = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    mask, proof = maskmod.derive_private_mask(read_str(public / "team24_processing/B.csv"),
        read_str(public / "team24_processing/B_self.csv"), read_str(public / "attack_targets/aia_challenge_24.csv"),
        tables["schema"]["aia_qi_columns"], tables["schema"]["submitted_c_columns"])
    diagnostics = {}
    with no_target_truth_reads(public) as access:
        if group == "shared":
            tv = ATTACK / "attackT/Tv105"
            sys.path.insert(0, str(tv))
            import runtime as rt
            import model as numeric
            import synthesis
            if Path(numeric.__file__).resolve() != tv / "model.py" or Path(rt.__file__).resolve() != tv / "runtime.py":
                raise ImportError("Wrong frozen Tv105 numeric module")
            selection = read_json(tv / "selection.json")
            for name, digest in selection["source_hashes"].items():
                if sha(tv / name) != digest:
                    raise ValueError("Frozen Tv105 source changed: " + name)
            hashes = {"Tv105/selection.json": sha(tv / "selection.json"), **selection["source_hashes"]}
            mias = {model: pd.DataFrame({"record_id": tables["candidates"].record_id.to_numpy()}) for model in ("Rv2", "Tv105")}
            aias = {model: [] for model in ("Rv2", "Tv105")}
            rv2_numeric = module_from_file("_panel_rv2_numeric", ATTACK / "Attack_R/v2/model.py")
            reuse_check_count = 0
            target_items = sorted(tables["targets"].items())
            if pilot:
                target_items = target_items[:1]
            for k, item in target_items:
                tick = time.monotonic()
                answers, diag = numeric.predict_all(c=item["release"], background=tables["background"],
                    candidates=tables["candidates"], challenge=item["challenge"], cfg=tables["cfg"], schema=tables["schema"])
                chosen = {"Rv2": answers["Rv2"], "Tv105": synthesis.predict(answers, selection)}
                # rank(rank(x)) is idempotent, including ties. Compare the shared
                # baseline to the exact frozen Rv2 finishing function with mask.
                shared_mia = np.asarray(answers["Rv2"]["mia"], float).copy()
                shared_mia[mask] = 0.
                exact_mia = rv2_numeric.finish_mia(rv2_numeric.ranks(answers["Rv2"]["mia"])[:, None], mask)[:, 0]
                if not np.array_equal(shared_mia, exact_mia):
                    raise AssertionError("Shared Rv2 baseline differs from frozen finish_mia")
                reuse_check_count += 1
                for name, answer in chosen.items():
                    confidence = np.asarray(answer["mia"], float).copy()
                    confidence[mask] = 0.
                    mias[name][str(k)] = confidence
                    aias[name].append(pd.DataFrame({"target_k": k, "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(),
                                                   "time_hat": answer["aia"]}))
                diagnostics[str(k)] = {"seconds": time.monotonic() - tick,
                    "old_route": diag["old_route"], "new_route": diag["new_route"],
                    "new_confidence": diag["new_confidence"], "old_confidence": diag["old_confidence"]}
                write_json(work / "progress.json", diagnostics)
                print(f"Preliminary shared target {k}: {diagnostics[str(k)]['seconds']:.1f}s", flush=True)
            if not pilot:
                rv1 = module_from_file("_panel_rv1_numeric", ATTACK / "Attack_R/v1/model.py")
                mselection = read_json(ATTACK / "Attack_M/v2/selection.json")
                if mselection["mia_strength"] != 1.0:
                    raise ValueError("Unexpected frozen M v2 selection")
                keys, confidence = rv1.predict_mia_all(targets={k: item["release"] for k, item in tables["targets"].items()},
                    background=tables["background"], candidates=tables["candidates"], private_mask=mask, strength=mselection["mia_strength"])
                mias["Rv1"] = pd.DataFrame({"record_id": tables["candidates"].record_id.to_numpy(),
                                             **{str(k): confidence[:, i] for i, k in enumerate(keys)}})
                aias = {name: pd.concat(parts, ignore_index=True) for name, parts in aias.items()}
                aias["Rv1"] = aias["Rv2"].copy()
                diagnostics["Rv1_mia_strength"] = mselection["mia_strength"]
            diagnostics["shared_reuse_verification"] = {"targets": reuse_check_count,
                "Rv2_mia_exact_frozen_finish_match": True, "Rv1_AIA_equals_Rv2_selected_v88_AIA": True,
                "rv2_numeric_sha256": sha(ATTACK / "Attack_R/v2/model.py"),
                "rv1_numeric_sha256": sha(ATTACK / "Attack_R/v1/model.py"),
                "tv105_predict_all_sha256": sha(tv / "model.py"),
                "explanation": "Same frozen v88 old combined answer; Rv2 double-rank is idempotent and verified per target after the known-private mask; Rv1 selected AIA copies that same v88 answer."}
        elif group == "pl1":
            if pilot:
                raise ValueError("pl1 joint transport requires all twenty targets; pilot is shared only")
            from prelim_transfer import methods
            modules = methods.load_legacy()
            targets = {k: item["release"] for k, item in tables["targets"].items()}
            challenges = {k: item["challenge"] for k, item in tables["targets"].items()}
            mia, _, md = methods.build_mia(modules, tables["candidates"], tables["background"], targets,
                                           tables["self_challenge"], tables["schema"])
            variants, ad = methods.build_aia(modules, tables["background"], targets, challenges)
            mia = mia.rename(columns={k: str(k) for k in targets})
            mias = {"pl1": mia[["record_id", *map(str, sorted(targets))]]}
            aias = {"pl1": variants["structured"][["target_k", "challenge_row_id", "time_hat"]]}
            diagnostics = {"mia": md, "aia": ad}
            hashes = methods.source_hashes()
        else:
            raise ValueError("Unknown prediction group")
    if access["blocked"]:
        raise RuntimeError("A prediction dependency suppressed a forbidden read")
    if verify_public(output) != manifest or any(sha(ROOT / n) != h for n, h in frozen.items()):
        raise ValueError("Public inputs or frozen dependencies changed during inference")
    provenance = {"round": "preliminary", "attacker_id": 24, "target_count": 1 if pilot else 20,
                  "main_algorithm_official_origin": origin, "adapter_sha256": sha(__file__),
                  "public_manifest_sha256": sha(public / "bundle_manifest.json"),
                  "known_nonshared_proof": proof, "target_truth_reads_blocked": access["blocked"],
                  "historical_training_exposure": True, "independent_population": False,
                  "new_hyperparameter_selection": False, "frozen_source_sha256": hashes,
                  "main_dependency_manifest_sha256": sha(ATTACK / "Attack_R/v2/production_dependencies.json"),
                  "seconds": time.monotonic() - started, "diagnostics": diagnostics}
    write_json(work / "inference.json", provenance)
    if not pilot:
        for name in mias:
            _save_prediction(output, name, mias[name], aias[name], tables, provenance)
    print(json.dumps({"group": group, "pilot": pilot, "seconds": provenance["seconds"]}), flush=True)


def score(output):
    # This fresh process intentionally never initializes the main algorithm runtime.
    sys.path.insert(0, str(MOCK))
    from mock_TS_using_prelim.common import verify_bundle
    from mock_TS_using_prelim.scoring import score_predictions
    import pandas as pd
    verify_public(output)
    before = verify_bundle()
    destination = output / "scores"
    destination.mkdir(exist_ok=False)
    rows = []
    for model in MODELS:
        result = score_predictions(output / "predictions" / model, attacker_id=24)
        result.update(model_id=model, evaluation_role="exposed_preliminary_transfer_diagnostic",
                      independent_population=False, main_official_rank_validation=False)
        write_json(destination / f"{model}.json", result)
        pd.DataFrame(result["by_target"]).to_csv(destination / f"{model}_by_target.csv", index=False)
        rows.append({"model_id": model, **result["summary"]})
    if verify_bundle() != before:
        raise ValueError("Preliminary evidence changed during scoring")
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False)
    write_json(destination / "summary.json", {"models": rows, "no_main_round_floor": True,
               "historical_training_exposure": True, "main_official_rank_validation": False})
    print(json.dumps(rows, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "predict", "score", "run", "_prepare", "_pilot", "_predict", "_score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("shared", "pl1"), default="shared")
    args = parser.parse_args(argv)
    output = output_path(args.output)
    if args.action == "_prepare": prepare(output)
    elif args.action == "_pilot": predict(output, "shared", pilot=True)
    elif args.action == "_predict": predict(output, args.group)
    elif args.action == "_score": score(output)
    elif args.action == "run":
        _launch("_prepare", output)
        _launch("_predict", output, group="shared")
        _launch("_predict", output, group="pl1")
        _launch("_score", output)
    else:
        _launch("_" + args.action, output, group=args.group if args.action == "predict" else None)


if __name__ == "__main__":
    main()
