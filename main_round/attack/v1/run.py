"""Generate, independently repeat, and validate actual v1 predictions locally."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attack_workspace import (ATTACK_TARGETS, ORIGINALS, TARGETS, load_inputs,
                              load_runtime, validate_frames, verify_hashes)
from v1.actual import apply_private_mask, derive_private_mask


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local-output/v1/predictions")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "local-output/v1").resolve()):
        raise ValueError("Output must be within local-output/v1")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty; preserve each generation and validation record")
    verified = verify_hashes()
    np, pd, modules, cfg, versions = load_runtime()
    from v1.model import predict
    selection_path = Path(__file__).with_name("selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected_variant"]
    model_path = Path(__file__).with_name("model.py")
    if digest_bytes(model_path.read_bytes()) != selection["model_sha256"]:
        raise ValueError("Model changed since development selection; revalidate before generating predictions")
    validator = modules["pwscup2026.kit.validate"]
    # Keep the wrapper's full distribution checks, then parse numeric inputs
    # with the same round-trip reader used by the official scoring loader.
    _, _, _, _, dist = load_inputs(np, pd, validator)
    read = lambda path: pd.read_csv(path, float_precision="round_trip")
    b, background = read(ORIGINALS / "B.csv"), read(ORIGINALS / "A_bg.csv")
    targets = {k: read(TARGETS / f"C_{k}.csv") for k in ATTACK_TARGETS}
    challenges = {k: read(TARGETS / f"aia_challenge_{k}.csv") for k in ATTACK_TARGETS}
    schema = json.loads((ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))
    private, private_report = derive_private_mask(
        pd.read_csv(ORIGINALS / "B.csv", dtype=str),
        pd.read_csv(ORIGINALS / "B_self.csv", dtype=str),
        pd.read_csv(TARGETS / "aia_challenge_10.csv", dtype=str),
        schema["aia_qi_columns"], schema["submitted_c_columns"])
    bs = read(ORIGINALS / "B_self.csv").set_index("record_id").loc[b.record_id]
    rare = bs.is_rare.to_numpy(bool)

    def predict_public(c, challenge):
        return predict(c=c.copy(deep=True), background=background.copy(deep=True),
                       candidates=b.copy(deep=True), challenge=challenge.copy(deep=True),
                       cfg=copy.deepcopy(cfg), variant=selected)

    def predict_all(with_diagnostics):
        mia = pd.DataFrame({"record_id": b.record_id.to_numpy()})
        aia_parts, diagnostics = [], []
        for k in ATTACK_TARGETS:
            result = predict_public(targets[k], challenges[k])
            raw = np.asarray(result["mia"], dtype=float)
            masked = apply_private_mask(raw, private)
            if not np.array_equal(np.unique(raw[~private], return_inverse=True)[1],
                                  np.unique(masked[~private], return_inverse=True)[1]):
                raise AssertionError("Private-mask transformation altered unknown ranks or ties")
            mia[str(k)] = masked
            aia_parts.append(pd.DataFrame({"target_k": k,
                "challenge_row_id": challenges[k].challenge_row_id.to_numpy(),
                "time_hat": result["aia"]}))
            if with_diagnostics:
                # Head counts are descriptive, not official false-positive rates.
                cutoff = np.sort(raw)[-11]
                rare_cutoff = raw[rare].max()
                diagnostics.append({"target": k,
                    "known_private_in_top11_including_ties_before": int((private & (raw >= cutoff)).sum()),
                    "known_private_at_rare_max_before": int((private & rare & (raw >= rare_cutoff)).sum()),
                    "known_private_nonzero_after": int(np.count_nonzero(masked[private])),
                    "unique_mia_scores": int(len(np.unique(masked))),
                    "aia_time_min": float(np.min(result["aia"])),
                    "aia_time_max": float(np.max(result["aia"]))})
        return mia, pd.concat(aia_parts, ignore_index=True), diagnostics

    mia, aia, diagnostics = predict_all(True)
    checks = validate_frames(np, pd, validator, mia, aia, cfg, dist)
    content = {"F_mia.csv": csv_bytes(mia), "F_aia.csv": csv_bytes(aia)}
    for name, original in (("F_mia.csv", mia), ("F_aia.csv", aia)):
        restored = pd.read_csv(io.BytesIO(content[name]), float_precision="round_trip")
        pd.testing.assert_frame_equal(original, restored, check_dtype=False, check_exact=True)
    validate_frames(np, pd, validator,
                    pd.read_csv(io.BytesIO(content["F_mia.csv"]), float_precision="round_trip"),
                    pd.read_csv(io.BytesIO(content["F_aia.csv"]), float_precision="round_trip"), cfg, dist)
    repeat_mia, repeat_aia, _ = predict_all(False)
    if content["F_mia.csv"] != csv_bytes(repeat_mia) or content["F_aia.csv"] != csv_bytes(repeat_aia):
        raise AssertionError("Two independent full prediction passes differ")
    # Sensitivity diagnostics do not select a model using unknown actual labels.
    c3 = targets[3]
    cols = schema["submitted_c_columns"]
    bg_keys = set(background[cols].itertuples(index=False, name=None))
    bg_rows = np.array([row in bg_keys for row in c3[cols].itertuples(index=False, name=None)])
    normal_mia = mia["3"].to_numpy()
    normal_aia = aia.loc[aia.target_k == 3, "time_hat"].to_numpy()
    sensitivity = []
    for label, retain in (("without_last_row", np.arange(len(c3)) != len(c3) - 1),
                          ("without_exact_background_rows", ~bg_rows),
                          ("without_both", (~bg_rows) & (np.arange(len(c3)) != len(c3) - 1))):
        p = predict_public(c3.loc[retain].reset_index(drop=True), challenges[3])
        adjusted = apply_private_mask(p["mia"], private)
        original_head = normal_mia >= np.sort(normal_mia)[-11]
        changed_head = adjusted >= np.sort(adjusted)[-11]
        sensitivity.append({"variant": label, "rows_removed": int((~retain).sum()),
            "max_mia_change": float(np.max(np.abs(adjusted - normal_mia))),
            "top11_membership_changed_including_ties": int((original_head != changed_head).sum()),
            "aia_predictions_changed": int(np.count_nonzero(p["aia"] != normal_aia)),
            "aia_changes_greater_than_half_year": int((np.abs(p["aia"] - normal_aia) > .5).sum())})
    verify_hashes()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": selected, "selection": selection,
        "input_verification": verified, "dependency_versions": versions,
        "private_inference": private_report, "official_format_validation": checks,
        "csv_roundtrip_exact": True, "two_complete_runs_byte_identical": True,
        "known_private_scores_zero_for_all_22": bool((mia.drop(columns="record_id").to_numpy()[private] == 0).all()),
        "unknown_rank_and_ties_preserved_for_all_22": True,
        "per_target": diagnostics, "c3_sensitivity_only": sensitivity,
        "c3_decision": "Retain all original rows. Sensitivity has no truth labels and does not establish a benefit of removal.",
        "output_sha256": {name: digest_bytes(data) for name, data in content.items()},
        "v1_source_sha256": {path.name: digest_bytes(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "official_score_measured": False, "official_submission_performed": False,
        "submission_zip_created": False, "github_push_performed": False}
    output.mkdir(parents=True, exist_ok=True)
    # Preserve existing predictions if re-run with a different candidate.
    for name, data in content.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different predictions: {path.name}")
    for name, data in content.items():
        (output / name).write_bytes(data)
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation": checks,
                      "byte_identical_repeat": True, "known_private": int(private.sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
