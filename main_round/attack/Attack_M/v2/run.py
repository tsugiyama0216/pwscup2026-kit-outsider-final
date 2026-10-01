"""Create M's v2 contest CSVs locally, repeat them, and validate actual IDs."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import attack_workspace as w
from M.v1.actual import derive_private_mask
from M.v1.run import csv_bytes
from M.v2.artifacts import (OUTPUT, HERE, load_selection, read_json, save_json, sha,
                            check_v1_preserved, timestamp)


def build_frames(result, b, challenges, pd):
    mia = pd.DataFrame({"record_id": b.record_id.to_numpy()})
    for i, k in enumerate(result["target_order"]):
        mia[str(k)] = result["mia"][:, i]
    aia = pd.concat([pd.DataFrame({"target_k": k,
        "challenge_row_id": challenges[k].challenge_row_id.to_numpy(),
        "time_hat": result["aia"][k]}) for k in result["target_order"]], ignore_index=True)
    return mia, aia


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "predictions")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Output must remain inside M/local-output/v2")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous predictions; choose an empty directory")
    selection = load_selection()
    final_check = read_json(OUTPUT / "final_check.json")
    if not final_check["completed"] or final_check["selection_sha256"] != sha(HERE / "selection.json"):
        raise ValueError("The selected method has not completed final checking")
    preservation = check_v1_preserved()
    sources_before = w.verify_hashes()
    np, pd, modules, cfg, versions = w.load_runtime()
    from threadpoolctl import threadpool_limits
    from M.v2.model import predict_all
    from M.v2.aia import diagnose
    validator = modules["pwscup2026.kit.validate"]
    _, _, _, _, dist = w.load_inputs(np, pd, validator)
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    b = read(w.ORIGINALS / "B.csv")
    background = read(w.ORIGINALS / "A_bg.csv")
    targets = {k: read(w.TARGETS / f"C_{k}.csv") for k in w.ATTACK_TARGETS}
    challenges = {k: read(w.TARGETS / f"aia_challenge_{k}.csv") for k in w.ATTACK_TARGETS}
    schema = read_json(w.ORIGINALS / "schema.json")
    private, private_report = derive_private_mask(
        pd.read_csv(w.ORIGINALS / "B.csv", dtype=str),
        pd.read_csv(w.ORIGINALS / "B_self.csv", dtype=str),
        pd.read_csv(w.TARGETS / "aia_challenge_10.csv", dtype=str),
        schema["aia_qi_columns"], schema["submitted_c_columns"])
    baseline_dir = ROOT / "M/local-output/v1/predictions"
    baseline_mia = read(baseline_dir / "F_mia.csv").set_index("record_id").loc[b.record_id]
    baseline_aia = read(baseline_dir / "F_aia.csv").set_index(["target_k", "challenge_row_id"])

    def predict():
        with threadpool_limits(limits=1):
            result = predict_all(targets=targets, background=background,
                candidates=b, challenges=challenges, cfg=cfg, private_mask=private,
                aia_variant=selection["aia_variant"], mia_strength=selection["mia_strength"])
        if not (result["mia"][private] == 0).all():
            raise AssertionError("Known private rows must remain zero for every target")
        return build_frames(result, b, challenges, pd)

    mia, aia = predict()
    validation = w.validate_frames(np, pd, validator, mia, aia, cfg, dist)
    content = {"F_mia.csv": csv_bytes(mia), "F_aia.csv": csv_bytes(aia)}
    for name, original in (("F_mia.csv", mia), ("F_aia.csv", aia)):
        restored = pd.read_csv(io.BytesIO(content[name]), float_precision="round_trip")
        pd.testing.assert_frame_equal(original, restored, check_dtype=False, check_exact=True)
    w.validate_frames(np, pd, validator,
        pd.read_csv(io.BytesIO(content["F_mia.csv"]), float_precision="round_trip"),
        pd.read_csv(io.BytesIO(content["F_aia.csv"]), float_precision="round_trip"), cfg, dist)
    repeat_mia, repeat_aia = predict()
    if csv_bytes(repeat_mia) != content["F_mia.csv"] or csv_bytes(repeat_aia) != content["F_aia.csv"]:
        raise AssertionError("Independent complete repeats were not byte-identical")
    per_target = []
    for k in w.ATTACK_TARGETS:
        old = baseline_mia[str(k)].to_numpy()
        new = mia[str(k)].to_numpy()
        hats = aia.loc[aia.target_k == k, "time_hat"].to_numpy()
        ids = challenges[k].challenge_row_id.to_numpy()
        old_hats = baseline_aia.loc[[(k, i) for i in ids], "time_hat"].to_numpy()
        old_head, new_head = old >= np.sort(old)[-11], new >= np.sort(new)[-11]
        routing = diagnose(targets[k], background, challenges[k], cfg)
        routes = routing["routes"][selection["aia_variant"]]
        if selection["mia_strength"] == 0. and not np.array_equal(old, new):
            raise AssertionError("Unadjusted MIA must exactly reproduce saved v1")
        if routes["reference"] == len(hats) and not np.array_equal(hats, old_hats):
            raise AssertionError("Unchanged AIA routing must exactly reproduce saved v1")
        per_target.append({"target": k,
            "mia_changed_values": int(np.count_nonzero(old != new)),
            "mia_top11_symmetric_difference_with_ties": int(np.count_nonzero(old_head != new_head)),
            "known_private_nonzero": int(np.count_nonzero(new[private])),
            "aia_changed_values": int(np.count_nonzero(hats != old_hats)),
            "aia_changes_gt_half_year": int(np.count_nonzero(np.abs(hats - old_hats) > .5)),
            "aia_reference_routes": routes["reference"],
            "aia_sex_only_routes": routes["sex_only"],
            "aia_soft_prefecture_routes": routes["soft_prefecture"],
            "aia_county_sex_support_median": routing["sex_prefecture_support_median"],
            "aia_min": float(hats.min()), "aia_max": float(hats.max())})
    sources_after = w.verify_hashes()
    load_selection()
    if sha(HERE / "selection.json") != final_check["selection_sha256"]:
        raise ValueError("Selection changed during complete prediction generation")
    check_v1_preserved()
    def head_overlap(values):
        head = values >= np.sort(values, axis=0)[-11, :]
        slots = int(head.sum())
        unique_people = int(head.any(axis=1).sum())
        return {"target_person_slots_including_ties": slots,
                "distinct_people": unique_people,
                "nonmember_slots_lower_bound_from_exclusivity": slots - unique_people}
    overlap = {"v1": head_overlap(baseline_mia[[str(k) for k in w.ATTACK_TARGETS]].to_numpy()),
               "v2": head_overlap(mia[[str(k) for k in w.ATTACK_TARGETS]].to_numpy()),
               "interpretation": "Top11 is a descriptive head set, not official FPR. Lower overlap alone does not establish improved accuracy."}
    import hashlib
    report = {"created_utc": timestamp(), "version": "M/v2", "selection": selection,
        "final_check": final_check, "source_checks_before": sources_before,
        "source_checks_after": sources_after, "v1_preservation": preservation,
        "dependency_versions": versions, "private_inference": private_report,
        "official_format_validation": validation, "csv_roundtrip_exact": True,
        "two_complete_runs_byte_identical": True,
        "known_private_scores_zero_for_all_22": True, "per_target": per_target,
        "cross_target_top11_overlap": overlap,
        "output_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in content.items()},
        "official_score_measured": False, "official_submission_performed": False,
        "submission_zip_created": False, "github_push_performed": False}
    output.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        (output / name).write_bytes(data)
    pd.DataFrame(per_target).to_csv(output / "changes_from_v1.csv", index=False)
    save_json(output / "validation.json", report)
    print({"output": str(output), "validation": validation,
           "byte_identical_repeat": True, "known_private": int(private.sum()),
           "aia_changed_gt_half_year": sum(r["aia_changes_gt_half_year"] for r in per_target)})


if __name__ == "__main__":
    main()
