"""Run one frozen component in its own Python process on actual contest inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from common import ATTACK, HERE, csv_bytes, load_actual, read_json, save_json, sha, verify_dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("mia", "aia"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((HERE / ".work").resolve()):
        raise ValueError("Intermediate output must remain under R v1/.work")
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    dependencies = verify_dependencies()
    if args.component == "aia":
        sys.path.insert(0, str(ATTACK / "Attak_T"))
        import attack_research
        from attack_research.published_v88.check import verify_package
        published = verify_package()
    from model import predict_mia_all, predict_aia_one
    from threadpoolctl import threadpool_limits

    w, np, pd, validator, cfg, dist, b, background, targets, challenges, originals, versions = load_actual()
    report = {"component": args.component, "source_checks": originals,
        "dependencies": dependencies, "dependency_versions": versions,
        "team_id": 10, "target_order": sorted(targets)}
    with threadpool_limits(limits=1):
        if args.component == "mia":
            from M.v1.actual import derive_private_mask
            from M.v2.artifacts import load_selection, check_v1_preserved
            selection = load_selection()
            final = read_json(ATTACK / "Attack_M/local-output/v2/final_check.json")
            if not final["completed"] or final["selection_sha256"] != sha(ATTACK / "Attack_M/v2/selection.json"):
                raise ValueError("M v2 final selection verification failed")
            schema = read_json(w.ORIGINALS / "schema.json")
            private, proof = derive_private_mask(
                pd.read_csv(w.ORIGINALS / "B.csv", dtype=str),
                pd.read_csv(w.ORIGINALS / "B_self.csv", dtype=str),
                pd.read_csv(w.TARGETS / "aia_challenge_10.csv", dtype=str),
                schema["aia_qi_columns"], schema["submitted_c_columns"])
            keys, values = predict_mia_all(targets=targets, background=background,
                candidates=b, private_mask=private, strength=selection["mia_strength"])
            frame = pd.DataFrame({"record_id": b.record_id.to_numpy(),
                **{str(k): values[:, i] for i, k in enumerate(keys)}})
            if not (values[private] == 0).all():
                raise AssertionError("Known non-shared candidates must remain zero")
            data = csv_bytes(frame)
            reference = ATTACK / "Attack_M/local-output/v2/predictions/F_mia.csv"
            if data != reference.read_bytes():
                raise AssertionError("Regenerated MIA differs from the frozen M v2 CSV")
            name = "F_mia.csv"
            report.update(private_inference=proof, mia_strength=selection["mia_strength"],
                identical_to_frozen_M_v2=True, v1_preservation=check_v1_preserved())
        else:
            from attack_research.v88.predict import CombinedAttack
            attack = CombinedAttack()
            frames, routes = [], {}
            for k in sorted(targets):
                times, inferred = predict_aia_one(attack, c=targets[k], background=background,
                    candidates=b, challenge=challenges[k], cfg=cfg)
                frames.append(pd.DataFrame({"target_k": k,
                    "challenge_row_id": challenges[k].challenge_row_id.to_numpy(), "time_hat": times}))
                routes[str(k)] = inferred
                print(f"T v88: target {k} completed ({len(times)} answers; route {inferred})", flush=True)
            frame = pd.concat(frames, ignore_index=True)
            data, name = csv_bytes(frame), "F_aia.csv"
            report.update(published_v88=published, routes=routes,
                original_v88_called_without_algorithm_changes=True)
    (destination / name).write_bytes(data)
    verify_dependencies()
    w.verify_hashes()
    report.update(rows=len(frame), csv_sha256=sha(destination / name),
        elapsed_seconds=perf_counter() - started, completed=True)
    save_json(destination / "report.json", report)
    print(f"{args.component.upper()} complete: {len(frame)} rows", flush=True)


if __name__ == "__main__":
    main()
