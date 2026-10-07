"""One complete actual-input v2f prediction, using only the frozen selection."""
import argparse
from pathlib import Path
import sys
import runtime as rt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from attack_research.v88.predict import CombinedAttack
from M.v1.actual import derive_private_mask
from model import predict_selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(rt.HERE/".work") or (out.exists() and any(out.iterdir())):
        raise ValueError("Use a new .work subdirectory")
    out.mkdir(parents=True, exist_ok=True)
    selected = rt.selected_check()
    before = rt.upstream_check()
    sys.path.insert(0, str(rt.HERE.parent/"v1"))
    from common import load_actual
    w, _, _, validator, cfg, dist, b, bg, targets, challenges, originals, versions = load_actual()
    schema = rt.read_json(w.ORIGINALS/"schema.json")
    mask, proof = derive_private_mask(pd.read_csv(w.ORIGINALS/"B.csv", dtype=str),
        pd.read_csv(w.ORIGINALS/"B_self.csv", dtype=str),
        pd.read_csv(w.TARGETS/"aia_challenge_10.csv", dtype=str),
        schema["aia_qi_columns"], schema["submitted_c_columns"])
    attack = CombinedAttack()
    mia = pd.DataFrame({"record_id": b.record_id.to_numpy()})
    aia = []
    routes = {}
    with threadpool_limits(limits=1):
        for k in sorted(targets):
            answers, route = predict_selected(attack, selected, c=targets[k], background=bg,
                candidates=b, challenge=challenges[k], cfg=cfg, private_mask=mask)
            mia[str(k)] = answers["mia"]
            aia.append(pd.DataFrame({"target_k": k,
                "challenge_row_id": challenges[k].challenge_row_id.to_numpy(), "time_hat": answers["aia"]}))
            routes[str(k)] = route
            print(f"v2f actual target {k} complete", flush=True)
    aia = pd.concat(aia, ignore_index=True)
    validation = w.validate_frames(np, pd, validator, mia, aia, cfg, dist)
    for name, frame in (("F_mia.csv", mia), ("F_aia.csv", aia)):
        (out/name).write_bytes(rt.csv_bytes(frame))
    assert rt.upstream_check() == before
    rt.selected_check()
    rt.save_json(out/"report.json", {"completed": True, "selection_sha256": rt.sha(rt.HERE/"selection.json"),
        "selected_mia": selected["mia"], "selected_aia": selected["aia"], "validation": validation,
        "private_proof": proof, "routes": routes, "upstream": before, "versions": versions,
        "file_sha256": {name: rt.sha(out/name) for name in ("F_mia.csv", "F_aia.csv")}})


if __name__ == "__main__":
    main()
