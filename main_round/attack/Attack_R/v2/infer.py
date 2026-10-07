"""One deterministic actual-input inference; output contains no credentials."""
import argparse
from pathlib import Path
import runtime as rt
import pandas as pd
from threadpoolctl import threadpool_limits
from model import compute_world


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(rt.OUTPUT.resolve()/"production") or out.exists():
        raise ValueError("Use a fresh R-v2/production output directory")
    selected = rt.selected_check()
    before = rt.upstream_check()
    common = rt.load_file("_rv2_original_common", rt.HERE.parent/"v1/common.py")
    actual = common.load_actual()
    w, np, _, validator, cfg, dist, b, bg, releases, challenges, originals, versions = actual
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    public = {"candidates": b, "self_labels": read(w.ORIGINALS/"B_self.csv"), "background": bg,
        "self_challenge": read(w.TARGETS/"aia_challenge_10.csv"),
        "targets": {k: {"release": releases[k], "challenge": challenges[k]} for k in releases},
        "schema": rt.read_json(w.ORIGINALS/"schema.json"), "cfg": cfg}
    with threadpool_limits(limits=1):
        all_answers = compute_world(**public)
    # An end-to-end check that the matched baselines really reproduce the
    # unchanged, archived R v1 and v2f submissions on actual inputs.
    for version, mia_option in (("v1", "R_v1"), ("v2f", "T")):
        for name, frame in (("F_mia.csv", all_answers["mia"][mia_option]),
                            ("F_aia.csv", all_answers["aia"]["T"])):
            if rt.csv_bytes(frame) != (rt.HERE.parent/version/"submission"/name).read_bytes():
                raise ValueError("Baseline reproduction differs: " + version + "/" + name)
    result = {"F_mia": all_answers["mia"][selected["mia"]],
              "F_aia": all_answers["aia"][selected["aia"]], "diagnostics": all_answers["diagnostics"]}
    valid = w.validate_frames(np, pd, validator, result["F_mia"], result["F_aia"], cfg, dist)
    if before != rt.upstream_check():
        raise ValueError("Upstream changed")
    rt.selected_check()
    out.mkdir(parents=True)
    for name in ("F_mia", "F_aia"):
        (out/(name+".csv")).write_bytes(rt.csv_bytes(result[name]))
    rt.save_json(out/"report.json", {"completed": True, "validation": valid,
        "diagnostics": result["diagnostics"], "originals": originals, "versions": versions,
        "actual_R_v1_and_v2f_baselines_byte_identical": True,
        "selection_sha256": rt.sha(rt.HERE/"selection.json"), "upstream": before,
        "file_sha256": {name: rt.sha(out/name) for name in ("F_mia.csv", "F_aia.csv")}})


if __name__ == "__main__":
    main()
