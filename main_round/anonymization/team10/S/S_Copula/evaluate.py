"""Private local diagnostics. This output is NOT a differentially private release.

Evaluation is isolated from dp_copula.py. Never select a synthetic release,
random seed, or postprocessing parameter by these private scores and then claim
that selection is covered by the generator's epsilon.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[1]
REPO = HERE.parents[4]
KIT = REPO.parent / "official-kit"
PIN = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
DATA = TEAM / "participant_data"
COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment(kit=KIT):
    kit = Path(kit).resolve()
    git = ["git", "-c", f"safe.directory={kit.as_posix()}", "-C", str(kit)]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != PIN or dirty:
        raise RuntimeError("Official kit must be unchanged at the recorded commit")
    sys.path.insert(0, str(kit / "starter/src"))
    import pwscup2026
    if not Path(pwscup2026.__file__).resolve().is_relative_to(kit):
        raise RuntimeError("Unexpected scoring package")
    cfg_path = kit / "starter/src/pwscup2026/kit/kit_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return cfg, {"kit_commit": head, "config_sha256": sha256(cfg_path),
                 "dependencies": {k: importlib.metadata.version(k) for k in
                 ("numpy", "pandas", "scipy", "scikit-learn", "lifelines", "statsmodels", "PyYAML")}}

def reference(data=DATA):
    from pwscup2026.common.submission_schema import read_scoring_csv
    from pwscup2026.scoring.utility import UtilityReference
    data = Path(data)
    b = read_scoring_csv(data / "B.csv")
    bs = read_scoring_csv(data / "B_self.csv")
    pd.testing.assert_frame_equal(b, bs.drop(columns="is_rare"))
    ref = json.loads((data / "utility_ref.json").read_text(encoding="utf-8"))
    return b, UtilityReference(mu=np.array(ref["mu"]), sigma=np.array(ref["sigma"]),
        band=ref["band"], b_is_rare=bs.is_rare.to_numpy(bool),
        km_signal_D=ref.get("km_signal_D"), km_floor=ref.get("km_floor"),
        a_bg=read_scoring_csv(data / "A_bg.csv"))

def evaluate_csv(path, data=DATA, kit=KIT):
    cfg, provenance = environment(kit)
    from pwscup2026.kit.selfscore import load_submitted_c, self_score
    from pwscup2026.common.submission_schema import validate_c_submission
    from pwscup2026.kit.validate import rare_count_diagnostic
    from pwscup2026.scoring.result import json_safe
    b, ref = reference(data)
    c = load_submitted_c(path)
    validation = validate_c_submission(c, cfg, n_expected=len(b))
    if list(c.columns) != COLUMNS:
        raise ValueError("Unexpected columns or order")
    if not validation.ok:
        raise ValueError(str(validation.errors))
    score = self_score(c, b, ref, cfg)
    schema = json.loads((Path(data) / "schema.json").read_text(encoding="utf-8"))
    provenance["C_sha256"] = sha256(path)
    provenance["input_hashes"] = {name: sha256(Path(data) / name) for name in
        ("B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json")}
    return json_safe({"score": score, "provenance": provenance,
        "validation": {"logical_ok": True, "rows": len(c),
            "no_missing": bool(not c.isna().any().any()),
            "duplicate_rows": int(c.duplicated().sum()),
            "exact_B_matches": int(c.merge(b[COLUMNS].drop_duplicates(), on=COLUMNS).shape[0]),
            "rare": rare_count_diagnostic(c, schema)},
        "scope": "Non-DP local evaluation; not official server protection or overall score",
        "used_for_release_selection": False})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--kit", type=Path, default=KIT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite diagnostics")
    result = evaluate_csv(args.csv, args.data, args.kit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "utility": result["score"]["utility"], "facets": result["score"]["facets"]}))

if __name__ == "__main__":
    main()
