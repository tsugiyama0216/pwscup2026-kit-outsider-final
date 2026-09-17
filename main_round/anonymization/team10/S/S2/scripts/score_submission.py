"""Run the pinned main-round official scorer on the frozen S2 release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
COMMIT = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
DEFAULT_KIT = MODEL.parent / "S1" / "local-output" / "official-kit" / f"pwscup2026-kit-{COMMIT}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT)
    args = parser.parse_args()
    kit = args.kit_root.resolve()
    source_provenance = json.loads((MODEL / "official_kit_provenance.json").read_text(encoding="utf-8"))
    archive_path = kit.parent / "main-process-20260912.zip"
    if source_provenance["commit"] != COMMIT or digest(archive_path) != source_provenance["archive_sha256"]:
        raise ValueError("Official source archive differs from the pinned main-round release")
    prefix = f"pwscup2026-kit-{COMMIT}/"
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            relative = item.filename.removeprefix(prefix)
            if ((relative.startswith("starter/src/pwscup2026/") and relative.endswith((".py", ".yaml")))
                    or relative == "participant_data/scoring_config.yaml"):
                if digest(kit / relative) != hashlib.sha256(archive.read(item)).hexdigest():
                    raise ValueError(f"Official scoring source changed: {relative}")
    source = kit / "starter" / "src"
    sys.path.insert(0, str(source))
    from pwscup2026.kit import selfscore
    from pwscup2026.scoring.result import json_safe

    if not Path(selfscore.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Scorer was not imported from the requested main-round kit")
    data = MODEL.parents[1] / "participant_data"
    staging = MODEL / "local-output" / "selfscore-dist"
    staging.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data.parent / "data_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if digest(data.parent / item["path"]) != item["sha256"]:
            raise ValueError(f"Original input changed: {item['path']}")
    for name in ("B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json"):
        shutil.copyfile(data / name, staging / name)
        if digest(data / name) != digest(staging / name):
            raise ValueError(f"Scoring reference copy differs: {name}")
    scoring_config = kit / "participant_data" / "scoring_config.yaml"
    shutil.copyfile(scoring_config, staging / "scoring_config.yaml")
    c_path = MODEL / "C.csv"
    report = json.loads((MODEL / "privacy_report.json").read_text(encoding="utf-8"))
    if digest(c_path) != report["output_sha256"]:
        raise ValueError("C.csv differs from the frozen release")
    from package_submission import check_lineage, check_originals, verify_zip
    check_lineage(MODEL, check_originals(data.parent))
    submission_path = MODEL / "submission" / "team10_S2_defense.zip"
    verify_zip(submission_path, c_path.read_bytes(), (data / "token.txt").read_bytes())
    c = selfscore.load_submitted_c(submission_path)
    b, reference, cfg, info = selfscore.load_local_reference(staging)
    if not info["b_self"] or not info["a_bg"] or len(b) != len(c):
        raise ValueError("Full main-round self-score references are required")
    import pandas as pd
    pd.testing.assert_frame_equal(b, selfscore.read_scoring_csv(data / "B.csv"))
    result = selfscore.self_score(c, b, reference, cfg)
    rare = selfscore.validate_mod.rare_count_diagnostic(
        c, selfscore.validate_mod.load_dist(staging)
    )
    score = json_safe({**result, "b_self": info["b_self"], "a_bg": info["a_bg"], "rare_gate": rare})
    (MODEL / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    provenance = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local utility only; no CodaBench submission, anonymity or overall score",
        "official_release": "main-process-20260912",
        "official_commit": COMMIT,
        "c_sha256": digest(c_path),
        "submission_zip_sha256": digest(submission_path),
        "scored_input": "submission/team10_S2_defense.zip",
        "score_sha256": digest(MODEL / "score.json"),
        "runner_sha256": digest(Path(__file__)),
        "scoring_config_sha256": digest(scoring_config),
        "reference_hashes": {name: digest(staging / name) for name in ("B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json")},
        "source_hashes": {str(p.relative_to(source)).replace("\\", "/"): digest(p) for p in sorted((source / "pwscup2026").rglob("*.py"))},
        "python_version": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lifelines", "matplotlib", "PyYAML")},
    }
    (MODEL / "evaluation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"utility": score["utility"], "facets": score["facets"], "rare_gate": rare}, ensure_ascii=False))


if __name__ == "__main__":
    main()
