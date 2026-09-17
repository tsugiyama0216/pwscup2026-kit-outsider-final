"""Verify user-supplied feedback against the frozen ZIP; never regenerate S1.

Requires the existing scoring environment and pinned official kit.
Only aggregate audit outputs in this directory are written. No token is logged.
"""
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import sys
import zipfile

HERE = Path(__file__).resolve().parent
S1 = HERE.parents[1] / "S1"
TEAM = S1.parents[1]
DATA = TEAM / "participant_data"
COMMIT = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
KIT = S1 / "local-output" / "official-kit" / f"pwscup2026-kit-{COMMIT}"
SOURCE = KIT / "starter" / "src"
DIST = S1 / "local-output" / "selfscore-dist"
SUBMISSION = S1 / "submission" / "team10_S1_defense.zip"
TOL = 1e-12


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (HERE / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


frozen_paths = [S1 / name for name in (
    "C.csv", "score.json", "validation.json", "privacy_report.json",
    "release_ledger.json", "config/aim_main_team10_S1.json",
    "scripts/generate_pwscup.py", "submission/team10_S1_defense.zip",
)]
before = {str(p.relative_to(S1)): digest(p) for p in frozen_paths}
provenance = json.loads((S1 / "official_kit_provenance.json").read_text(encoding="utf-8"))
archive_path = KIT.parent / "main-process-20260912.zip"
assert provenance["commit"] == COMMIT
assert digest(archive_path) == provenance["archive_sha256"]
source_count = 0
with zipfile.ZipFile(archive_path) as archive:
    for member in archive.infolist():
        rel = member.filename.removeprefix(f"pwscup2026-kit-{COMMIT}/")
        if ((rel.startswith("starter/src/pwscup2026/") and rel.endswith((".py", ".yaml")))
                or rel == "participant_data/scoring_config.yaml"):
            assert digest(KIT / rel) == sha256(archive.read(member)).hexdigest(), rel
            source_count += 1
assert source_count > 20

manifest = json.loads((TEAM / "data_manifest.json").read_text(encoding="utf-8"))
assert all(digest(TEAM / item["path"]) == item["sha256"] for item in manifest["files"])
reference_hashes = {}
for name in ("B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json"):
    assert digest(DIST / name) == digest(DATA / name), name
    reference_hashes[name] = digest(DATA / name)
assert digest(DIST / "scoring_config.yaml") == digest(KIT / "participant_data/scoring_config.yaml")

with zipfile.ZipFile(SUBMISSION) as archive:
    assert sorted(archive.namelist()) == ["C.csv", "token.txt"]
    assert archive.read("C.csv") == (S1 / "C.csv").read_bytes()
    # Deliberately retain only the boolean; never token contents or token hashes.
    token_match = archive.read("token.txt") == (DATA / "token.txt").read_bytes()
    assert token_match

sys.path.insert(0, str(SOURCE))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pwscup2026.kit import selfscore
from pwscup2026.common.submission_schema import validate_c_submission
from pwscup2026.scoring.result import json_safe

assert Path(selfscore.__file__).resolve().is_relative_to(SOURCE)
c = selfscore.load_submitted_c(SUBMISSION)
b, ref, cfg, info = selfscore.load_local_reference(DIST)
pd.testing.assert_frame_equal(b, selfscore.read_scoring_csv(DATA / "B.csv"))
assert info["b_self"] and info["a_bg"]
validation = validate_c_submission(c, cfg, n_expected=len(b))
assert not validation.errors

score = json_safe(selfscore.self_score(c, b, ref, cfg))
write_json("rescored_zip.json", score)
feedback = json.loads((HERE / "user_feedback.json").read_text(encoding="utf-8"))
previous = json.loads((S1 / "score.json").read_text(encoding="utf-8"))
comparisons = {}
for facet, official in feedback["utility_facets"].items():
    local = score["facets"][facet]
    comparisons[facet] = {
        "user_supplied_official": official, "rescored_zip": local,
        "previous_local": previous["facets"][facet],
        "absolute_difference_official_local": abs(local - official),
        "matches_at_1e_12": abs(local - official) <= TOL,
    }
    assert comparisons[facet]["matches_at_1e_12"]
    assert abs(local - previous["facets"][facet]) <= TOL
rare = score["detail"]["U_rare"]
gate = {key: rare[key] for key in ("rare_detected_C", "rare_gate", "rare_true_B")}
gate["gated_axes"] = rare["u_rare_gated"]
assert gate == feedback["u_rare_gate"]

# Independent calculation: SciPy Spearman per pair, not scorer's correlation helper.
corr_detail = score["detail"]["U_gen"]["correlation"]
assert corr_detail["method"] == "spearman"
pair_values = []
for left, right in combinations(corr_detail["features"], 2):
    corr_b = float(spearmanr(b[left], b[right]).statistic)
    corr_c = float(spearmanr(c[left], c[right]).statistic)
    pair_values.append({"pair": [left, right], "B": corr_b, "C": corr_c,
                        "absolute_difference": abs(corr_c - corr_b)})
distances = np.array([p["absolute_difference"] for p in pair_values])
assert np.isfinite(distances).all() and len(distances) == 36
mean_dist = float(distances.mean())
tail_dist = float(np.quantile(distances, corr_detail["tail_quantile"]))
assert abs(mean_dist - corr_detail["observed_mean_dist"]) <= TOL
assert abs(tail_dist - corr_detail["observed_tail_dist"]) <= TOL
mean_score = float(np.clip(1 - mean_dist / corr_detail["mean_scale"], 0, 1))
tail_score = float(np.clip(1 - tail_dist / corr_detail["tail_scale"], 0, 1))
assert min(mean_score, tail_score) == score["detail"]["U_gen"]["corr_score"] == 0
parts = {key: score["detail"]["U_gen"][key] for key in (
    "marginal_mean", "corr_score", "pmse_score", "time_dz", "tail_score")}
assert min(parts.values()) == score["facets"]["U_gen"]
assert min(score["facets"].values()) == score["utility"] == 0

# Positive control for the scorer. B is scored in memory, never written as C.
# This is not a submission candidate or privacy evaluation.
positive = json_safe(selfscore.self_score(b.loc[:, c.columns], b, ref, cfg))
assert positive["utility"] >= 1 - TOL
after = {str(p.relative_to(S1)): digest(p) for p in frozen_paths}
assert before == after

audit = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "feedback_provenance": "User pasted in chat on 2026-09-17; no server job ID/log/remote artifact provided",
    "official_release": "main-process-20260912", "official_commit": COMMIT,
    "source_files_verified_against_release_archive": source_count,
    "archive_sha256": digest(archive_path),
    "reference_hashes": reference_hashes,
    "original_manifest_matches": True, "B_self_without_rare_equals_B": True,
    "frozen_s1_sha256": before, "frozen_s1_unchanged": True,
    "package": {"members": ["C.csv", "token.txt"], "C_exact_bytes_match": True,
                "token_matches_team10_source": token_match, "token_or_token_hash_recorded": False,
                "rows": len(c), "columns": len(c.columns), "validation_errors": validation.errors},
    "facet_comparison": comparisons, "rare_gate_matches": True, "rare_gate": gate,
    "independent_scipy_correlation": {
        "pairs": pair_values, "mean_absolute_difference": mean_dist,
        "q90_absolute_difference": tail_dist, "mean_score": mean_score,
        "tail_score": tail_score, "correlation_score": min(mean_score, tail_score)},
    "U_gen_components": parts,
    "positive_control": {"description": "B as C in memory only; scorer health check, not a submission",
                         "utility": positive["utility"], "facets": positive["facets"]},
    "reference_attack": {"recomputed": False,
                         "reason": "Server MIA candidate set and membership ground truth are not in team10 processing inputs"},
    "conclusion": "U=0 reproduced from frozen submission ZIP and independently traced to correlation error exceeding mean scale 0.4",
}
write_json("verification.json", audit)
print(json.dumps({"facets": comparisons, "rare_gate": gate,
                  "mean_corr_distance": mean_dist, "q90_corr_distance": tail_dist,
                  "U_gen_components": parts, "positive_control": audit["positive_control"],
                  "frozen_s1_unchanged": True}, ensure_ascii=False, indent=2))
