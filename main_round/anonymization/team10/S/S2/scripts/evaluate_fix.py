"""Evaluate the frozen S2 release and a fixed-seed, same-DP-model control pair."""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

MODEL = Path(__file__).resolve().parents[1]
S1 = MODEL.parent / "S1"
KIT = S1 / "local-output/official-kit/pwscup2026-kit-3dd71bd3436f2abfdbb64da97e860f22729ea054"
sys.path.insert(0, str(KIT / "starter/src"))
from pwscup2026.kit import selfscore
from pwscup2026.scoring.result import json_safe
from pwscup2026.common.submission_schema import validate_c_submission


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    b, ref, cfg, _ = selfscore.load_local_reference(MODEL / "local-output/selfscore-dist")
    evaluation_provenance = load(MODEL / "evaluation_provenance.json")
    for relative, expected in evaluation_provenance["source_hashes"].items():
        assert digest(KIT / "starter/src" / relative) == expected
    for name, expected in evaluation_provenance["reference_hashes"].items():
        assert digest(MODEL / "local-output/selfscore-dist" / name) == expected
    control_dir = MODEL / "local-output/sampler-comparison"
    control_provenance = load(control_dir / "provenance.json")
    assert control_provenance["additional_dp_measurements"] == 0
    assert control_provenance["model_snapshot_sha256"] == digest(MODEL / "model_snapshot.npz")
    frames, scores, independent = {}, {}, {}
    scores["S1"] = load(S1 / "score.json")
    scores["S2"] = load(MODEL / "score.json")
    for label in ("S1", "S2"):
        path = (S1 if label == "S1" else MODEL) / "C.csv"
        frames[label] = selfscore.load_submitted_c(path)
    for method in ("round", "sample"):
        assert control_provenance["methods"][method]["sha256"] == digest(control_dir / f"{method}.csv")
        label = "same_model_" + method
        frame = selfscore.load_submitted_c(control_dir / f"{method}.csv")
        assert not validate_c_submission(frame, cfg, n_expected=len(b)).errors
        frames[label] = frame
        scores[label] = json_safe(selfscore.self_score(frame, b, ref, cfg))
    features = scores["S2"]["detail"]["U_gen"]["correlation"]["features"]
    for label, c in frames.items():
        differences = []
        for left, right in combinations(features, 2):
            rb = float(spearmanr(b[left], b[right]).statistic)
            rc = float(spearmanr(c[left], c[right]).statistic)
            differences.append(abs(rb - rc))
        assert len(differences) == 36 and np.isfinite(differences).all()
        mean = float(np.mean(differences))
        tail = float(np.quantile(differences, 0.9))
        official = scores[label]["detail"]["U_gen"]["correlation"]
        assert abs(mean - official["observed_mean_dist"]) < 1e-12
        assert abs(tail - official["observed_tail_dist"]) < 1e-12
        independent[label] = {
            "mean_absolute_correlation_error": mean,
            "q90_absolute_correlation_error": tail,
            "row_index_spearman": {
                name: float(spearmanr(c[name], np.arange(len(c))).statistic)
                for name in features
            },
        }
    old_audit = load(MODEL.parent / "research/s1_official_verification_20260917/verification.json")
    assert all(digest(S1 / name) == expected for name, expected in old_audit["frozen_s1_sha256"].items())
    assert digest(MODEL / "C.csv") == evaluation_provenance["c_sha256"]
    assert digest(MODEL / "submission/team10_S2_defense.zip") == evaluation_provenance["submission_zip_sha256"]
    summary = {label: {
        "utility": score["utility"], "facets": score["facets"],
        "corr_score": score["detail"]["U_gen"]["corr_score"],
        **independent[label],
    } for label, score in scores.items()}
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local utility only; reference MIA and official CodaBench score not available",
        "S1_unchanged": True,
        "S2_C_and_zip_unchanged_after_scoring": True,
        "model_snapshot_sha256": digest(MODEL / "model_snapshot.npz"),
        "control_provenance": control_provenance,
        "comparison": summary,
        "same_model_control_scores": {method: scores["same_model_" + method] for method in ("round", "sample")},
        "limitations": [
            "S1 and S2 use separate DP measurements, so their score difference alone is not a controlled sampler comparison.",
            "The round/sample control pair uses one saved S2 model and seed 0 fixed before evaluation; no new private measurement occurs.",
            "Private self-scoring is an evaluation, not itself a DP guarantee for the entire development and candidate selection process.",
        ],
    }
    (MODEL / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
