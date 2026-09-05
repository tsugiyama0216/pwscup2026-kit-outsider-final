"""Validate public-feature-only rare proxies before attack-policy testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from rare_proxy import (  # noqa: E402
    VISIBLE_RARE_COLUMNS,
    fit_proxy_models,
    partial_mahalanobis,
    stable_rank,
    top_k_mask,
)


SEED = 20270300


def metrics(labels: np.ndarray, score: np.ndarray, *, k: int) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    ids = np.arange(len(labels), dtype=np.int64)
    selected = top_k_mask(score, ids, k)
    true_positive = int((selected & labels).sum())
    return {
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, score)),
        "average_precision": float(average_precision_score(labels, score)),
        "top_k": int(k),
        "top_k_true_positive": true_positive,
        "top_k_precision": float(true_positive / max(k, 1)),
        "top_k_recall": float(true_positive / max(int(labels.sum()), 1)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    detector = shadow.load_rare_detector(dist)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    if not candidates["record_id"].equals(b_self["record_id"]):
        raise ValueError("B_practice and B_self are not aligned")
    background_labels = shadow.public_rare_mask(background, detector)

    method_names = ("partial_mahal", "quadratic_logit", "extra_trees", "ensemble")
    oof = {name: np.full(len(background), np.nan, dtype=float) for name in method_names}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(
        folds.split(background, background_labels.astype(int))
    ):
        models = fit_proxy_models(
            background.iloc[train].reset_index(drop=True),
            background_labels[train],
            detector,
            seed=SEED + fold,
        )
        test_frame = background.iloc[test].reset_index(drop=True)
        scores = models.scores(test_frame)
        for name in method_names:
            oof[name][test] = scores[name]
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("OOF proxy score is incomplete")

    models = fit_proxy_models(background, background_labels, detector, seed=SEED)
    candidate_scores = models.scores(candidates)
    candidate_scores["partial_mahal_raw"] = partial_mahalanobis(candidates, detector)
    full_detector_labels = shadow.public_rare_mask(candidates, detector)
    true_labels = b_self["is_rare"].astype(bool).to_numpy()

    report: dict[str, Any] = {
        "scope": "rare-proxy validation using public data only",
        "visible_columns": list(VISIBLE_RARE_COLUMNS),
        "missing_detector_columns_in_aia_challenge": ["BMI", "FPG"],
        "background_oof": {},
        "candidate_vs_full_public_detector": {},
        "candidate_vs_B_self_true_rare": {},
        "self_challenge": {
            "available": False,
            "reason": (
                "The retained B_practice/B_self cohort is not the cohort behind "
                "the preliminary team-shared target-24 challenge; no exact public-QI "
                "links are present. Shadow validation is used instead."
            ),
        },
    }
    for name in method_names:
        report["background_oof"][name] = metrics(
            background_labels, oof[name], k=int(background_labels.sum())
        )
        report["candidate_vs_full_public_detector"][name] = metrics(
            full_detector_labels,
            candidate_scores[name],
            k=int(full_detector_labels.sum()),
        )
        report["candidate_vs_B_self_true_rare"][name] = metrics(
            true_labels,
            candidate_scores[name],
            k=int(true_labels.sum()),
        )

    # Predeclare the attack-stage proxy using only these source-quality checks.
    selection_value = {
        name: (
            report["candidate_vs_B_self_true_rare"][name]["top_k_recall"]
            + 0.25 * report["background_oof"][name]["roc_auc"]
        )
        for name in method_names
    }
    selected = max(method_names, key=lambda name: (selection_value[name], name))
    report["selection"] = {
        "method": selected,
        "criterion": (
            "B_self top-prevalence recall + 0.25*A_bg OOF AUC"
        ),
        "values": selection_value,
    }
    report.update(
        {
            "network_access": False,
            "token_read": False,
            "production_candidate_changed": False,
            "submission_performed": False,
        }
    )
    (HERE / "rare_proxy_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    score_table = pd.DataFrame(
        {
            "record_id": candidates["record_id"].astype(int),
            "full_public_detector": full_detector_labels.astype(int),
            "B_self_true_rare": true_labels.astype(int),
            **{name: candidate_scores[name] for name in method_names},
        }
    )
    score_table.to_csv(HERE / "rare_proxy_candidate_scores.csv", index=False)
    print(json.dumps(report["selection"], indent=2))
    print(
        pd.DataFrame(report["candidate_vs_B_self_true_rare"])
        .T[["roc_auc", "top_k_recall", "top_k_precision"]]
        .to_string()
    )


if __name__ == "__main__":
    main()
