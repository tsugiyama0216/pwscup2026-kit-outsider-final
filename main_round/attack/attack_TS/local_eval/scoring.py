"""Use the immutable official score functions; unknown AIA phi is not invented."""
import numpy as np
from pwscup2026.scoring.attack_score import score_mia, score_aia


def validate_predictions(predictions, case):
    if not isinstance(predictions, dict) or set(predictions) != {"mia", "aia"}:
        raise ValueError("predict must return exactly {'mia': array, 'aia': array}")
    mia, aia = (np.asarray(predictions[k], dtype=float) for k in ("mia", "aia"))
    if mia.shape != (len(case.candidates),) or aia.shape != (len(case.challenge),):
        raise ValueError("Prediction lengths must match the supplied query row order")
    if not np.isfinite(mia).all() or not np.isfinite(aia).all():
        raise ValueError("Predictions must be finite")
    if not ((mia >= 0) & (mia <= 1)).all() or not (aia >= 0).all():
        raise ValueError("MIA must be in [0,1], AIA time must be nonnegative")
    return mia, aia


def score_arrays(mia, aia, case, cfg, mia_truth=None, aia_member=None):
    truth = case.mia_truth if mia_truth is None else mia_truth
    member = case.aia_member if aia_member is None else aia_member
    m = score_mia(mia, truth, cfg, rare_mask=case.mia_rare)
    a = score_aia(None, aia[member], case.aia_onset[member], case.aia_time[member],
                  None, aia[~member], case.aia_onset[~member], case.aia_time[~member], cfg,
                  member_rare_mask=case.aia_rare[member], control_rare_mask=case.aia_rare[~member])
    return {"mia": float(m.score), "mia_all": m.breakdown["tpr_at_low_fpr"],
            "mia_rare": m.breakdown["tpr_rare"], "mia_auc": m.breakdown["auc"],
            "aia_no_phi": float(a.score), "aia_all": a.breakdown["r_time_all"],
            "aia_rare": a.breakdown["r_time_rare"],
            "aia_member_success": a.breakdown["succ_time_member"],
            "aia_control_success": a.breakdown["succ_time_control"],
            "combined_no_phi": float(m.score + a.score)}


def null_diagnostics(mia, aia, case, cfg, repetitions):
    """Label randomization diagnostics only, not official chance/floor correction."""
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(np.random.SeedSequence([case.metadata["seed"], 76193]))
    mia_values, aia_values = [], []
    for _ in range(repetitions):
        truth = case.mia_truth.copy()
        for rare in (False, True):
            indices = np.flatnonzero(case.mia_rare == rare)
            truth[indices] = rng.permutation(truth[indices])
        # Exchange member/control within exact-matched pairs, retaining all quotas.
        flips = rng.integers(0, 2, size=163).astype(bool)
        member = np.logical_xor(case.aia_member, flips[case.aia_pair])
        score = score_arrays(mia, aia, case, cfg, truth, member)
        mia_values.append(score["mia"])
        aia_values.append(score["aia_no_phi"])
    return {"null_repetitions": repetitions, "mia_null_mean": float(np.mean(mia_values)),
            "mia_null_p95": float(np.quantile(mia_values, .95)),
            "aia_null_mean": float(np.mean(aia_values)), "aia_null_p95": float(np.quantile(aia_values, .95))}
