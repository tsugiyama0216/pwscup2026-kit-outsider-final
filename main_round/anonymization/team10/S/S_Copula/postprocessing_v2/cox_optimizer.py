"""Synthetic-only Efron Cox calibration by survival-triple permutation.

No fitting or reading of private records occurs here. Supply only a synthetic
table, a saved DP/public coefficient target and public nuisance parameters.
The complete predictor table stays unchanged. Swapping whole survival triples
preserves their empirical joint distribution, including censoring consistency.
When strata are supplied, that guarantee also holds separately in each stratum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

COX_COLUMNS = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]
OUTCOME_COLUMNS = ["time", "onset", "death"]


def features(df):
    frame = df.copy()
    frame["log_TG"] = np.log(frame.TG.to_numpy(float))
    frame["log_ALT"] = np.log(frame.ALT.to_numpy(float))
    return frame[COX_COLUMNS].to_numpy(float)


class EfronRiskSets:
    """Exact Efron score, observed information and score-loss derivative."""

    def __init__(self, time, event):
        self.time = np.asarray(time, float)
        self.event = np.asarray(event, float)
        if np.any(np.diff(self.time) < 0):
            raise ValueError("Risk-set rows must be sorted by ascending time")
        unique, first, inverse = np.unique(self.time, return_index=True, return_inverse=True)
        self.first, self.inverse = first, inverse
        count = np.bincount(inverse, weights=self.event).astype(int)
        active = np.flatnonzero(count)
        self.tie_group = np.repeat(active, count[active])
        self.fraction = np.concatenate([np.arange(count[g]) / count[g] for g in active])

    def evaluate(self, x, beta, *, information=False, gradient_vector=None, score_jacobian=False):
        n, p = x.shape
        eta = x @ beta
        weight = np.exp(eta - np.max(eta))
        weighted = weight[:, None] * x
        risk0 = np.cumsum(weight[::-1])[::-1][self.first]
        risk1 = np.cumsum(weighted[::-1], axis=0)[::-1][self.first]
        deaths0 = np.bincount(self.inverse, weights=weight * self.event)
        deaths1 = np.zeros_like(risk1)
        np.add.at(deaths1, self.inverse, weighted * self.event[:, None])
        g, f = self.tie_group, self.fraction
        denominator = risk0[g] - f * deaths0[g]
        mean = (risk1[g] - f[:, None] * deaths1[g]) / denominator[:, None]
        score = x.T @ self.event - mean.sum(axis=0)
        info, gradient = None, None
        if information:
            second = weighted[:, :, None] * x[:, None, :]
            risk2 = np.cumsum(second[::-1], axis=0)[::-1][self.first]
            death2 = np.zeros_like(risk2)
            np.add.at(death2, self.inverse, second * self.event[:, None, None])
            moment = (risk2[g] - f[:, None, None] * death2[g]) / denominator[:, None, None]
            info = np.sum(moment - mean[:, :, None] * mean[:, None, :], axis=0)
        if gradient_vector is not None or score_jacobian:
            a_group = np.bincount(g, weights=1 / denominator, minlength=len(self.first))
            a_tie = np.bincount(g, weights=f / denominator, minlength=len(self.first))
            v_group = np.zeros_like(risk1)
            v_tie = np.zeros_like(risk1)
            np.add.at(v_group, g, mean / denominator[:, None])
            np.add.at(v_tie, g, mean * (f / denominator)[:, None])
            a = np.cumsum(a_group)[self.inverse] - self.event * a_tie[self.inverse]
            v = np.cumsum(v_group, axis=0)[self.inverse] - self.event[:, None] * v_tie[self.inverse]
            coefficient = self.event - weight * a
            jacobian_vector = weight[:, None] * (a[:, None] * x - v)
            if score_jacobian:
                gradient = (coefficient[:, None, None] * np.eye(p)
                            - jacobian_vector[:, :, None] * beta[None, None, :])
            else:
                direction = np.asarray(gradient_vector)
                gradient = coefficient[:, None] * direction - (jacobian_vector @ direction)[:, None] * beta
        return score, info, gradient


def _fit(x, time, event):
    frame = pd.DataFrame(x, columns=COX_COLUMNS)
    frame["time"], frame["onset"] = time, event
    model = CoxPHFitter().fit(frame, duration_col="time", event_col="onset")
    return model.params_.to_numpy(float), model.standard_errors_.to_numpy(float)


def calibrate(synthetic, targetBeta, public_model, *, strata=None, iterations=150,
              candidates=4096, trials=12, seed=20260918, target_information=None,
              information_weight=0.0, max_changed_fraction=1.0,
              coefficient_tolerance=0.005):
    """Return (C, diagnostics), with deterministic synthetic-only selection.

    ``targetBeta`` is a nine-vector in public standardized units, or a saved
    target dictionary with ``beta_standardized``. ``target_information`` may
    only be publicly estimated or DP-released observed information for the
    desired output size. The score uses the *actual synthetic risk sets*.
    No target information is silently substituted from an unprotected cohort.

    The default information penalty is zero. A caller can supply independently
    validated public information and a nonnegative penalty. All swaps must
    improve the exact score/information objective; analytic gradients propose
    swaps but never accept them without exact evaluation.
    """
    if iterations < 0 or candidates < 1 or trials < 1 or not 0 <= max_changed_fraction <= 1:
        raise ValueError("Invalid calibration settings")
    if information_weight < 0 or (information_weight and target_information is None):
        raise ValueError("An information penalty requires an explicit public/DP target")
    original = synthetic.copy().reset_index(drop=True)
    beta = np.asarray(targetBeta["beta_standardized"] if isinstance(targetBeta, dict) else targetBeta, float)
    center, scale = np.asarray(public_model["center"]), np.asarray(public_model["scale"])
    if beta.shape != (9,) or np.any(scale <= 0):
        raise ValueError("Invalid public model or coefficient target")
    order = np.argsort(original.time.to_numpy(float), kind="stable")
    x = ((features(original) - center) / scale)[order]
    time, event = original.time.to_numpy(float)[order], original.onset.to_numpy(float)[order]
    if not np.isfinite(x).all() or not np.isfinite(beta).all() or event.sum() < 10:
        raise ValueError("Finite inputs and at least ten synthetic events are required")
    groups = np.zeros(len(x), int) if strata is None else np.asarray(strata)
    if len(groups) != len(x):
        raise ValueError("Strata length mismatch")
    groups = groups[order]
    group_rows = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    group_rows = [rows for rows in group_rows if len(rows) > 1]
    if not group_rows:
        return original, {"method": "synthetic_efron_triple_swaps", "swaps": 0}
    risk = EfronRiskSets(time, event)
    score, initial_information, _ = risk.evaluate(x, beta, information=True)
    inv_information = np.linalg.pinv(initial_information, rcond=1e-8)
    if target_information is not None:
        target_information = np.asarray(target_information, float)
        target_se = np.sqrt(np.diag(np.linalg.pinv(target_information)))
        if target_information.shape != (9, 9) or not np.all(target_se > 0):
            raise ValueError("Invalid public/DP information target")
    def evaluate(values, need_gradient=False):
        score, info, _ = risk.evaluate(values, beta, information=bool(information_weight))
        delta = inv_information @ score
        loss = float(delta @ delta)
        if information_weight:
            se = np.sqrt(np.diag(np.linalg.pinv(info)))
            loss += information_weight * float(np.mean(np.log(se / target_se) ** 2))
        gradient = None
        if need_gradient:
            _, _, gradient = risk.evaluate(values, beta, score_jacobian=True)
        return loss, gradient
    initial_beta, initial_se = _fit(x, time, event)
    loss, gradient = evaluate(x, True)
    history = [loss]
    permutation = np.arange(len(x))
    rng = np.random.default_rng(seed)
    swaps = 0
    for iteration in range(iterations):
        if np.sqrt(loss) <= coefficient_tolerance:
            break
        left, right = [], []
        for rows in group_rows:
            size = max(1, int(candidates * len(rows) / len(x)))
            left.append(rng.choice(rows, size=size))
            right.append(rng.choice(rows, size=size))
        left, right = np.concatenate(left), np.concatenate(right)
        score_delta = np.einsum("ijk,ik->ij", gradient[left] - gradient[right], x[right] - x[left])
        current_delta = inv_information @ risk.evaluate(x, beta)[0]
        predicted_delta = current_delta + score_delta @ inv_information.T
        proposed = np.sum(predicted_delta ** 2, axis=1)
        ranked = np.argsort(proposed, kind="stable")[:trials]
        best_loss, best_pair = loss, None
        for k in ranked:
            a, b = left[k], right[k]
            if a == b:
                continue
            new_perm = permutation.copy()
            new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            if np.mean(new_perm != np.arange(len(x))) > max_changed_fraction:
                continue
            x[[a, b]] = x[[b, a]]
            candidate_loss, _ = evaluate(x)
            x[[a, b]] = x[[b, a]]
            if candidate_loss < best_loss:
                best_loss, best_pair = candidate_loss, (a, b)
        if best_pair is None:
            # Near an optimum linear approximations overstate large swaps.
            # Next iteration resamples, without accepting any worsening move.
            continue
        a, b = best_pair
        x[[a, b]] = x[[b, a]]
        permutation[[a, b]] = permutation[[b, a]]
        loss, gradient = evaluate(x, True)
        history.append(loss)
        swaps += 1
    result = original.copy()
    # Predictor vector originally at order[permutation[k]] now receives the
    # survival triple originally at order[k]. Predictor rows remain untouched.
    result.loc[order[permutation], OUTCOME_COLUMNS] = original.loc[order, OUTCOME_COLUMNS].to_numpy()
    fitted_beta, fitted_se = _fit(x, time, event)
    _, final_information, _ = risk.evaluate(x, beta, information=True)
    diagnostics = {
        "method": "synthetic_efron_triple_swaps", "selection_input": "synthetic_C_DP_target_public_model_only",
        "additional_private_queries": 0, "iterations_requested": iterations, "swaps": swaps,
        "objective_history": history, "initial_coefficient_l2_error": float(np.linalg.norm(initial_beta-beta)),
        "selected_coefficient_l2_error": float(np.linalg.norm(fitted_beta-beta)),
        "initial_beta_standardized": initial_beta.tolist(), "selected_beta_standardized": fitted_beta.tolist(),
        "initial_se_standardized": initial_se.tolist(), "selected_se_standardized": fitted_se.tolist(),
        "information_weight": information_weight, "changed_survival_fraction": float(np.mean(permutation != np.arange(len(x)))),
        "predictor_rows_unchanged": True, "survival_triples_preserved_per_stratum": True,
        "seed": seed,
    }
    if target_information is not None:
        diagnostics["target_se_standardized"] = target_se.tolist()
        diagnostics["selected_se_log_rmse"] = float(np.sqrt(np.mean(np.log(fitted_se/target_se)**2)))
    return result, diagnostics
