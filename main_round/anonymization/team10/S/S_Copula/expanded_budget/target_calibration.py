"""Synthetic-only joint beta/SE calibration by within-rare survival swaps."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "postprocessing_v2"))
from cox_optimizer import EfronRiskSets, features, _fit, OUTCOME_COLUMNS


def score_information_derivatives(x, beta, risk):
    """Exact Efron score/info and their feature-row derivative factors.

    For displacement h of x[i], dI = h B[i]^T + B[i] h^T +
    (beta^T h) C[i]. Thus pair-swap information proposals need O(p^2),
    avoiding n*p*p*p tensor materialization.
    """
    score, information, jacobian = risk.evaluate(x, beta, information=True, score_jacobian=True)
    weight = np.exp(x @ beta - np.max(x @ beta))
    weighted = weight[:, None]*x
    first, inverse = risk.first, risk.inverse
    r0 = np.cumsum(weight[::-1])[::-1][first]
    r1 = np.cumsum(weighted[::-1], axis=0)[::-1][first]
    second = weighted[:, :, None]*x[:, None, :]
    r2 = np.cumsum(second[::-1], axis=0)[::-1][first]
    d0 = np.bincount(inverse, weights=weight*risk.event)
    d1, d2 = np.zeros_like(r1), np.zeros_like(r2)
    np.add.at(d1, inverse, weighted*risk.event[:, None])
    np.add.at(d2, inverse, second*risk.event[:, None, None])
    g, f = risk.tie_group, risk.fraction
    denominator = r0[g]-f*d0[g]
    mu = (r1[g]-f[:, None]*d1[g])/denominator[:, None]
    mu2 = (r2[g]-f[:, None, None]*d2[g])/denominator[:, None, None]
    a0 = np.bincount(g, weights=1/denominator, minlength=len(first))
    a1 = np.bincount(g, weights=f/denominator, minlength=len(first))
    v0, v1 = np.zeros_like(r1), np.zeros_like(r1)
    w0, w1 = np.zeros_like(r2), np.zeros_like(r2)
    np.add.at(v0, g, mu/denominator[:, None])
    np.add.at(v1, g, mu*(f/denominator)[:, None])
    term = 2*mu[:, :, None]*mu[:, None, :]-mu2
    np.add.at(w0, g, term/denominator[:, None, None])
    np.add.at(w1, g, term*(f/denominator)[:, None, None])
    a = np.cumsum(a0)[inverse]-risk.event*a1[inverse]
    v = np.cumsum(v0, axis=0)[inverse]-risk.event[:, None]*v1[inverse]
    w = np.cumsum(w0, axis=0)[inverse]-risk.event[:, None, None]*w1[inverse]
    b = weight[:, None]*(a[:, None]*x-v)
    c = weight[:, None, None]*(a[:, None, None]*x[:, :, None]*x[:, None, :]
                               -x[:, :, None]*v[:, None, :]-v[:, :, None]*x[:, None, :]+w)
    return score, information, jacobian, b, c


def calibrate_target(synthetic, target, public_target, detector, *, seed=20260918,
                     iterations=250, candidates=4096, trials=12, se_weight=1.0,
                     coefficient_tolerance=0.005, se_log_tolerance=0.015):
    """Match a saved DP beta AND standard-error target, using synthetic C only.

    Required targets are ``beta_standardized`` and
    ``standard_errors_standardized`` in public center/scale units. No private
    fitting or validation is called. Every static row remains unchanged; each
    detector rare/nonrare stratum retains its complete survival-triple
    multiset. Both coefficient and information derivatives guide proposals,
    and exact score/information reevaluation gates each accepted swap.
    """
    if iterations < 0 or candidates < 1 or trials < 1 or se_weight < 0:
        raise ValueError("Invalid public optimization settings")
    beta = np.asarray(target["beta_standardized"], float)
    target_se = np.asarray(target["standard_errors_standardized"], float)
    if beta.shape != (9,) or target_se.shape != (9,) or not np.isfinite(beta).all() or not np.all(np.isfinite(target_se) & (target_se > 0)):
        raise ValueError("Nine finite coefficients and positive standard errors are required")
    original = synthetic.copy().reset_index(drop=True)
    center, scale = np.asarray(public_target["center"]), np.asarray(public_target["scale"])
    order = np.argsort(original.time.to_numpy(float), kind="stable")
    x = ((features(original)-center)/scale)[order]
    time, event = original.time.to_numpy(float)[order], original.onset.to_numpy(float)[order]
    if not np.isfinite(x).all() or event.sum() < 10:
        raise ValueError("Finite synthetic rows and at least ten events are required")
    centered = original[detector["columns"]].to_numpy(float)-np.asarray(detector["mu"])
    radii2 = np.einsum("ij,jk,ik->i", centered, np.linalg.pinv(detector["sigma"]), centered)
    groups = (radii2 >= float(detector["band"]["lo"])**2)[order]
    group_rows = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    group_rows = [r for r in group_rows if len(r) > 1]
    risk = EfronRiskSets(time, event)
    initial_beta, initial_se = _fit(x, time, event)
    def objective(score, information):
        covariance = np.linalg.inv(information)
        diagonal = np.diag(covariance)
        if not np.all(diagonal > 0):
            return float("inf"), covariance, np.full(9, np.nan), np.full(9, np.nan)
        delta = covariance @ score
        logse = .5*np.log(diagonal)-np.log(target_se)
        loss = float(np.mean((delta/target_se)**2)+se_weight*np.mean(logse**2))
        return loss, covariance, delta, logse
    score, info, jacobian, b, c = score_information_derivatives(x, beta, risk)
    loss, covariance, delta, logse = objective(score, info)
    history = [loss]
    permutation = np.arange(len(x))
    rng = np.random.default_rng(seed)
    accepted = 0
    for iteration in range(iterations):
        if not group_rows or (np.linalg.norm(delta) <= coefficient_tolerance and np.sqrt(np.mean(logse**2)) <= se_log_tolerance):
            break
        left, right = [], []
        for rows in group_rows:
            size = max(1, int(candidates*len(rows)/len(x)))
            left.append(rng.choice(rows, size))
            right.append(rng.choice(rows, size))
        left, right = np.concatenate(left), np.concatenate(right)
        dx = x[right]-x[left]
        db = b[left]-b[right]
        dscore = np.einsum("ijk,ik->ij", jacobian[left]-jacobian[right], dx)
        dinfo = dx[:, :, None]*db[:, None, :]+db[:, :, None]*dx[:, None, :]+(dx@beta)[:, None, None]*(c[left]-c[right])
        predicted_delta = delta+(dscore-np.einsum("ijk,k->ij", dinfo, delta))@covariance.T
        dvar = -np.einsum("ai,nij,ja->na", covariance, dinfo, covariance, optimize=True)
        predicted_logse = logse+.5*dvar/np.diag(covariance)
        predicted = np.mean((predicted_delta/target_se)**2, axis=1)+se_weight*np.mean(predicted_logse**2, axis=1)
        ranked = np.argsort(predicted, kind="stable")[:trials]
        best_loss, pair = loss, None
        for k in ranked:
            i, j = left[k], right[k]
            if i == j:
                continue
            x[[i, j]] = x[[j, i]]
            candidate_score, candidate_info, _ = risk.evaluate(x, beta, information=True)
            candidate_loss = objective(candidate_score, candidate_info)[0]
            x[[i, j]] = x[[j, i]]
            if candidate_loss < best_loss:
                best_loss, pair = candidate_loss, (i, j)
        if pair is None:
            continue
        i, j = pair
        x[[i, j]] = x[[j, i]]
        permutation[[i, j]] = permutation[[j, i]]
        score, info, jacobian, b, c = score_information_derivatives(x, beta, risk)
        loss, covariance, delta, logse = objective(score, info)
        history.append(loss)
        accepted += 1
    result = original.copy()
    result.loc[order[permutation], OUTCOME_COLUMNS] = original.loc[order, OUTCOME_COLUMNS].to_numpy()
    fitted_beta, fitted_se = _fit(x, time, event)
    trace = {
        "method": "synthetic_exact_Efron_beta_SE_survival_swaps",
        "selection_inputs": "synthetic C, saved DP beta/SE target, public constants only",
        "additional_epsilon_exact": "0", "iterations_requested": iterations,
        "swaps": accepted, "seed": seed, "se_weight": se_weight, "objective_history": history,
        "initial_coefficient_l2_error": float(np.linalg.norm(initial_beta-beta)),
        "selected_coefficient_l2_error": float(np.linalg.norm(fitted_beta-beta)),
        "initial_se_log_rmse": float(np.sqrt(np.mean(np.log(initial_se/target_se)**2))),
        "selected_se_log_rmse": float(np.sqrt(np.mean(np.log(fitted_se/target_se)**2))),
        "initial_beta_standardized": initial_beta.tolist(), "selected_beta_standardized": fitted_beta.tolist(),
        "initial_se_standardized": initial_se.tolist(), "selected_se_standardized": fitted_se.tolist(),
        "changed_survival_fraction": float(np.mean(permutation != np.arange(len(x)))),
        "predictor_rows_unchanged": True, "rare_stratum_survival_triples_preserved": True,
    }
    return result, trace
