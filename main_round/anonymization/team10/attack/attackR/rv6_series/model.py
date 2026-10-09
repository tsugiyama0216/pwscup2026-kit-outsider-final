"""Six fixed public-input hypotheses. No scoring, hidden truth or filesystem access.

predict_all accepts aligned baseline arrays produced by Rv2 for the SAME public
problem. Saved production Rv2 CSVs must never be reused on unrelated mock rows.
MIA outputs are ranking scores, not calibrated membership probabilities.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor

FULL = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT']
CATS = ['sex', 'prefecture']
PUBLIC = FULL + ['time', 'sex', 'prefecture', 'smoking', 'onset', 'death']


def canonical(frame):
    return frame.sort_values([c for c in PUBLIC if c in frame], kind='stable').reset_index(drop=True)


def numeric(frame, columns):
    values = frame[columns].to_numpy(float).copy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite public numeric input')
    for i, column in enumerate(columns):
        if column in ('TG', 'ALT'):
            if (values[:, i] < 0).any():
                raise ValueError('Negative laboratory value')
            values[:, i] = np.log1p(values[:, i])
    return values


def check_vector(values, n, kind):
    values = np.asarray(values, float)
    if values.shape != (n,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid ' + kind + ' baseline')
    if kind == 'MIA' and (values > 1).any():
        raise ValueError('MIA baseline outside [0,1]')
    return values


class Metric:
    """Reference-fitted scale/whitening; categorical mismatch has fixed weight."""
    def __init__(self, background, columns, whiten=False, shrinkage=.2):
        self.columns = columns
        x = numeric(background, columns)
        self.center = np.median(x, axis=0)
        self.scale = np.maximum(np.std(x, axis=0), 1e-6)
        self.transform = np.eye(len(columns))
        if whiten:
            z = (x - self.center) / self.scale
            cov = np.cov(z, rowvar=False)
            cov = (1 - shrinkage) * cov + shrinkage * np.eye(len(columns))
            vals, vecs = np.linalg.eigh(cov)
            self.transform = (vecs * (1 / np.sqrt(np.maximum(vals, 1e-6)))) @ vecs.T

    def encode(self, frame):
        return ((numeric(frame, self.columns) - self.center) / self.scale) @ self.transform

    def distances(self, query, reference):
        d = cdist(self.encode(query), self.encode(reference), 'sqeuclidean') / len(self.columns)
        for col, weight in [('sex', 1.), ('prefecture', .25)]:
            d += weight * (query[col].to_numpy()[:, None] != reference[col].to_numpy()[None, :])
        return d

    def radii(self, query, reference, ks, leave_self_out=False):
        if len(reference) - int(leave_self_out) < max(ks):
            raise ValueError('Insufficient neighbors')
        parts = []
        indices = np.asarray(ks) - 1
        for start in range(0, len(query), 192):
            d = self.distances(query.iloc[start:start + 192], reference)
            if leave_self_out:
                d[np.arange(len(d)), start + np.arange(len(d))] = np.inf
            parts.append(np.partition(d, indices, axis=1)[:, indices])
        return np.concatenate(parts)


def null_percentile(values, reference, candidates, background):
    """Conditional midrank; empirical background evidence, NOT certified FPR."""
    out = np.empty(len(values))
    for sex in pd.unique(candidates.sex):
        for pref in pd.unique(candidates.prefecture):
            ix = (candidates.sex.to_numpy() == sex) & (candidates.prefecture.to_numpy() == pref)
            use = (background.sex.to_numpy() == sex) & (background.prefecture.to_numpy() == pref)
            if use.sum() < 40:
                use = background.sex.to_numpy() == sex
            if use.sum() < 40:
                use = np.ones(len(background), bool)
            r = np.sort(reference[use])
            out[ix] = (.5 + .5 * (np.searchsorted(r, values[ix], 'left') + np.searchsorted(r, values[ix], 'right'))) / (len(r) + 1)
    return out


def ranks(values):
    return (rankdata(values, method='average') - .5) / len(values)


def bounded_logit(values, cap):
    return np.clip(logit(np.clip(values, 1e-12, 1 - 1e-12)), -cap, cap)


def blend_membership(old, evidence, private, settings):
    old = check_vector(old, len(evidence), 'MIA')
    private = np.asarray(private, bool)
    if private.shape != old.shape or private.all():
        raise ValueError('Invalid known-negative mask')
    use = ~private
    zold = bounded_logit(ranks(old[use]), settings['mia_logit_cap'])
    znew = bounded_logit(evidence[use], settings['mia_logit_cap'])
    w = settings['mia_new_weight']
    result = np.zeros(len(old))
    result[use] = expit((1 - w) * zold + w * znew)
    return result


class MembershipEvidence:
    def __init__(self, candidates, background, settings):
        self.b = candidates
        self.bg = canonical(background)
        self.s = settings
        self.groups = {}
        for name, specs in [('axis', [(FULL, False), (FULL + ['time'], False)]),
                            ('white', [(FULL, True), (QI, True)])]:
            views = []
            for cols, white in specs:
                metric = Metric(self.bg, cols, white, settings['covariance_shrinkage'])
                rb = metric.radii(candidates, self.bg, settings['neighbors'])
                rn = metric.radii(self.bg, self.bg, settings['neighbors'], True)
                views.append((metric, rb, rn))
            self.groups[name] = views

    def predict(self, c, name):
        c = canonical(c)
        score, null = [], []
        for metric, rb, rn in self.groups[name]:
            dc = metric.radii(self.b, c, self.s['neighbors'])
            dn = metric.radii(self.bg, c, self.s['neighbors'])
            score.append(np.log((rb + 1e-10) / (dc + 1e-10)) @ np.asarray(self.s['neighbor_weights']))
            null.append(np.log((rn + 1e-10) / (dn + 1e-10)) @ np.asarray(self.s['neighbor_weights']))
        weights = np.asarray(self.s[name + '_view_weights'])
        return null_percentile(weights @ np.asarray(score), weights @ np.asarray(null), self.b, self.bg)


def specificity(scores, private, settings):
    """Remove a soft person-wide propensity; multiple target memberships allowed.

    No row sum, cohort assignment, positive count or exclusivity constraint.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError('Specificity needs multiple targets')
    use = ~np.asarray(private, bool)
    logits = np.zeros_like(scores)
    for j in range(scores.shape[1]):
        logits[use, j] = bounded_logit(ranks(scores[use, j]), settings['mia_logit_cap'])
    propensity = np.median(logits[use], axis=1)
    propensity -= np.median(propensity)
    result = np.zeros_like(scores)
    result[use] = expit(logits[use] - settings['specificity_strength'] * propensity[:, None])
    return result


def window_options(times, baseline, tau=.5):
    """Include the center of every maximal closed 2*tau support interval."""
    values = np.sort(np.unique(np.asarray(times, float)))
    ends = np.searchsorted(values, values + 2 * tau + 1e-12, side='right') - 1
    return np.unique(np.r_[values, (values + values[ends]) / 2, baseline])


def window_mass(times, weights, options, tau=.5):
    order = np.argsort(times, kind='stable')
    times = np.asarray(times)[order]
    sums = np.r_[0., np.cumsum(np.asarray(weights)[order])]
    lo = np.searchsorted(times, options - tau - 1e-12, 'left')
    hi = np.searchsorted(times, options + tau + 1e-12, 'right')
    return np.clip(sums[hi] - sums[lo], 0., 1.)


def contrast(pc, pb):
    return (pc - pb) / np.maximum(1 - pb, .05)


def choose_window(ct, cw, bt, bw, old, ratio, min_mass, min_gain, max_ratio, tau):
    # Background window endpoints are also necessary: excess support can peak
    # between two dense background modes, not only at a member support center.
    options = window_options(np.r_[ct, bt], old, tau)
    pc = window_mass(ct, cw, options, tau)
    pb = window_mass(bt, bw, options, tau)
    score = contrast(pc, pb)
    old_index = int(np.flatnonzero(options == old)[0])
    # Prefer the baseline under ties, then its nearest equally good center.
    order = np.lexsort((options, np.abs(options - old), -score))
    best = int(order[0])
    gain = float(score[best] - score[old_index])
    use = ratio <= max_ratio and pc[best] >= min_mass and gain >= min_gain
    return float(options[best] if use else old), {
        'gain': gain, 'member_proxy_mass': float(pc[best]),
        'background_mass': float(pb[best]), 'distance_ratio': float(ratio)}


class AttributeEvidence:
    def __init__(self, background, settings):
        self.bg = canonical(background)
        self.s = settings
        self.metric = Metric(self.bg, QI)
        self.categories = {c: sorted(pd.unique(self.bg[c]), key=str) for c in CATS}
        self.bg_forest = None

    def encode(self, frame):
        parts = [self.metric.encode(frame)]
        for c in CATS:
            parts.append(np.column_stack([(frame[c].to_numpy() == v).astype(float) for v in self.categories[c]]))
        return np.column_stack(parts)

    def fit_forest(self, frame):
        s = self.s
        tree = ExtraTreesRegressor(n_estimators=s['forest_trees'], min_samples_leaf=s['forest_min_leaf'],
                                  max_features=s['forest_max_features'], random_state=s['forest_seed'], n_jobs=1)
        tree.fit(self.encode(frame), frame.time.to_numpy(float))
        return tree

    def forest_weights(self, tree, reference, query):
        train = tree.apply(self.encode(reference))
        test = tree.apply(self.encode(query))
        weights = np.zeros((len(query), len(reference)))
        for t in range(train.shape[1]):
            for leaf in np.unique(test[:, t]):
                ri = np.flatnonzero(train[:, t] == leaf)
                qi = np.flatnonzero(test[:, t] == leaf)
                weights[np.ix_(qi, ri)] += 1. / (train.shape[1] * len(ri))
        return weights

    def predict(self, c, challenge, old, forest=True):
        s = self.s
        c = canonical(c)
        old = check_vector(old, len(challenge), 'AIA')
        dc = self.metric.distances(challenge, c)
        db = self.metric.distances(challenge, self.bg)
        ct, bt = c.time.to_numpy(float), self.bg.time.to_numpy(float)
        if not np.isfinite(ct).all() or not np.isfinite(bt).all() or (ct < 0).any() or (bt < 0).any():
            raise ValueError('Invalid public time')
        if forest:
            if self.bg_forest is None:
                self.bg_forest = self.fit_forest(self.bg)
            wc = self.forest_weights(self.fit_forest(c), c, challenge)
            wb = self.forest_weights(self.bg_forest, self.bg, challenge)
        knn, trees, ensemble, details = old.copy(), old.copy(), old.copy(), []
        for i in range(len(challenge)):
            ci = np.argsort(dc[i], kind='stable')[:min(s['aia_neighbors_c'], len(c))]
            bi = np.argsort(db[i], kind='stable')[:min(s['aia_neighbors_bg'], len(self.bg))]
            cw = np.exp(-(dc[i, ci] - dc[i, ci[0]]) / max(dc[i, ci[-1]], 1e-9)); cw /= cw.sum()
            bw = np.exp(-(db[i, bi] - db[i, bi[0]]) / max(db[i, bi[-1]], 1e-9)); bw /= bw.sum()
            ratio = (dc[i, ci[0]] + 1e-12) / (db[i, bi[0]] + 1e-12)
            knn[i], kd = choose_window(ct[ci], cw, bt[bi], bw, old[i], ratio,
                                      s['aia_min_mass'], s['aia_min_gain'], s['aia_distance_ratio'], s['aia_tau'])
            td = None
            if forest:
                trees[i], td = choose_window(ct, wc[i], bt, wb[i], old[i], ratio,
                                            s['forest_min_mass'], s['forest_min_gain'], s['forest_distance_ratio'], s['aia_tau'])
                # Select among actual modal predictions; never average remote modes.
                opts = np.unique([old[i], knn[i], trees[i]])
                local = contrast(window_mass(ct[ci], cw, opts), window_mass(bt[bi], bw, opts))
                learned = contrast(window_mass(ct, wc[i], opts), window_mass(bt, wb[i], opts))
                objective = .5 * local + .5 * learned
                oldpos = int(np.flatnonzero(opts == old[i])[0])
                j = np.lexsort((opts, np.abs(opts - old[i]), -objective))[0]
                if (local[j] >= local[oldpos] - 1e-12 and learned[j] >= learned[oldpos] - 1e-12
                        and objective[j] - objective[oldpos] >= s['ensemble_aia_min_gain']):
                    ensemble[i] = opts[j]
            details.append({'knn': kd, 'forest': td})
        return knn, trees, ensemble, details


def predict_all(candidates, background, targets, challenges, baseline_mia, baseline_aia,
                known_negative, settings, progress=None):
    """Return six {mia: {k: array}, aia: {k: array}} dictionaries and diagnostics.

    The caller aligns array positions to candidates/challenges. Known negatives
    must be externally justified public deductions; use all-False otherwise.
    IDs are never learned or treated as explanatory variables.
    """
    keys = sorted(targets)
    if set(keys) != set(challenges) or set(keys) != set(baseline_mia) or set(keys) != set(baseline_aia):
        raise ValueError('Target keys do not match')
    if len(keys) < 2:
        raise ValueError('Joint suite needs at least two targets')
    known_negative = np.asarray(known_negative, bool)
    if known_negative.shape != (len(candidates),) or known_negative.all():
        raise ValueError('Invalid known-negative mask')
    outputs = {n: {'mia': {}, 'aia': {}} for n in settings['series']}
    evidence = MembershipEvidence(candidates, background, settings)
    attribute = AttributeEvidence(background, settings)
    diagnostics = {}
    for k in keys:
        oldm = check_vector(baseline_mia[k], len(candidates), 'MIA')
        olda = check_vector(baseline_aia[k], len(challenges[k]), 'AIA')
        for name, mode in [('Rv6', 'axis'), ('Rv7', 'white')]:
            outputs[name]['mia'][k] = blend_membership(oldm, evidence.predict(targets[k], mode), known_negative, settings)
            outputs[name]['aia'][k] = olda.copy()
        knn, trees, merged, details = attribute.predict(targets[k], challenges[k], olda)
        for name, times in [('Rv9', knn), ('Rv10', trees), ('Rv11', merged)]:
            outputs[name]['aia'][k] = times
        for name in ['Rv9', 'Rv10']:
            m = oldm.copy(); m[known_negative] = 0
            outputs[name]['mia'][k] = m
        outputs['Rv8']['aia'][k] = olda.copy()
        diagnostics[str(k)] = {'aia_rows': details}
        if progress:
            progress(k)
    base = np.column_stack([baseline_mia[k] for k in keys])
    normalized = specificity(base, known_negative, settings)
    for j, k in enumerate(keys):
        outputs['Rv8']['mia'][k] = normalized[:, j]
        merged = sum(w * bounded_logit(outputs[n]['mia'][k], settings['mia_logit_cap'])
                     for w, n in zip(settings['ensemble_mia_weights'], ['Rv6', 'Rv7', 'Rv8']))
        outputs['Rv11']['mia'][k] = expit(merged)
        outputs['Rv11']['mia'][k][known_negative] = 0
    return outputs, diagnostics
