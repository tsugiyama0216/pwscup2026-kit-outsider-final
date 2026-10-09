"""Rv12 public-only contrast veto and joint MIA specificity.

The veto never guesses hidden membership. It suppresses a baseline time only
when two neighborhoods find strong background support, no positive release
excess, and no compelling near-exact link. All thresholds are global.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.stats import rankdata

QI = ['age', 'SBP', 'TG', 'HDL', 'ALT']
FEATURES = ['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
ABSTAIN = 11.0
VARIANTS = {
    'retain': None,
    'censor_strict': {'background_min': .80, 'excess_max': .00, 'ratio_min': 1.5, 'censor_only': True},
    'censor_moderate': {'background_min': .70, 'excess_max': .05, 'ratio_min': 1.0, 'censor_only': True},
    'contrast_strict': {'background_min': .70, 'excess_max': .00, 'ratio_min': 1.5, 'censor_only': False},
    'contrast_moderate': {'background_min': .60, 'excess_max': .05, 'ratio_min': 1.0, 'censor_only': False},
}


def ranks(values):
    return (rankdata(values, method='average') - .5) / len(values)


def membership(old, known_negative):
    old = np.asarray(old, float)
    mask = np.asarray(known_negative, bool)
    if old.ndim != 2 or old.shape[1] < 2 or mask.shape != (len(old),) or mask.all():
        raise ValueError('Invalid joint MIA input')
    if not np.isfinite(old).all() or np.any((old < 0) | (old > 1)):
        raise ValueError('Invalid MIA scores')
    use = ~mask
    z = np.column_stack([np.clip(logit(ranks(old[use, j])), -7., 7.) for j in range(old.shape[1])])
    propensity = np.median(z, axis=1)
    propensity -= np.median(propensity)
    result = np.zeros_like(old)
    result[use] = expit(z - .35 * propensity[:, None])
    return result


def numeric(frame):
    x = frame[QI].to_numpy(float).copy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite QI')
    for i, name in enumerate(QI):
        if name in ('TG','ALT'):
            if (x[:,i] < 0).any():
                raise ValueError('Negative lab value')
            x[:,i] = np.log1p(x[:,i])
    return x


def canonical(frame):
    return frame.sort_values(FEATURES, kind='stable').reset_index(drop=True)


class AttributeVeto:
    def __init__(self, background):
        self.bg = canonical(background)
        self.scale = np.maximum(numeric(self.bg).std(axis=0), 1e-6)
        self.xbg = numeric(self.bg) / self.scale
        if len(self.bg) < 64:
            raise ValueError('Insufficient background')

    def distances(self, query, reference, encoded):
        d = cdist(numeric(query)/self.scale, encoded, 'sqeuclidean') / len(QI)
        for col, weight in [('sex', 1.), ('prefecture', .25)]:
            d += weight * (query[col].to_numpy()[:,None] != reference[col].to_numpy()[None,:])
        return d

    @staticmethod
    def masses(distances, times, old, sizes):
        order = np.argsort(distances, axis=1, kind='stable')[:, :max(sizes)]
        ds = np.take_along_axis(distances, order, axis=1)
        tt = times[order]
        out = []
        for n in sizes:
            w = np.exp(-(ds[:,:n] - ds[:,:1]) / np.maximum(ds[:,n-1:n],1e-9))
            w /= w.sum(axis=1, keepdims=True)
            out.append(np.sum(w * (np.abs(tt[:,:n]-old[:,None]) <= .5), axis=1))
        return np.column_stack(out), ds

    def evidence(self, release, challenge, old):
        c = canonical(release)
        old = np.asarray(old, float)
        if old.shape != (len(challenge),) or not np.isfinite(old).all() or (old < 0).any():
            raise ValueError('Invalid baseline AIA')
        ct, bt = c.time.to_numpy(float), self.bg.time.to_numpy(float)
        if min(len(c),len(self.bg)) < 64 or not np.isfinite(ct).all() or not np.isfinite(bt).all():
            raise ValueError('Invalid reference')
        # The abstention sentinel needs a known endpoint, not a presumed one.
        if np.any((ct < 0)|(ct > 10)) or np.any((bt < 0)|(bt > 10)):
            raise ValueError('Rv12 requires the contest time range [0,10]')
        dc = self.distances(challenge,c,numeric(c)/self.scale)
        db = self.distances(challenge,self.bg,self.xbg)
        pc, cs = self.masses(dc, ct, old, [8,32])
        pb, bs = self.masses(db, bt, old, [16,64])
        # Approximate equal mass for different reference sample sizes.
        k = min(len(self.bg), max(1,int(round(len(self.bg)/len(c)))))
        rb = np.partition(db,k-1,axis=1)[:,k-1]
        ratio = (cs[:,0]+1e-12)/(rb+1e-12)
        separation = (cs[:,0]+1e-12)/(cs[:,1]+1e-12)
        linked = (ratio < .1) | ((separation < .1)&(ratio < .5))
        return {'pc': pc, 'pb': pb, 'ratio': ratio, 'strong_link': linked}


def veto(old, evidence, settings):
    old = np.asarray(old,float)
    if settings is None:
        return old.copy(), np.zeros(len(old),bool)
    pc, pb = evidence['pc'], evidence['pb']
    remove = ((old <= 10.) & ~evidence['strong_link']
              & (evidence['ratio'] > settings['ratio_min'])
              & np.all(pb >= settings['background_min'],axis=1)
              & np.all(pc-pb <= settings['excess_max'],axis=1))
    if settings['censor_only']:
        remove &= old >= 9.5
    return np.where(remove,ABSTAIN,old), remove


def predict_all(*, candidates, background, targets, baseline_mia, baseline_aia, known_negative):
    keys = sorted(targets)
    if set(keys) != set(baseline_mia) or set(keys) != set(baseline_aia):
        raise ValueError('Target mismatch')
    m = membership(np.column_stack([baseline_mia[k] for k in keys]),known_negative)
    engine = AttributeVeto(background)
    output = {name: {'F_mia': pd.DataFrame({'record_id':candidates.record_id.to_numpy(),
                **{str(k):m[:,j] for j,k in enumerate(keys)}}), 'aia_parts': [], 'diagnostics': {}}
              for name in VARIANTS}
    for k in keys:
        item, old = targets[k], np.asarray(baseline_aia[k],float)
        if set(item) != {'release','challenge'}:
            raise ValueError('Only public tables are accepted')
        ev = engine.evidence(item['release'],item['challenge'],old)
        for name, settings in VARIANTS.items():
            answer, mask = veto(old,ev,settings)
            output[name]['aia_parts'].append(pd.DataFrame({'target_k':k,
                'challenge_row_id':item['challenge'].challenge_row_id.to_numpy(),'time_hat':answer}))
            output[name]['diagnostics'][str(k)] = {'suppressed':int(mask.sum()),
                'old_active':int((old<=10).sum()),'strong_links':int(ev['strong_link'].sum())}
    for value in output.values():
        value['F_aia'] = pd.concat(value.pop('aia_parts'),ignore_index=True)
    return output
