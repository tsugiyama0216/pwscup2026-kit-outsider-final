"""Public-input bootstrap AIA correction using a person-disjoint time prior."""
import numpy as np

from ..v5.conditioned import visible_conditioning
from ..v12.features import extract
from ..v40.exact import correct


def predict_bootstrap_aia(c, background, challenge, old_aia, specialist, prior, weight):
    x, times, hx, _ = extract(c, background, challenge)
    member = visible_conditioning(challenge,
        specialist['membership'].predict_proba(x)[:, 1])
    model = specialist['grid']
    hp = model.predict_proba(hx).reshape(len(x), times.shape[1], -1)
    index = {int(k): i for i, k in enumerate(model.classes_)}
    mh = hp[:, :, index[3]]/np.maximum(hp[:, :, index[2]]+hp[:, :, index[3]], 1e-8)
    ch = hp[:, :, index[1]]/np.maximum(hp[:, :, index[0]]+hp[:, :, index[1]], 1e-8)
    prior_hit = prior.hit_probability(challenge, times)
    mh_new = (1-weight)*mh+weight*prior_hit
    ch_new = (1-weight)*ch+weight*prior_hit
    gain = member[:, None]*mh_new-.1*(1-member[:, None])*ch_new
    best = gain.argmax(axis=1)
    chosen = times[np.arange(len(x)), best].astype(float)
    candidate_aia = np.where(gain[np.arange(len(x)), best] > 0, chosen, 11.)
    _, exact = correct(c, challenge, old_aia)
    return np.where(exact, old_aia, candidate_aia)
