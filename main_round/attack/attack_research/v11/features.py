import numpy as np
from scipy.special import ndtr, logsumexp

from ..features import canonical, distance, top_indices, QI_CONT
from ..v2.features import extract_aia
from local_eval.data import FEATURES, QI


def neighbor_view(query, c, background, columns, logs, scale):
    d = distance(query, c, background, columns, logs)
    indices = top_indices(d, count=16)
    distances = np.take_along_axis(d, indices, axis=1)
    logw = -distances * len(columns) / (2*scale**2)
    weights = np.exp(logw-logsumexp(logw, axis=1)[:, None])
    event = (c.onset.to_numpy()+c.death.to_numpy()) > 0
    time = np.where(event, c.time.to_numpy(float), 10.)[indices]
    return distances, weights, time, event[indices]


def extract(c, background, challenge):
    c, background = canonical(c, FEATURES), canonical(background, FEATURES)
    query = challenge[QI].reset_index(drop=True)
    base, original, _, _ = extract_aia(c, background, query)
    views = [neighbor_view(query, c, background, QI_CONT, False, .5),
        neighbor_view(query, c, background, QI_CONT, True, .5),
        neighbor_view(query, c, background, ['age', 'SBP', 'HDL'], False, .5)]
    reference = neighbor_view(query, background, background, QI_CONT, True, .5)
    proposals = [original]
    membership = [base]
    for distances, weights, time, events in views:
        proposals.append(time[:, :4])
        event_weights = weights*events
        event_mean = (event_weights*time).sum(axis=1)/np.maximum(event_weights.sum(axis=1), 1e-8)
        event_mean = np.where(event_weights.sum(axis=1)>1e-8, event_mean, 10.)
        proposals.append(event_mean[:, None])
        membership.append(np.column_stack([np.log1p(distances[:, 0]), np.log1p(distances[:, 1]),
            weights[:, 0], (weights*events).sum(axis=1), (weights*time).sum(axis=1),
            (weights*(time-event_mean[:,None])**2).sum(axis=1)]))
    # Offsets are learned on noisy shadow answers, not extrapolated from a different candidate set.
    proposals += [original[:, :4]-.5, original[:, :4]+.5]
    times = np.clip(np.column_stack(proposals), 0., 10.)
    x = np.column_stack(membership).astype(np.float32)
    expanded = [np.repeat(x, times.shape[1], axis=0), times.reshape(-1,1),
        np.abs(times[:,:,None]-original[:,None,:]).reshape(-1,original.shape[1]),
        (times>=9.5).astype(float).reshape(-1,1)]
    for view_index, (_, weights, donor_time, events) in enumerate(views+[reference]):
        delta = times[:,:,None]-donor_time[:,None,:]
        exact = (np.abs(delta)<=.5).astype(float)
        event = events[:,None,:]
        window_features = [(weights[:,None,:]*exact).sum(axis=2)]
        for sd in (.5, 1.):
            mass = ndtr((delta+.5)/sd)-ndtr((delta-.5)/sd)
            # Non-events have known censoring time 10; only event times are uncertain.
            mass = np.where(event, mass, exact)
            window_features.append((weights[:,None,:]*mass).sum(axis=2))
        window_features += [(weights[:,None,:]*exact*event).sum(axis=2),
            np.min(np.abs(delta)+np.where(event,0.,100.),axis=2)]
        expanded.append(np.stack(window_features,axis=2).reshape(len(query)*times.shape[1],-1))
    hx = np.column_stack(expanded).astype(np.float32)
    # Repeated identical time proposals must not multiply that answer's training weight.
    rounded = np.round(times,6)
    repeats = (rounded[:,:,None] == rounded[:,None,:]).sum(axis=2)
    weight = 1./repeats
    if not np.isfinite(x).all() or not np.isfinite(hx).all():
        raise ArithmeticError('Nonfinite AIA features')
    return x, times.astype(np.float32), hx, weight.reshape(-1)


def predict(models, c, background, challenge):
    from ..v5.conditioned import visible_conditioning
    x, times, hx, _ = extract(c, background, challenge)
    member = visible_conditioning(challenge, models['membership'].predict_proba(x)[:,1])
    out = {}
    for name, model in models.items():
        if name == 'membership':
            continue
        probability = model.predict_proba(hx).reshape(len(x),times.shape[1],-1)
        index = {int(k): i for i,k in enumerate(model.classes_)}
        mh = probability[:,:,index[3]]/np.maximum(probability[:,:,index[2]]+probability[:,:,index[3]],1e-8)
        ch = probability[:,:,index[1]]/np.maximum(probability[:,:,index[0]]+probability[:,:,index[1]],1e-8)
        for penalty in (.1,.25,.5,.75,1.):
            gain = member[:,None]*mh-penalty*(1-member[:,None])*ch
            chosen = gain.argmax(axis=1)
            score = gain[np.arange(len(x)),chosen]
            selected = times[np.arange(len(x)),chosen].astype(float)
            out[f'{name}_{penalty:g}'] = np.where(score>0,selected,11.)
    return out
