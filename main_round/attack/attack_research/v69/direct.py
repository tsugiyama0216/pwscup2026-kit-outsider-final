"""Public-QI nearest-release time variants without hidden labels."""
import numpy as np

from ..v67.likelihood import observation_likelihood


def variants(c, background, challenge, schema, strength, previous, margins):
    ll = observation_likelihood(challenge, c, background, schema, strength)
    best = np.argmax(ll, axis=1)
    top_two = np.partition(ll, (-2, -1), axis=1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    event = ((c.onset.to_numpy() + c.death.to_numpy()) > 0)[best]
    direct = np.where(event, c.time.to_numpy(float)[best], 10.)
    active = previous <= 10.
    answers = {'previous_v65': previous,
        'direct_active': np.where(active, direct, 11.),
        'event_active': np.where(active & event, direct, previous),
        'direct_all': direct}
    for threshold in margins:
        use = event & (margin >= threshold)
        answers[f'event_active_margin{threshold:g}'] = np.where(active & use, direct, previous)
        answers[f'event_activate_margin{threshold:g}'] = np.where(use, direct, previous)
    return answers
