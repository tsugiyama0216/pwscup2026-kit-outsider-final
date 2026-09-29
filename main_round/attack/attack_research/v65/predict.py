"""Use public Gaussian donor likelihood when C is inferred weak-jitter."""
import json

import numpy as np
from scipy.stats import rankdata

import attack_workspace as workspace
from local_eval.data import FEATURES
from ..v8.likelihood import observation_loglikelihood
from ..v62.predict import CombinedAttack as EarlierAttack


class CombinedAttack:
    def __init__(self):
        self.previous = EarlierAttack()
        self.schema = json.loads((workspace.ORIGINALS / 'schema.json').read_text())

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers, inferred = self.previous.predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        previous = answers['combined']
        answers['previous_v62'] = previous
        if inferred == 'jitter':
            ll = observation_loglikelihood(candidates[FEATURES], c, background,
                self.schema, 0.1)
            confidence = rankdata(ll.max(axis=1), method='average') / len(candidates)
            answers['combined'] = {'mia': np.asarray(confidence, dtype=float),
                                   'aia': previous['aia']}
        return answers, inferred
