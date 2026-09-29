"""Extend frozen v82 with a public-C Gaussian-copula time prior for copula AIA."""
import json

import joblib
import numpy as np

import attack_workspace as workspace
from ..v82.predict import CombinedAttack as PreviousAttack
from ..v83.run import old_probabilities, decide
from ..v87.run import hit_prior


class CombinedAttack:
    def __init__(self):
        self.previous = PreviousAttack()
        base = workspace.OUTPUT/'attack-research'
        self.membership = joblib.load(base/'round27'/'models.joblib')['regularized']
        self.grid = joblib.load(base/'round18-specialist'/'models.joblib')
        selection = json.loads((base/'round88'/'selection.json').read_text())
        self.ridge = float(selection['ridge'])
        self.weight = float(selection['weight'])
        self.mode = selection['mode']

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers, inferred = self.previous.predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        previous = answers['combined']
        answers['previous_v82'] = previous
        if inferred == 'copula':
            member, times, mh, ch = old_probabilities('copula', c, background,
                challenge, None, self.membership, self.grid)
            np.testing.assert_array_equal(decide('copula', member, times, mh, ch),
                previous['aia'])
            prior = hit_prior(c, challenge, times, self.ridge)
            new_mh = (1-self.weight)*mh+self.weight*prior
            new_ch = ch if self.mode == 'member_only' else \
                (1-self.weight)*ch+self.weight*prior
            improved = decide('copula', member, times, new_mh, new_ch)
            answers['combined'] = {'mia': previous['mia'], 'aia': improved}
        return answers, inferred
