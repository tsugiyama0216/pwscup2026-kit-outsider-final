"""Extend frozen v70 only for publicly detected bootstrap AIA queries."""
import json

import joblib

import attack_workspace as workspace
from ..v70.predict import CombinedAttack as EarlierAttack
from .blend import predict_bootstrap_aia


class CombinedAttack:
    def __init__(self):
        self.previous = EarlierAttack()
        base = workspace.TS_OUTPUT/'attack-research'
        self.specialist = joblib.load(base/'round30-specialist'/'models.joblib')
        self.prior = joblib.load(base/'round80-public-qi-prior'/'prior.joblib')
        self.weight = float(json.loads((base/'round82'/'selection.json').read_text())['bootstrap_prior_weight'])

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers, inferred = self.previous.predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        previous = answers['combined']
        answers['previous_v70'] = previous
        if inferred == 'bootstrap':
            improved = predict_bootstrap_aia(c, background, challenge, previous['aia'],
                self.specialist, self.prior, self.weight)
            answers['combined'] = {'mia': previous['mia'], 'aia': improved}
        return answers, inferred
