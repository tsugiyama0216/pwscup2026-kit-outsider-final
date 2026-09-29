"""Apply the frozen rare-positive copula MIA model to public inputs."""
import json

import joblib

import attack_workspace as workspace
from ..v9.features import extract
from ..v31.predict import CombinedAttack as EarlierAttack


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        base=workspace.OUTPUT/'attack-research'
        selection=json.loads((base/'round34'/'selection.json').read_text())
        self.specialist=joblib.load(base/'round33'/'models.joblib')[selection['copula_model']]

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v31']=previous
        if inferred=='copula':
            improved=self.specialist.predict_proba(extract(c,background,candidates))[:,1]
            answers['combined']={'mia':improved,'aia':previous['aia']}
        return answers,inferred
