"""Apply the frozen bootstrap specialist using only public attack inputs."""
import json

import joblib

import attack_workspace as workspace
from ..v12.features import predict as grid_predict
from ..v28.predict import CombinedAttack as EarlierAttack


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        base=workspace.TS_OUTPUT/'attack-research'
        self.specialist=joblib.load(base/'round30-specialist'/'models.joblib')
        self.method=json.loads((base/'round31'/'selection.json').read_text())['bootstrap_method']

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v28']=previous
        if inferred=='bootstrap':
            improved=grid_predict(self.specialist,c,background,challenge)[self.method]
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
