"""Apply the frozen copula AIA decision using public attack inputs only."""
import json

import joblib

import attack_workspace as workspace
from ..v34.predict import CombinedAttack as EarlierAttack
from ..v36.decision import variants


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        base=workspace.OUTPUT/'attack-research'
        self.membership=joblib.load(base/'round27'/'models.joblib')['regularized']
        self.grid=joblib.load(base/'round18-specialist'/'models.joblib')
        self.method=json.loads((base/'round37'/'selection.json').read_text())['copula_decision']

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v34']=previous
        if inferred=='copula':
            temperature=float(self.method.split('_')[0][1:])
            penalty=[] if self.method.endswith('_ratio') else [float(self.method.split('_')[1][1:])]
            improved=variants(self.membership,self.grid,c,background,challenge,
                [temperature],penalty)[self.method]
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
