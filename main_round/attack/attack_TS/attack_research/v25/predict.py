import joblib
import attack_workspace as workspace

from ..v23.predict import CombinedAttack as EarlierAttack
from ..v24.features import extract


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        self.baseline=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round02'/'models.joblib')['mia']
        self.specialist=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round24'/'models.joblib')['shallow']

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v23']=previous
        if inferred=='strong_jitter':
            improved=self.specialist.predict_proba(extract(c,background,candidates,self.baseline))[:,1]
            answers['combined']={'mia':improved,'aia':previous['aia']}
        return answers,inferred
