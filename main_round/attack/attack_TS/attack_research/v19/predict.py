import joblib
import attack_workspace as workspace

from ..v15.predict import CombinedAttack as EarlierAttack
from ..v12.features import predict as grid_predict


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        self.specialist=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round18-specialist'/'models.joblib')

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v15']=previous
        if inferred=='copula':
            improved=grid_predict(self.specialist,c,background,challenge)['grid_0.25']
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
