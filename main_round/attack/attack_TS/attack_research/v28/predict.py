import joblib
import attack_workspace as workspace

from ..v25.predict import CombinedAttack as EarlierAttack
from ..v27.predict import predict as membership_predict


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        self.membership=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round27'/'models.joblib')['regularized']
        self.grid=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round18-specialist'/'models.joblib')

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v25']=previous
        if inferred=='copula':
            improved=membership_predict(self.membership,self.grid,c,background,challenge)
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
