import joblib
import attack_workspace as workspace

from ..v19.predict import CombinedAttack as EarlierAttack
from ..v22.adaptive import predict as adaptive_predict


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()
        self.specialist=joblib.load(workspace.TS_OUTPUT/'attack-research'/'round12'/'models.joblib')

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v19']=previous
        if inferred=='strong_jitter':
            improved=adaptive_predict(self.specialist,c,background,challenge)
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
