import attack_workspace as workspace

from ..v14.predict import CombinedAttack as EarlierAttack
from ..v13.decision import predict as adaptive_predict
from ..v2.features import extract_aia


class CombinedAttack:
    def __init__(self):
        self.previous=EarlierAttack()

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v14']=previous
        if inferred in ('bootstrap','copula'):
            model=self.previous.previous.models['round02']
            improved=adaptive_predict(model,extract_aia,c,background,challenge)
            answers['combined']={'mia':previous['mia'],'aia':improved}
        return answers,inferred
