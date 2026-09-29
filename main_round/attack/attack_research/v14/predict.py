import joblib
import attack_workspace as workspace

from ..v10.predict import PublicAttack
from ..v12.features import predict as grid_predict


class CombinedAttack:
    def __init__(self):
        self.previous=PublicAttack()
        self.specialist=joblib.load(workspace.OUTPUT/'attack-research'/'round12'/'models.joblib')

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers,inferred=self.previous.predict(c=c,background=background,candidates=candidates,
            challenge=challenge,cfg=cfg)
        previous=answers['combined']
        answers['previous_v10']=previous
        if inferred=='strong_jitter':
            stronger=grid_predict(self.specialist,c,background,challenge)['grid_0.5']
            answers['combined']={'mia':previous['mia'],'aia':stronger}
        return answers,inferred
