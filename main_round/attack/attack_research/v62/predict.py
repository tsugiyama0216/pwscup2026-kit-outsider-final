"""Apply exact public rounded-record matching only when C is inferred rounded."""
import json

import attack_workspace as workspace
from ..v37.predict import CombinedAttack as EarlierAttack
from ..v61.match import rounded_public_match


class CombinedAttack:
    def __init__(self):
        self.previous = EarlierAttack()
        self.schema = json.loads((workspace.ORIGINALS / 'schema.json').read_text())

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers, inferred = self.previous.predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        previous = answers['combined']
        answers['previous_v37'] = previous
        if inferred == 'rounded':
            answers['combined'] = {
                'mia': rounded_public_match(c, candidates, self.schema),
                'aia': previous['aia']}
        return answers, inferred
