"""Correct weak-jitter AIA event times only for high-margin public QI matches."""
import json

import attack_workspace as workspace
from ..v65.predict import CombinedAttack as EarlierAttack
from ..v69.direct import variants


class CombinedAttack:
    def __init__(self):
        self.previous = EarlierAttack()
        self.schema = json.loads((workspace.ORIGINALS / 'schema.json').read_text())

    def predict(self, *, c, background, candidates, challenge, cfg):
        answers, inferred = self.previous.predict(c=c, background=background,
            candidates=candidates, challenge=challenge, cfg=cfg)
        previous = answers['combined']
        answers['previous_v65'] = previous
        if inferred == 'jitter':
            improved = variants(c, background, challenge, self.schema, 0.1,
                                previous['aia'], [5.0])['event_active_margin5']
            answers['combined'] = {'mia': previous['mia'], 'aia': improved}
        return answers, inferred
