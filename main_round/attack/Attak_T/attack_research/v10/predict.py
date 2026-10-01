"""Public-input-only inference for the combined candidate and fixed comparisons."""
from types import SimpleNamespace
import json
import joblib
import numpy as np

import attack_workspace as workspace


class PublicAttack:
    def __init__(self):
        from prelim_transfer.methods import load_legacy
        base = workspace.TS_OUTPUT/'attack-research'
        self.models = {r: joblib.load(base/r/'models.joblib') for r in ('round01', 'round02')}
        self.synthetic = joblib.load(base/'round09'/'models.joblib')['deep']
        self.router = joblib.load(base/'round03'/'router.joblib')
        self.routing = json.loads((base/'round03'/'selection.json').read_text())['routes']
        self.modules = load_legacy()

    def predict(self, *, c, background, candidates, challenge, cfg):
        from local_eval.methods import official, multiview
        from ..run import learned_predictions as first
        from ..v2.run import learned_predictions as second
        from ..v2.features import extract_aia
        from ..v3.fusion import profile
        from ..v5.conditioned import predictions
        from ..v9.features import extract
        from prelim_transfer.methods import build_aia

        # This container deliberately cannot supply truth, source indices or random seeds.
        public = SimpleNamespace(background=background, candidates=candidates, challenge=challenge)
        old = official(c=c, background=background, candidates=candidates, challenge=challenge, cfg=cfg)
        mv = multiview(c=c, background=background, candidates=candidates, challenge=challenge, cfg=cfg)
        inferred = str(self.router.predict(profile(np, c, background)[None, :])[0])
        experts = {}
        for rnd, fn in [('round01', first), ('round02', second)]:
            m, _ = fn(np, self.models[rnd], c, public)
            m.update(official=old['mia'], multiview=mv['mia'])
            experts.update({rnd+':'+name: value for name, value in m.items()})
        previous_mia = experts[self.routing['mia'][inferred]]
        mia = self.synthetic.predict_proba(extract(c, background, candidates))[:, 1] if inferred == 'copula' else previous_mia
        aia = predictions(self.models['round02'], extract_aia, c, background, challenge,
            self.modules['structured'].stratified_half_selection)['conditioned_0.5']
        frames, _ = build_aia(self.modules, background, {'0': c}, {'0': challenge})
        structured = frames['structured'].set_index('challenge_row_id').loc[challenge.challenge_row_id, 'time_hat'].to_numpy()
        return {
            'combined': {'mia': mia, 'aia': aia},
            'previous_v7': {'mia': previous_mia, 'aia': aia},
            'multiview_structured': {'mia': mv['mia'], 'aia': structured},
            'official': old,
        }, inferred
