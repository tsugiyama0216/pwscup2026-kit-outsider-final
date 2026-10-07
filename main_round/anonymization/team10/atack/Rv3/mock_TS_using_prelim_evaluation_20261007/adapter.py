"""Frozen Rv3 on the actual preliminary team24 public tables; no tuning."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RV3 = HERE.parent
ROOT = RV3.parents[4]
MASK_SOURCE = ROOT / 'main_round/attack/Attack_M/v1/actual.py'
MASK_SHA = '3a9b3ce631d17f60192c1815bf73d9bdfeaba64b415ad4c7f6b4ff0ef49f5393'
MODEL_NAME = 'Rv3 frozen / preliminary team24 adapter'

def sha(path):
    if Path(path).name == 'token.txt':
        raise ValueError('Never read or hash credentials')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_frozen():
    selection = json.loads((RV3/'selection.json').read_text(encoding='utf-8'))
    dependencies = json.loads((RV3/'production_dependencies.json').read_text(encoding='utf-8'))
    if sha(RV3/'selection.json') != dependencies['selection_sha256']:
        raise ValueError('Frozen selection changed')
    sources = {**selection['source_sha256'],
               **json.loads((RV3/'production_sources.json').read_text(encoding='utf-8'))}
    for name, expected in sources.items():
        if sha(RV3/name) != expected:
            raise ValueError('Frozen Rv3 source changed: '+name)
    if sha(MASK_SOURCE) != MASK_SHA:
        raise ValueError('Frozen public-mask implementation changed')
    if (selection['mia'], selection['aia']) != ('full5', 'forest_contrast5_1.0'):
        raise ValueError('Unexpected frozen Rv3 components')
    return selection, sources

@contextmanager
def no_data_file_reads():
    """Audit trusted inference; does not claim OS filesystem isolation."""
    state = {'active': True, 'blocked': []}
    def audit(event, args):
        if event != 'open' or not state['active'] or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in {'.csv','.zip','.joblib','.pkl','.pickle','.parquet','.npz'}:
            state['blocked'].append(str(path))
            raise RuntimeError('Inference must use only the supplied public tables')
    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state['active'] = False

def predict_world(*, candidates, self_labels, background, self_challenge,
                  targets, schema, cfg, attacker_id=24,
                  self_release=None, utility_reference=None):
    selected, sources = check_frozen()
    if attacker_id != 24 or len(targets) != 20 or attacker_id in targets:
        raise ValueError('Expected preliminary team24 and 20 other targets')
    if len(candidates) != 1049 or candidates.record_id.duplicated().any():
        raise ValueError('Invalid preliminary candidates')
    # Do not import Rv3 world/run/confirm modules: they initialize main-round tools.
    sys.path.insert(0, str(RV3))
    from selected import predict_selected
    import model
    import selected as numeric
    if Path(model.__file__).resolve() != RV3/'model.py' or Path(numeric.__file__).resolve() != RV3/'selected.py':
        raise ImportError('Wrong Rv3 numeric module')
    spec = importlib.util.spec_from_file_location('_rv3_prelim_public_mask', MASK_SOURCE)
    mask_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_module)
    with no_data_file_reads() as audit:
        mask, proof = mask_module.derive_private_mask(
            candidates.astype(str), self_labels.astype(str), self_challenge.astype(str),
            schema['aia_qi_columns'], schema['submitted_c_columns'])
        fm = pd.DataFrame({'record_id': candidates.record_id.to_numpy()})
        parts = []
        for k, item in sorted(targets.items()):
            if set(item) != {'release','challenge'}:
                raise ValueError('Unexpected target fields')
            q = item['challenge']
            answer = predict_selected(selected, c=item['release'], background=background,
                                      candidates=candidates, challenge=q, cfg=cfg)
            m, a = np.asarray(answer['mia']).copy(), np.asarray(answer['aia'])
            if (m.shape != (1049,) or a.shape != (len(q),) or len(q) != 326
                or not np.isfinite(m).all() or not np.isfinite(a).all()
                or (m < 0).any() or (m > 1).any() or (a < 0).any()):
                raise ValueError('Invalid prediction values or shapes')
            m[mask] = 0.
            fm[str(k)] = m
            parts.append(pd.DataFrame({'target_k':k,
                'challenge_row_id':q.challenge_row_id.to_numpy(),'time_hat':a}))
            print('Frozen Rv3 preliminary target', k, 'complete', flush=True)
    if audit['blocked']:
        raise RuntimeError('A dependency suppressed an input audit failure')
    if check_frozen()[1] != sources:
        raise ValueError('Frozen sources changed during prediction')
    return {'F_mia':fm, 'F_aia':pd.concat(parts,ignore_index=True),
            'diagnostics':{'adapter_only':True,'algorithm_or_parameters_changed':False,
                'attacker_id':24,'target_count':20,'private_mask':proof,
                'selection':{k:selected[k] for k in ['mia','aia']},
                'selection_sha256':sha(RV3/'selection.json'),
                'verified_Rv3_sources':sources,'mask_source_sha256':MASK_SHA,
                'input_scope':'Supplied preliminary public tables only',
                'blocked_data_file_reads':audit['blocked'],
                'stored_fitted_assets_used':False,'official_submission_performed':False}}
