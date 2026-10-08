"""Frozen Rv4 public API for both main and preliminary worlds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime as rt
from model import variants
MODEL_NAME='team10/attack/attackR/Rv4'
def predict_world(*,candidates,self_labels,background,self_challenge,targets,schema,cfg,**unused):
    selection=rt.read(rt.HERE/'selection.json')
    for name,h in selection['source_sha256'].items():
        if rt.sha(rt.HERE/name)!=h:raise ValueError('Frozen Rv4 source changed')
    rt.preserve()
    v=variants(candidates=candidates,self_labels=self_labels,background=background,self_challenge=self_challenge,targets=targets,schema=schema,cfg=cfg)
    rt.preserve()
    return {'F_mia':v['mia'][selection['mia']],'F_aia':v['aia'][selection['aia']],'diagnostics':v['diagnostics']}
