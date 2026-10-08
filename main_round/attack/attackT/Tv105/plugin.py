"""Selected pl2 entry point for mock_TS's public single-target API."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import runtime as rt
from model import predict_all
from synthesis import predict as synthesize

def predict(*,c,background,candidates,challenge,cfg):
    selection=json.loads((HERE/'selection.json').read_text(encoding='utf-8'))
    for name,h in selection['source_hashes'].items():
        if rt.sha(HERE/name)!=h:raise ValueError('Selected source changed: '+name)
    rt.dependencies()
    schema=json.loads((rt.ROOT/'main_round/anonymization/team10/participant_data/schema.json').read_text(encoding='utf-8-sig'))
    answers,_=predict_all(c=c,background=background,candidates=candidates,challenge=challenge,cfg=cfg,schema=schema)
    return synthesize(answers,selection)
