from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[1]/'Tv105';sys.path.insert(0,str(P))
import runtime as rt
from model import predict_all
def predict(*,c,background,candidates,challenge,cfg):
    schema=json.loads((rt.ROOT/'main_round/anonymization/team10/participant_data/schema.json').read_text(encoding='utf-8-sig'))
    rt.dependencies()
    answers,_=predict_all(c=c,background=background,candidates=candidates,challenge=challenge,cfg=cfg,schema=schema)
    return answers['Tv102']
