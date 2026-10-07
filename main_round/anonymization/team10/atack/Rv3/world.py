"""Post-selection joint 22-target diagnostics on previously saved Sugiyama worlds."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,importlib.util,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];MOCK=ROOT/'main_round/mock_test';ATTACK=ROOT/'main_round/attack'
sys.path.insert(0,str(MOCK));sys.path.insert(0,str(ATTACK/'official-kit/starter/src'))
sys.path.insert(0,str(HERE))
import numpy as np
import pandas as pd
from selected import predict_selected
from confirm import check

MODEL_NAME='team10/atack/Rv3'
def predict_world(*,candidates,self_labels,background,self_challenge,targets,schema,cfg):
    selection=check()
    spec=importlib.util.spec_from_file_location('_rv3_world_mask',ATTACK/'Attack_M/v1/actual.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mask,proof=module.derive_private_mask(candidates.astype(str),self_labels.astype(str),self_challenge.astype(str),schema['aia_qi_columns'],schema['submitted_c_columns'])
    fm=pd.DataFrame({'record_id':candidates.record_id});parts=[]
    for k,item in sorted(targets.items()):
        a=predict_selected(selection,c=item['release'],background=background,candidates=candidates,challenge=item['challenge'],cfg=cfg)
        a['mia'][mask]=0;fm[str(k)]=a['mia']
        parts.append(pd.DataFrame({'target_k':k,'challenge_row_id':item['challenge'].challenge_row_id,'time_hat':a['aia']}))
        print('Rv3 joint target',k,'complete',flush=True)
    return dict(F_mia=fm,F_aia=pd.concat(parts,ignore_index=True),diagnostics={'mask':proof})

def main():
    from sugiyama_v2.io import load_world,prediction_arguments,snapshot,save_json
    from sugiyama_v2.scoring import score_world
    from sugiyama_v2.runner import per_target_diagnostics
    from mock_TS.common import cfg
    records=[]
    out=MOCK/'local-output/Rv3-joint-confirmation'
    if out.exists():raise ValueError('Preserve completed or partial diagnostics')
    out.mkdir(parents=True)
    selection=check()
    for seed in [2026100721,2026100722]:
        saved=MOCK/'local-output/R-v2/confirmation'/str(seed)
        public,private,manifest,before=load_world(saved/'case')
        result=predict_world(**prediction_arguments(public,cfg()))
        target=out/str(seed);target.mkdir()
        for name in ['F_mia','F_aia']:result[name].to_csv(target/(name+'.csv'),index=False,float_format='%.17g')
        fm=pd.read_csv(target/'F_mia.csv',float_precision='round_trip');fa=pd.read_csv(target/'F_aia.csv',float_precision='round_trip')
        score=score_world(fm,fa,public,saved/'scoring_truth',cfg())
        rows=per_target_diagnostics(fm,fa,public,saved/'scoring_truth',cfg(),manifest['metadata'],score)
        rows['seed']=seed;records.extend(rows.to_dict(orient='records'))
        save_json(target/'score.json',score)
        if snapshot(saved/'case')[1]!=before:raise ValueError('Saved world changed')
    frame=pd.DataFrame(records);frame.to_csv(HERE/'evidence/joint_confirmation.csv',index=False)
    save_json(HERE/'joint_confirmation.json',dict(completed=True,mean_attack_power=float(frame.attack_power.mean()),mean_mia=float(frame.mia_power.mean()),mean_aia=float(frame.aia_power.mean()),conditions='Previously used 2 worlds, 22 targets each, same saved 1000-draw mock floors as Rv2; diagnostic only, no reselection',official_score_measured=False))
    print(frame[['attack_power','mia_power','aia_power']].mean().to_dict(),flush=True)
if __name__=='__main__':main()
