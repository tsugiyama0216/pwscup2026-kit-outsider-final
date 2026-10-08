"""Create non-credential, actual team10 CSVs from frozen Tv105."""
import os,sys,argparse,json
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import runtime as rt
import numpy as np
import pandas as pd
from plugin import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():raise ValueError('Choose a new directory under Tv105')
    if not (HERE/'evaluation/confirmation/comparison_summary.csv').exists():raise ValueError('Complete frozen confirmation first')
    rt.dependencies()
    common=rt.load('_pl2_actual',rt.ATTACK/'Attack_R/v1/common.py')
    actual=common.load_actual();w,np_,pd_,validator,cfg,dist,b,bg,cs,qs,*_=actual
    schema=json.loads((w.ORIGINALS/'schema.json').read_text(encoding='utf-8-sig'))
    maskmod=rt.load('_pl2_mask',rt.ATTACK/'Attack_M/v1/actual.py')
    read=lambda p:pd.read_csv(p,dtype=str,keep_default_na=False)
    mask,proof=maskmod.derive_private_mask(read(w.ORIGINALS/'B.csv'),read(w.ORIGINALS/'B_self.csv'),
         read(w.TARGETS/'aia_challenge_10.csv'),schema['aia_qi_columns'],schema['submitted_c_columns'])
    mia=pd.DataFrame({'record_id':b.record_id});aia=[]
    for k in sorted(cs):
        ans=predict(c=cs[k],background=bg,candidates=b,challenge=qs[k],cfg=cfg)
        ans['mia'][mask]=0;mia[str(k)]=ans['mia']
        aia.append(pd.DataFrame({'target_k':k,'challenge_row_id':qs[k].challenge_row_id,'time_hat':ans['aia']}))
        print('Tv105 actual target',k,'complete',flush=True)
    aia=pd.concat(aia,ignore_index=True)
    valid=w.validate_frames(np,pd,validator,mia,aia,cfg,dist)
    out.mkdir(parents=True)
    mia.to_csv(out/'F_mia.csv',index=False,float_format='%.17g');aia.to_csv(out/'F_aia.csv',index=False,float_format='%.17g')
    recheck=w.validate_frames(np,pd,validator,pd.read_csv(out/'F_mia.csv',float_precision='round_trip'),
                              pd.read_csv(out/'F_aia.csv',float_precision='round_trip'),cfg,dist)
    rt.write(out/'verification.json',{'validation':valid,'readback_validation':recheck,'known_negative_proof':proof,
        'selection_sha256':rt.sha(HERE/'selection.json'),'sha256':{n:rt.sha(out/n) for n in ['F_mia.csv','F_aia.csv']},
        'zip_created':False,'official_submission':False,'official_score_measured':False})
if __name__=='__main__':main()
