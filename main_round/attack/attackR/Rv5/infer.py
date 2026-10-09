import argparse
import runtime as rt
import numpy as np
import pandas as pd
from pathlib import Path
from model import prepare,membership,attribute

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(rt.HERE) or out.exists():raise ValueError('Fresh Rv5 directory required')
    verified=rt.verify();settings=rt.read(rt.HERE/'protocol.json')
    actual=rt.actual();w,_,_,validator,cfg,dist,b,bg,cs,qs,originals,versions=actual
    olddir=rt.ATTACK/'Attack_R/v2/submission'
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    oldm=read(olddir/'F_mia.csv').set_index('record_id').loc[b.record_id]
    olda=read(olddir/'F_aia.csv')
    maskmod=rt.load('_rv5_private',rt.ATTACK/'Attack_M/v1/actual.py')
    schema=rt.read(w.ORIGINALS/'schema.json');strings=lambda p:pd.read_csv(p,dtype=str,keep_default_na=False)
    mask,proof=maskmod.derive_private_mask(strings(w.ORIGINALS/'B.csv'),strings(w.ORIGINALS/'B_self.csv'),strings(w.TARGETS/'aia_challenge_10.csv'),schema['aia_qi_columns'],schema['submitted_c_columns'])
    cache=prepare(b,bg);mia=pd.DataFrame({'record_id':b.record_id});aia=[];diags={}
    for k in sorted(cs):
        oldtimes=olda[olda.target_k==k].set_index('challenge_row_id').loc[qs[k].challenge_row_id].time_hat.to_numpy()
        scores,md=membership(cs[k],b,cache,oldm[str(k)].to_numpy(),mask,settings)
        times,ad=attribute(cs[k],bg,qs[k],oldtimes,settings)
        mia[str(k)]=scores;aia.append(pd.DataFrame({'target_k':k,'challenge_row_id':qs[k].challenge_row_id,'time_hat':times}))
        md['changed']=int(np.sum(scores!=oldm[str(k)].to_numpy()))
        md['baseline_top30_retained']=len(set(np.argsort(-oldm[str(k)].to_numpy())[:30])&set(np.argsort(-scores)[:30]))
        diags[str(k)]={'mia':md,'aia':ad};print('Rv5 target',k,'MIA changes',md['changed'],'AIA changes',ad['changed'],flush=True)
    aia=pd.concat(aia,ignore_index=True)
    valid=w.validate_frames(np,pd,validator,mia,aia,cfg,dist);rt.verify();out.mkdir(parents=True)
    for n,f in [('F_mia.csv',mia),('F_aia.csv',aia)]:
        f.to_csv(out/n,index=False,lineterminator='\n',float_format='%.17g')
    w.validate_frames(np,pd,validator,read(out/'F_mia.csv'),read(out/'F_aia.csv'),cfg,dist)
    rt.write(out/'report.json',{'validation':valid,'originals':originals,'versions':versions,'dependencies_verified':verified,'private_proof':proof,'targets':diags,'mock_evaluation_used':False,'official_score_measured':False})
if __name__=='__main__':main()
