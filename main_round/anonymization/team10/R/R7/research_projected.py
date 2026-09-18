"""Third batch: restore correlations while preserving corrected Cox risk."""
import io,json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize
from joint_transport import transport

GRID=[dict(radial_strength=r,joint_strength=.1,regularization=.25,covariance_weight=0.,information_strength=s)
      for r in (.1,.25) for s in (.3,.4,.5)]

if __name__=='__main__':
    cfg,_=check_environment(); b=read_training_data(); results=[]
    from pwscup2026.scoring.utility import _score_u_spec,_score_u_corr
    for settings in GRID:
        base,a=synthesize(b,cfg,**settings)
        for p in (.25,.5,.75):
            setting={**settings,'preserve_strength':p}
            c,ta=transport(b,base,cfg,strength=p,preserve_risk=True)
            c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
            spec,_=_score_u_spec(c,b)
            corr,_=_score_u_corr(c,b,cfg)
            if min(spec,corr)<.9311957588557712:
                result={**setting,'screened_out':True,'spec_upper_bound':spec,'corr_upper_bound':corr,'eligible':False}
            else:
                score,val=evaluate_frame(c,b,cfg)
                result={**setting,'utility':score['utility'],'facets':score['facets'],
                        'rare_detail':score['detail']['U_rare'],'eligible':score['utility']>.9193239002532438 and score['facets']['U_valid']>=1.-1e-12}
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='rare_detail'}),flush=True)
            eligible=[r for r in results if r['eligible']]
            with (HERE/'projected_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
                json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
                stream.write('\n')
