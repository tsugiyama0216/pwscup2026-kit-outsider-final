"""Eighth batch: preserve radial magnitudes during tail copula correction."""
import io,json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize

GRID=[dict(radial_strength=r,joint_strength=0.,regularization=.25,covariance_weight=0.,information_strength=0.,preserve_strength=0.,majority_strength=.5,outcome_steps=60,significance_guard=True,tail_strength=t,preserve_tail_radius=True)
      for r in (.25,.5) for t in (.25,.5,.75)]

if __name__=='__main__':
    cfg,_=check_environment(); b=read_training_data(); results=[]
    for settings in GRID:
        c,a=synthesize(b,cfg,**settings)
        c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
        score,val=evaluate_frame(c,b,cfg)
        result={**settings,'utility':score['utility'],'facets':score['facets'],'outcome_audit':a['outcome_calibration'],
                'rare_detail':score['detail']['U_rare'],'eligible':score['utility']>.9193239002532438 and score['facets']['U_valid']>=1.-1e-12}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rare_detail','outcome_audit')}),flush=True)
        eligible=[r for r in results if r['eligible']]
        with (HERE/'radius_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
            stream.write('\n')
