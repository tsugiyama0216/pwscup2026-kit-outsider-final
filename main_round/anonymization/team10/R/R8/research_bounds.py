"""Batch 7: active-group bounds and Cox significance margin."""
import io,json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize

GRID=[dict(majority_strength=m,uncertainty_strength=s,tail_strength=.25,radial_strength=0.,outcome_steps=80,calibrate_rare_rate=True,uncertainty_group='stable_majority',active_range_limits=True,significance_margin=g)
      for m in (.25,.5) for s in (.35,.5) for g in (2.02,2.1)]

if __name__=='__main__':
    cfg,_=check_environment(); b=read_training_data(); results=[]
    for settings in GRID:
        c,a=synthesize(b,cfg,**settings)
        c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
        score,val=evaluate_frame(c,b,cfg)
        result={**settings,'utility':score['utility'],'facets':score['facets'],'rare_detail':score['detail']['U_rare'],
                'eligible':score['utility']>.9568840160028894+1e-9 and score['facets']['U_valid']>=1.-1e-12}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rare_detail'}),flush=True)
        eligible=[r for r in results if r['eligible']]
        with (HERE/'bounds_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2); stream.write('\n')
