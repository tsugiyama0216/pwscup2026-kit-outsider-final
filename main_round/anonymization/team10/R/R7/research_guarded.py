"""Sixth batch: retain source significance classes in the target Cox effects."""
import io,json
import numpy as np
import pandas as pd
from evaluation import HERE,DATA,check_environment,read_training_data,evaluate_frame
from model import synthesize,CONT
from outcomes import calibrate_outcomes

GRID=[dict(radial_strength=r,joint_strength=0.,regularization=.25,covariance_weight=0.,information_strength=0.,preserve_strength=0.,majority_strength=m,outcome_steps=60,significance_guard=True)
      for r in (.1,.25,.5) for m in (.5,1.)]

if __name__=='__main__':
    cfg,_=check_environment(); b=read_training_data(); results=[]
    detector=json.loads((DATA/'schema.json').read_text())['rare_detector']
    for settings in GRID:
        args={k:v for k,v in settings.items() if k not in ('outcome_steps','significance_guard')}
        c,a=synthesize(b,cfg,**args)
        delta=c[CONT].to_numpy()-np.asarray(detector['mu'])
        mask=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',delta,np.linalg.pinv(detector['sigma']),delta)))>=detector['band']['lo']
        c,oa=calibrate_outcomes(b,c,mask,settings['outcome_steps'],significance_guard=True)
        c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
        score,val=evaluate_frame(c,b,cfg)
        result={**settings,'utility':score['utility'],'facets':score['facets'],'outcome_audit':oa,
                'rare_detail':score['detail']['U_rare'],'eligible':score['utility']>.9193239002532438 and score['facets']['U_valid']>=1.-1e-12}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rare_detail','outcome_audit')}),flush=True)
        eligible=[r for r in results if r['eligible']]
        with (HERE/'guarded_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
            stream.write('\n')
