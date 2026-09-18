"""Second batch: weak partial correction or covariance-constrained calibration."""
import io,json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize

GRID=[dict(radial_strength=r,joint_strength=j,regularization=.25,covariance_weight=w,information_strength=s)
      for r in (.1,.25) for j in (.1,.25) for w,s in ((0.,0.),(0.,.25),(6.,1.))]

if __name__=='__main__':
    cfg,_=check_environment(); b=read_training_data(); results=[]
    for settings in GRID:
        try:
            c,a=synthesize(b,cfg,**settings)
            c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
            score,v=evaluate_frame(c,b,cfg)
            result={**settings,'utility':score['utility'],'facets':score['facets'],
                    'rare_detail':score['detail']['U_rare'],'eligible':score['utility']>.9193239002532438 and score['facets']['U_valid']>=1.-1e-12}
        except (ValueError,RuntimeError) as e:
            result={**settings,'error':str(e),'eligible':False}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rare_detail'}),flush=True)
        eligible=[r for r in results if r['eligible']]
        with (HERE/'balanced_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
            stream.write('\n')
