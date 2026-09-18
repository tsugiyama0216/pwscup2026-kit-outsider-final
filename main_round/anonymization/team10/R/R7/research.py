"""First fixed grid: radial mass, joint dependence, and information calibration."""
import io
import json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize

GRID=[(r,j,l) for r in (0.,.25,.5) for j in (.25,.5) for l in (.25,1.)]

if __name__=='__main__':
    cfg,_=check_environment()
    b=read_training_data()
    results=[]
    for radial,joint,reg in GRID:
        settings={'radial_strength':radial,'joint_strength':joint,'regularization':reg}
        try:
            c,audit=synthesize(b,cfg,**settings)
            c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
            score,val=evaluate_frame(c,b,cfg)
            result={**settings,'utility':score['utility'],'facets':score['facets'],
                    'rare_detail':score['detail']['U_rare'],'eligible':score['utility']>.9193239002532438 and score['facets']['U_valid']>=1.-1e-12}
        except (ValueError,RuntimeError) as e:
            result={**settings,'error':str(e),'eligible':False}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rare_detail'}),flush=True)
        eligible=[r for r in results if r['eligible']]
        with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
            stream.write('\n')
