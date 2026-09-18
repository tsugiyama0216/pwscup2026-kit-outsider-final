"""Four predeclared aggregate-loss balances, with unmodified official scoring."""
import io,json,hashlib
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from model import synthesize

GRID=[dict(steps=200,marginal_weight=m,rare_weight=r,correlation_weight=c,information_weight=i,movement_weight=.02,trust_radius=1.)
      for m,r,c,i in [(4.,4.,100.,100.),(10.,4.,300.,100.),(4.,10.,300.,100.),(10.,10.,300.,300.)]]

def run(grid,name):
    cfg,_=check_environment(); b=read_training_data(); results=[]
    baseline=json.loads((HERE/'R8_user_reported_result.json').read_text())['utility_facets']
    for settings in grid:
        c,a=synthesize(b,cfg,**settings)
        data=c.to_csv(index=False,lineterminator='\n').encode('utf-8')
        c=pd.read_csv(io.BytesIO(data),float_precision='round_trip')
        score,validation=evaluate_frame(c,b,cfg)
        eligible=score['utility']>min(baseline.values())+1e-9 and all(score['facets'][k]>=v-1e-12 for k,v in baseline.items())
        result={'settings':settings,'utility':score['utility'],'facets':score['facets'],'detail':score['detail'],
                'eligible':eligible,'audit':a,'validation':validation,'csv_sha256':hashlib.sha256(data).hexdigest()}
        results.append(result)
        print(json.dumps({k:result[k] for k in ['settings','utility','facets','eligible']}),flush=True)
        with (HERE/(name+'_summary.json')).open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results},stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__': run(GRID,'initial')
