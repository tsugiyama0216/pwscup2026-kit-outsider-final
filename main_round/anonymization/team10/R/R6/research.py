"""Predefined tail-copula grid, one frozen R4 base and no seed search."""
import io
import json
import pandas as pd
from evaluation import HERE, check_environment, read_training_data, evaluate_frame
from copula import synthesize

GRID=[(s,c) for s in (.25,.5,.75,1.) for c in (0.,.25,.5)]

if __name__=='__main__':
    cfg,_=check_environment()
    b=read_training_data()
    results=[]
    for strength,cox in GRID:
        c,audit=synthesize(b,cfg,strength,cox)
        c=pd.read_csv(io.StringIO(c.to_csv(index=False)),float_precision='round_trip')
        score,validation=evaluate_frame(c,b,cfg)
        result={'strength':strength,'cox_strength':cox,'utility':score['utility'],
                'facets':score['facets'],'rare_detail':score['detail']['U_rare'],
                'rare':validation['rare_diagnostic']}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rare_detail'}),flush=True)
        with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
            json.dump({'candidates':results},stream,indent=2)
    eligible=[r for r in results if r['utility']>.8683850877951735 and r['facets']['U_valid']>=1.-1e-12]
    with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump({'candidates':results,'selected':max(eligible,key=lambda r:r['utility']) if eligible else None},stream,indent=2)
        stream.write('\n')
