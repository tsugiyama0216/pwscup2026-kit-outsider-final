"""Predeclared clinical quantile grid, selected by official local utility."""
import io,json,hashlib
import pandas as pd
from threadpoolctl import threadpool_limits
from refinement import HERE,R9,check_environment,read_training_data,evaluate_frame,synthesize

GRID=[dict(phase=p,strength=s,steps=0) for p in [0.,.25,.5,.75,1.] for s in [.5,1.]]

def run(grid=GRID,label='initial'):
    cfg,_=check_environment(); b=read_training_data()
    base=json.loads((R9/'report.json').read_text(encoding='utf-8'))['score']
    output=HERE.parents[3]/'local-output'/'r11_candidates'
    output.mkdir(parents=True,exist_ok=True)
    report=dict(baseline=base,grid=grid,candidates=[],selection='maximize utility with all facets nondecreasing from R9; no AUC selection')
    with threadpool_limits(limits=1):
        for index,settings in enumerate(grid):
            c,audit=synthesize(b,cfg,**settings)
            payload=c.to_csv(index=False,lineterminator='\n').encode()
            c=pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
            score,validation=evaluate_frame(c,b,cfg)
            eligible=score['utility']>base['utility']+1e-9 and all(score['facets'][k]>=v-1e-12 for k,v in base['facets'].items())
            record=dict(index=index,settings=settings,score=score,validation=validation,audit=audit,
                        eligible=eligible,csv_sha256=hashlib.sha256(payload).hexdigest())
            report['candidates'].append(record)
            (output/f'{label}_{index}.csv').write_bytes(payload)
            (HERE/f'{label}_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
            print(json.dumps(dict(index=index,settings=settings,utility=score['utility'],facets=score['facets'],eligible=eligible)),flush=True)

if __name__=='__main__': run()
