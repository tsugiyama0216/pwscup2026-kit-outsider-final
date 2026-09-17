"""Eight predefined marginal-transport settings; no seed or row-order search."""
import io
import json
import pandas as pd
from evaluation import HERE, check_environment, read_training_data, evaluate_frame
from transport import transport

def main():
    cfg,_=check_environment()
    from pwscup2026.scoring.result import json_safe
    b=read_training_data()
    c=pd.read_csv(HERE.parent/'R3/C.csv',float_precision='round_trip')
    records=[]
    for scope in ['global','sex']:
        for strength in [0.25,0.5,0.75,1.0]:
            label=f'{scope}-{strength}'
            folder=HERE/'local-output'/label
            folder.mkdir(exist_ok=False)
            generated,audit=transport(b,c,cfg,strength,scope)
            payload=generated.to_csv(index=False,lineterminator='\n').encode()
            score,validation=evaluate_frame(pd.read_csv(io.BytesIO(payload),float_precision='round_trip'),b,cfg)
            (folder/'C.csv').write_bytes(payload)
            (folder/'report.json').write_text(json.dumps(json_safe({'score':score,'validation':validation,'audit':audit}),indent=2,allow_nan=False),encoding='utf-8')
            record={'label':label,'utility':score['utility'],'facets':score['facets'],'rare':validation['rare_diagnostic']['rare_detected']}
            records.append(record)
            print(json.dumps(record),flush=True)
    (HERE/'local-output/experiments.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
