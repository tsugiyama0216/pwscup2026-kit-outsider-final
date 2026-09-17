"""Four joint-transport settings constrained to preserve the R3 Cox predictor."""
import io
import json
import pandas as pd
from evaluation import HERE,check_environment,read_training_data,evaluate_frame
from joint_transport import transport

def main():
    cfg,_=check_environment()
    from pwscup2026.scoring.result import json_safe
    b=read_training_data()
    c=pd.read_csv(HERE.parent/'R3/C.csv',float_precision='round_trip')
    records=[]
    for strength in [0.25,0.5,0.75,1.0]:
        label=f'projected-{strength}'
        folder=HERE/'local-output'/label
        folder.mkdir(exist_ok=False)
        generated,audit=transport(b,c,cfg,strength,0.0,True)
        payload=generated.to_csv(index=False,lineterminator='\n').encode()
        score,validation=evaluate_frame(pd.read_csv(io.BytesIO(payload),float_precision='round_trip'),b,cfg)
        (folder/'C.csv').write_bytes(payload)
        (folder/'report.json').write_text(json.dumps(json_safe({'score':score,'validation':validation,'audit':audit}),indent=2,allow_nan=False),encoding='utf-8')
        record={'label':label,'utility':score['utility'],'facets':score['facets'],'rare':validation['rare_diagnostic']['rare_detected']}
        records.append(record)
        print(json.dumps(record),flush=True)
    (HERE/'local-output/projected_experiments.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
