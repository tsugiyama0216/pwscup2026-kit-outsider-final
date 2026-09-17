"""Fixed, bounded R3 ablation grid; no submission packaging or seed search."""
import io
import json
from pathlib import Path
import pandas as pd
from calibration import calibrate
from evaluation import HERE, check_environment, read_training_data, evaluate_frame

def main():
    cfg,_ = check_environment()
    from pwscup2026.scoring.result import json_safe
    b=read_training_data()
    c=pd.read_csv(HERE.parent/'R2/C.csv',float_precision='round_trip')
    records=[]
    for causes in [('onset',),('onset','death')]:
        for strength in [0.5,0.75,1.0]:
            label='-'.join(causes)+'-'+str(strength)
            out=HERE/'local-output'/label
            out.mkdir(exist_ok=False)
            generated,audit=calibrate(b,c,cfg,strength,causes)
            payload=generated.to_csv(index=False,lineterminator='\n').encode()
            generated=pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
            score,validation=evaluate_frame(generated,b,cfg)
            (out/'C.csv').write_bytes(payload)
            report=json_safe({'score':score,'validation':validation,'audit':audit})
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            record={'label':label,'utility':score['utility'],'facets':score['facets'],'clipped':audit['clipped_counts']}
            records.append(record)
            print(json.dumps(record),flush=True)
    (HERE/'local-output/experiments.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
