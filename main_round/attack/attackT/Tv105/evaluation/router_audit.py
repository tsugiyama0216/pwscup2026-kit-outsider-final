"""Audit frozen router on generator-held validation; never used to tune scores."""
import os,sys,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime as rt
import numpy as np,pandas as pd,joblib
from local_eval.data import make_release,STYLES,FEATURES
from attack_research.v3.fusion import profile

def main():
    out=rt.MOCK/'local-output/mock_TS/pl2_20261008'
    router=joblib.load(rt.ATTACK/'Attak_T/local-output/attack-research/round03/router.joblib')
    schema=json.loads((out/'development/suite/case_001/public/schema.json').read_text(encoding='utf-8'))
    rows=[]
    for case in ['case_001','case_002','case_003']:
        bg=pd.read_csv(out/'development/suite'/case/'public/A_bg.csv',float_precision='round_trip').sort_values(FEATURES,kind='stable').reset_index(drop=True)
        for draw in range(3):
            seed=101091+draw;rng=np.random.default_rng(seed);order=rng.permutation(len(bg));cut=min(1049,len(bg)//2)
            target=bg.iloc[order[:cut]];ref=bg.iloc[order[cut:]]
            for style in STYLES:
                c=make_release(target,ref,style,seed,schema)
                observed=str(router.predict(np.nan_to_num(profile(np,c,ref))[None,:])[0])
                rows.append({'case_id':case,'draw':draw,'actual':style,'frozen_T88_predicted':observed,'correct':style==observed})
    df=pd.DataFrame(rows);dest=rt.HERE/'evaluation'
    df.to_csv(dest/'frozen_router_validation.csv',index=False)
    pd.crosstab(df.actual,df.frozen_T88_predicted).to_csv(dest/'frozen_router_confusion.csv')
    rt.write(dest/'frozen_router_summary.json',{'accuracy':float(df.correct.mean()),'validation_releases':len(df),
       'note':'Same public-background generated validation profiles as Tv101. Frozen T historical exposure retained. Actual saved Cs have no one-to-one label in these 7 simple generator classes.'})
    print(df.groupby('actual').correct.mean().to_string())
if __name__=='__main__':main()
