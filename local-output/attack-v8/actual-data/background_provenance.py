"""Trace released rows to actual/toy public backgrounds without private labels."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from attack_phase_v5.aia_structured_ensemble import copula_event_links
from attack_phase_v8.event_additions import strict_aia_frame
from audit_actual_inputs import ACTUAL,TARGETS

OUT=Path(__file__).resolve().parent
VIEWS={
    'full':('age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death'),
    'static':('age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG'),
    'qi':('age','SBP','TG','HDL','ALT','sex','prefecture'),
    'stable3':('age','SBP','TG','sex','prefecture'),
    'stable3_numeric':('age','SBP','TG'),
    'labs2_numeric':('SBP','TG'),
    'clinical3':('BMI','FPG','smoking','sex','prefecture'),
}

def main():
    sources={'actual_bg':pd.read_csv(ACTUAL/'A_bg.csv'),
             'toy_bg':pd.read_csv(ROOT/'participant_data/A_bg.csv'),
             'actual_B':pd.read_csv(ACTUAL/'B.csv')}
    base=strict_aia_frame(pd.read_csv(ROOT/'submissions/prelim/team24/attack-candidate-v7-feedback-stack/F_aia.csv'))
    rows=[]
    details=[]
    for target in sorted(base.target_k.unique()):
        released=pd.read_csv(TARGETS/f'C_{target}.csv')
        challenge=pd.read_csv(TARGETS/f'aia_challenge_{target}.csv')
        masks={}
        for view,columns in VIEWS.items():
            idx=pd.MultiIndex.from_frame(released[list(columns)])
            for name,source in sources.items():
                mask=idx.isin(pd.MultiIndex.from_frame(source[list(columns)]))
                masks[(view,name)]=mask
                rows.append({'target':int(target),'view':view,'reference':name,'matched_rows':int(mask.sum())})
        matched=np.zeros(len(released),bool)
        tolerant_counts={}
        for name in ('actual_bg','toy_bg'):
            matched |= masks[('stable3',name)]
            matched |= masks[('stable3_numeric',name)]
            matched |= masks[('qi',name)]
            columns=['age','SBP','TG']
            reference=sources[name][columns].to_numpy(dtype=float)
            scale=np.maximum(reference.std(axis=0),1.)
            distances,_=NearestNeighbors(n_neighbors=1).fit(reference/scale).kneighbors(released[columns].to_numpy(dtype=float)/scale)
            near=distances[:,0] <= 1e-10
            tolerant_counts[name]=int(near.sum())
            matched |= near
        residual=released.loc[~matched].reset_index(drop=True)
        block=base.loc[base.target_k.eq(target)].set_index('challenge_row_id').loc[challenge.challenge_row_id]
        record={'target':int(target),'stable_or_qi_bg_rows':int(matched.sum()),'tolerant_stable3':tolerant_counts,
                'residual_rows':len(residual),'residual_events':int(((residual.onset+residual.death)>0).sum())}
        if len(residual)>=2:
            for q in (.5,.9):
                times,gate,info=copula_event_links(residual,challenge,margin_quantile=q,horizon_time=10.)
                record[f'q{int(100*q)}_gated']=int(gate.sum())
                added=gate & block.time_hat.eq(11.).to_numpy()
                record[f'q{int(100*q)}_added_ids']=challenge.loc[added,'challenge_row_id'].astype(int).tolist()
        details.append(record)
    pd.DataFrame(rows).to_csv(OUT/'background_provenance.csv',index=False)
    (OUT/'residual_link_inventory.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows).pivot_table(index=['target','reference'],columns='view',values='matched_rows').to_string())
    print(json.dumps([d for d in details if d['stable_or_qi_bg_rows']>0],indent=2))

if __name__=='__main__':
    main()
