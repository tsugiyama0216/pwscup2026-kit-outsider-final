"""Additional development-only density candidates, before final confirmation."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,subprocess,argparse,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];MOCK=ROOT/'main_round/mock_test';BASE=MOCK/'local-output/mock_TS'
sys.path.insert(0,str(MOCK))
import numpy as np
import pandas as pd
from mock_TS.common import read_json,write_json,read_csv,cfg
from pwscup2026.scoring.attack_score import score_mia
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(job):
    bundle,out=job;out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B',str(HERE/'refine.py'),'child','--bundle',str(bundle),'--output',str(out)],check=True,stdout=log,stderr=subprocess.STDOUT)
    return job
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['child','development']);p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='child':
        from density import candidates_all
        np.savez(a.output,**candidates_all(read_csv(a.bundle/'C.csv'),read_csv(a.bundle/'A_bg.csv'),read_csv(a.bundle/'B.csv')));return
    out=BASE/'Rv3_density_development'
    if out.exists():raise ValueError('Preserve previous run')
    out.mkdir(parents=True)
    source={n:sha(HERE/n) for n in ['density.py','refine.py']}
    write_json(out/'started.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,confirmation_started=False))
    jobs=[(p.parent/'input',out/p.parent.parent.name/p.parent.name/'options.npz') for p in sorted((BASE/'Rv3_development').glob('case_*/*/options.npz'))]
    if len(jobs)!=36:raise ValueError('Wait for all initial development conditions')
    records=[];config=cfg()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(execute,j) for j in jobs]):
            bundle,target=future.result();case_id=target.parent.parent.name;release=target.parent.name
            b=read_csv(bundle/'B.csv');truth=read_csv(BASE/'rv2_20261007/suite'/case_id/'private/mia_truth.csv').set_index('record_id').loc[b.record_id]
            got=np.load(target)
            for name in got.files:
                value=score_mia(got[name],truth.member.to_numpy(int),config,rare_mask=truth.is_rare.to_numpy(bool)).score
                records.append(dict(case=case_id,release=release,component='mia',option=name,score=value))
            pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
            print('density',case_id,release,'complete',flush=True)
    frame=pd.concat([read_csv(HERE/'evidence/development_scores.csv'),pd.DataFrame(records)],ignore_index=True)
    means={part:frame[frame.component.eq(part)].groupby('option').score.mean().to_dict() for part in ['mia','aia']}
    old=read_json(HERE/'selection.json')
    write_json(HERE/'initial_selection.json',old)
    selected={part:max(values,key=values.get) for part,values in means.items()}
    selected.update(source_sha256={**old['source_sha256'],**source},frozen_utc=datetime.now(timezone.utc).isoformat(),confirmation_not_started=True,selection_scope='All initial and density refinement development candidates; confirmation not used')
    if any(sha(HERE/n)!=v for n,v in selected['source_sha256'].items()):raise ValueError('Source changed')
    write_json(HERE/'selection.json',selected)
    frame.to_csv(HERE/'evidence/final_development_scores.csv',index=False)
    write_json(HERE/'final_development.json',dict(completed=True,means=means,selection=selected,selected_total=sum(means[k][selected[k]] for k in means),official_score_measured=False))
    print(selected,flush=True)
if __name__=='__main__':main()
