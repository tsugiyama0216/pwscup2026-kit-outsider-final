"""Separate public-only child inference from official scoring and model selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import argparse, json, sys, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
MOCK=ROOT/'main_round/mock_test'
sys.path.insert(0,str(MOCK))
import numpy as np
import pandas as pd
from mock_TS.common import read_json,write_json,read_csv,cfg

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(): return {n:sha(HERE/n) for n in ['model.py','protocol.json','experiment.py']}

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['child','development','confirmation']);p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.stage=='child':
        from model import candidates_all
        b=args.bundle
        public=dict(c=read_csv(b/'C.csv'),background=read_csv(b/'A_bg.csv'),candidates=read_csv(b/'B.csv'),challenge=read_csv(b/'aia_challenge_0.csv'),cfg=read_json(b/'cfg.json'))
        # Model is pure numerical code and cannot load old assets or hidden tables.
        import model
        import inspect
        if any(t in inspect.getsource(model) for t in ['read_csv','read_text','joblib','open(']): raise ValueError('Unexpected model IO')
        a=candidates_all(**public)
        np.savez(args.output,**{f'{part}__{name}':v for part,values in a.items() for name,v in values.items()})
        return
    from mock_TS.runner import load_suite,find_case,export_bundle
    from mock_TS.scoring import estimate_floor,score_predictions,_aia_result
    from pwscup2026.scoring.attack_score import score_mia
    stage=args.stage
    out=MOCK/'local-output/mock_TS'/f'Rv3_{stage}'
    if out.exists(): raise ValueError('Preserve previous outputs')
    out.mkdir(parents=True)
    (HERE/'evidence').mkdir(exist_ok=True)
    source=hashes()
    if stage=='development': suite=MOCK/'local-output/mock_TS/rv2_20261007/suite';selection=None
    else:
        selection=read_json(HERE/'selection.json')
        if selection['source_sha256']!=source: raise ValueError('Frozen source changed')
        from mock_TS.generation import generate
        suite=out/'suite'; spec=read_json(HERE/'protocol.json')['confirmation']
        generate(suite,**spec)
    suite,manifest=load_suite(suite)
    write_json(out/'started.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,suite_manifest=sha(suite/'manifest.json'),selection_sha256=sha(HERE/'selection.json') if selection else None))
    records=[];jobs=[]
    config=cfg()
    for ci,item in enumerate(manifest['cases']):
        case_id=item['id'];case=find_case(suite,manifest,case_id)
        floor=estimate_floor(case,manifest['seed']+ci+1000003,1000)
        write_json(out/case_id/'floor.json',floor)
        b=read_csv(case/'public/B.csv');q=read_csv(case/'public/aia_challenge_0.csv')
        for rel in manifest['releases']:
            name=rel['name']; work=out/case_id/name
            bundle=export_bundle(suite,case_id,name,work/'input')
            jobs.append((case_id,name,case,work,bundle,b,q,floor))
    def execute(job):
        case_id,name,case,work,bundle,b,q,floor=job
        with (work/'child.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(HERE/'experiment.py'),'child','--bundle',str(bundle),'--output',str(work/'options.npz')],check=True,stdout=log,stderr=subprocess.STDOUT)
        return job
    with ThreadPoolExecutor(max_workers=3) as pool:
        for completed in as_completed([pool.submit(execute,job) for job in jobs]):
            case_id,name,case,work,bundle,b,q,floor=completed.result()
            predictions=np.load(work/'options.npz')
            mt=read_csv(case/'private/mia_truth.csv').set_index('record_id').loc[b.record_id]
            at=read_csv(case/'private/aia_truth.csv').set_index('challenge_row_id').loc[q.challenge_row_id].reset_index()
            for key in predictions.files:
                part,option=key.split('__');v=predictions[key]
                if selection and option not in [selection[part],{'mia':'full5','aia':'nearest_soft'}[part]]:continue
                if part=='mia': score=score_mia(v,mt.member.to_numpy(int),config,rare_mask=mt.is_rare.to_numpy(bool)).score
                else:score=float(np.clip(_aia_result(v,at,config).score-floor['phi'],0,1))
                records.append(dict(case=case_id,release=name,component=part,option=option,score=score))
            pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
            print(stage,case_id,name,'complete',flush=True)
    frame=pd.DataFrame(records)
    means={part:frame[frame.component.eq(part)].groupby('option').score.mean().to_dict() for part in ['mia','aia']}
    if hashes()!=source: raise ValueError('Source changed')
    if selection is None:
        selection={part:max(values,key=values.get) for part,values in means.items()}
        selection.update(source_sha256=source,frozen_utc=datetime.now(timezone.utc).isoformat(),confirmation_not_started=True)
        write_json(HERE/'selection.json',selection)
    exact=[]
    for case_id,name,case,work,bundle,b,q,floor in jobs:
        a=np.load(work/'options.npz')
        fm=pd.DataFrame({'record_id':b.record_id,'0':a['mia__'+selection['mia']]})
        fa=pd.DataFrame({'target_k':0,'challenge_row_id':q.challenge_row_id,'time_hat':a['aia__'+selection['aia']]})
        for n,f in [('F_mia.csv',fm),('F_aia.csv',fa)]: f.to_csv(work/n,index=False,float_format='%.17g')
        result=score_predictions(case,read_csv(work/'F_mia.csv'),read_csv(work/'F_aia.csv'),floor,work/'selected_official_score')
        expected=sum(frame[(frame['case']==case_id)&(frame.release==name)&(frame.component==part)&(frame.option==selection[part])].score.item() for part in ['mia','aia'])
        if abs(result['score_1']-expected)>1e-12:raise ValueError('Direct and full official scoring differ')
        exact.append(dict(case=case_id,release=name,**{k:result[k] for k in ['score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']}))
    pd.DataFrame(exact).to_csv(HERE/'evidence'/f'{stage}_official.csv',index=False)
    summary=dict(completed=True,stage=stage,means=means,selection=selection,selected_total=sum(means[k][selection[k]] for k in means),source_sha256=source,official_score_measured=False,suite_manifest_sha256=sha(suite/'manifest.json'))
    write_json(HERE/(stage+'.json'),summary)
    (HERE/'evidence').mkdir(exist_ok=True)
    (HERE/'evidence'/f'{stage}_scores.csv').write_bytes((out/'scores.csv').read_bytes())
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
