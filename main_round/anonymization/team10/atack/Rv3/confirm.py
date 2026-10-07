"""Check selected-only inference equivalence, then confirm on new fixed mock cases."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,subprocess,argparse,hashlib
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];MOCK=ROOT/'main_round/mock_test'
sys.path.insert(0,str(MOCK))
import numpy as np
import pandas as pd
from mock_TS.common import read_csv,read_json,write_json
BASE=MOCK/'local-output/mock_TS'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
    s=read_json(HERE/'selection.json')
    for n,h in s['source_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('Frozen selection source changed')
    for n,h in read_json(HERE/'production_sources.json').items():
        if sha(HERE/n)!=h:raise ValueError('Frozen production source changed')
    return s
def execute(job):
    bundle,out=job
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B',str(HERE/'confirm.py'),'child','--bundle',str(bundle),'--output',str(out)],check=True,stdout=log,stderr=subprocess.STDOUT)
    return job
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['child','equivalence','confirmation']);p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    selection=check()
    if args.mode=='child':
        from selected import predict_selected
        b=args.bundle
        result=predict_selected(selection,c=read_csv(b/'C.csv'),background=read_csv(b/'A_bg.csv'),candidates=read_csv(b/'B.csv'),challenge=read_csv(b/'aia_challenge_0.csv'),cfg=read_json(b/'cfg.json'))
        np.savez(args.output,**result);return
    if args.mode=='equivalence':
        out=BASE/'Rv3_selected_equivalence'
        if out.exists():raise ValueError('Preserve prior run')
        jobs=[(p.parent/'input',out/p.parent.parent.name/p.parent.name/'selected.npz') for p in sorted((BASE/'Rv3_development').glob('case_*/*/options.npz'))]
        if len(jobs)!=36:raise ValueError('Expected 36 development conditions')
        with ThreadPoolExecutor(max_workers=3) as pool:
            for bundle,target in pool.map(execute,jobs):
                ref=np.load(bundle.parent/'options.npz');got=np.load(target)
                for part in ['mia','aia']:
                    key=part+'__'+selection[part]
                    if key in ref.files:expected=ref[key]
                    else:expected=np.load(BASE/'Rv3_density_development'/bundle.parent.parent.name/bundle.parent.name/'options.npz')[selection[part]]
                    np.testing.assert_array_equal(got[part],expected)
        write_json(HERE/'equivalence.json',dict(completed=True,conditions=36,selected_predictions_exactly_equal_exhaustive=True,production_sources_sha256=sha(HERE/'production_sources.json')))
        print('Selected-only inference equals exhaustive inference on all 36 cases',flush=True);return
    if not read_json(HERE/'equivalence.json')['completed']:raise ValueError('Equivalence required')
    out=BASE/'Rv3_confirmation'
    if out.exists():raise ValueError('Preserve previous confirmation')
    from mock_TS.generation import generate
    from mock_TS.runner import load_suite,find_case,export_bundle
    from mock_TS.scoring import estimate_floor,score_predictions
    suite=out/'suite';generate(suite,**read_json(HERE/'protocol.json')['confirmation'])
    suite,manifest=load_suite(suite)
    write_json(out/'started.json',dict(utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(HERE/'selection.json'),production_sources_sha256=sha(HERE/'production_sources.json')))
    jobs=[];meta={}
    for ci,item in enumerate(manifest['cases']):
        case=find_case(suite,manifest,item['id']);floor=estimate_floor(case,manifest['seed']+ci+1000003,1000)
        write_json(out/item['id']/'floor.json',floor)
        for release in manifest['releases']:
            work=out/item['id']/release['name'];bundle=export_bundle(suite,item['id'],release['name'],work/'input')
            target=work/'selected.npz';jobs.append((bundle,target));meta[str(target)]=(item['id'],release['name'],case,floor)
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(execute,j) for j in jobs]):
            bundle,target=future.result();case_id,name,case,floor=meta[str(target)]
            a=np.load(target);b=read_csv(bundle/'B.csv');q=read_csv(bundle/'aia_challenge_0.csv')
            fm=pd.DataFrame({'record_id':b.record_id,'0':a['mia']});fa=pd.DataFrame({'target_k':0,'challenge_row_id':q.challenge_row_id,'time_hat':a['aia']})
            for n,f in [('F_mia.csv',fm),('F_aia.csv',fa)]:f.to_csv(target.parent/n,index=False,float_format='%.17g')
            result=score_predictions(case,read_csv(target.parent/'F_mia.csv'),read_csv(target.parent/'F_aia.csv'),floor,target.parent/'scoring_truth')
            write_json(target.parent/'result.json',result)
            records.append(dict(case=case_id,release=name,**{k:result[k] for k in ['score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']}))
            pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
            print('confirmation',case_id,name,'complete',flush=True)
    check();frame=pd.DataFrame(records)
    write_json(HERE/'confirmation.json',dict(completed=True,stage='confirmation',conditions=len(frame),selected_total=float(frame.score_1.mean()),mia_power=float(frame.mia_power.mean()),aia_power=float(frame.aia_power.mean()),selection_sha256=sha(HERE/'selection.json'),production_sources_sha256=sha(HERE/'production_sources.json'),suite_manifest_sha256=sha(suite/'manifest.json'),official_score_measured=False))
    (HERE/'evidence/confirmation_official.csv').write_bytes((out/'scores.csv').read_bytes())
    print(frame[['score_1','mia_power','aia_power']].mean().to_dict(),flush=True)
if __name__=='__main__':main()
