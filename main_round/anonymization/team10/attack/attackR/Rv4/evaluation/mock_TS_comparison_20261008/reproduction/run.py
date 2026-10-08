"""Matched 36-condition mock_TS rescoring without selecting or modifying models."""
import os,sys,json,hashlib,importlib.util,subprocess,shutil
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(os.environ['RV4_REPOSITORY']).resolve();MOCK=ROOT/'main_round/mock_test'
HERE=Path(__file__).resolve().parent;BASE=MOCK/'local-output/mock_TS'
ARCHIVE=BASE/'rv2_20261007/evaluation_parallel';SUITE=BASE/'rv2_20261007/suite'
OUT=BASE/'Rv2_Rv3_Rv4_comparison_20261008'
sys.path.insert(0,str(MOCK))
import numpy as np
import pandas as pd
from mock_TS import runner,scoring
from mock_TS.common import read_json,read_csv,write_json,sha256,verify_files,code_hashes,public_tables
from adapter import corrected
RV4=ROOT/'main_round/anonymization/team10/attack/attackR/Rv4';RV3=ROOT/'main_round/anonymization/team10/atack/Rv3';RV2=ROOT/'main_round/attack/Attack_R/v2'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def check():
    preservation={}
    for v in [RV2,RV3,RV4]:
        for p in v.glob('*.py'):preservation[str(p)]=sha256(p)
        for p in (v/'submission').glob('*'):
            if p.is_file():preservation[str(p)]=sha256(p)
        s=read_json(v/'selection.json')
        for n,h in s['source_sha256'].items():assert sha256(v/n)==h,(v,n)
    assert (read_json(RV4/'selection.json')['mia'],read_json(RV4/'selection.json')['aia'])==('confident','rescue10')
    for n,h in read_json(RV2/'production_dependencies.json')['files_sha256'].items():assert sha256(ROOT/n)==h,n
    for n,h in read_json(RV3/'production_sources.json').items():assert sha256(RV3/n)==h,n
    return preservation
def adapter_check(bundle,baseline):
    from types import SimpleNamespace
    m=load('_rv4_equivalence',RV4/'model.py');t=public_tables(bundle)
    old={'F_mia':read_csv(baseline/'F_mia.csv'),'F_aia':read_csv(baseline/'F_aia.csv')}
    evidence=[]
    for count in [0,163]:
        mask=np.arange(len(t['candidates']))<count
        fake=SimpleNamespace(ATTACK=Path('.'),load=lambda *args:SimpleNamespace(derive_private_mask=lambda *a:(mask,{'fixture':True})))
        previous=sys.modules.get('runtime');sys.modules['runtime']=fake
        try:
            exact=m.variants(candidates=t['candidates'],self_labels=t['candidates'],background=t['background'],self_challenge=t['challenge'],targets={0:{'release':t['c'],'challenge':t['challenge']}},schema={'aia_qi_columns':[],'submitted_c_columns':[]},cfg=t['cfg'],baseline=old)
        finally:
            if previous is None:sys.modules.pop('runtime',None)
            else:sys.modules['runtime']=previous
        adapted=corrected(m,c=t['c'],background=t['background'],candidates=t['candidates'],challenge=t['challenge'],old_mia=old['F_mia']['0'].to_numpy(),old_aia=old['F_aia'].time_hat.to_numpy(),mask=mask)
        np.testing.assert_array_equal(adapted['mia'],exact['mia']['confident']['0'].to_numpy())
        np.testing.assert_array_equal(adapted['aia'],exact['aia']['rescue10'].time_hat.to_numpy())
        evidence.append({'test_mask_rows':count,'exact_array_equality':True})
    return evidence
def predict_job(job):
    case_id,name,bundle,baseline,expected=job
    wd=OUT/'workers'/uuid4().hex;wd.mkdir(parents=True)
    for n in expected:shutil.copyfile(bundle/n,wd/n)
    (wd/'baseline').mkdir()
    for n in ['F_mia.csv','F_aia.csv']:shutil.copyfile(baseline/n,wd/'baseline'/n)
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
    with (wd/'attack.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B',str(HERE/'worker.py')],cwd=wd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    verify_files(wd,expected)
    return job,wd
def main():
    if OUT.exists():raise ValueError('Preserve prior comparison')
    OUT.mkdir(parents=True)
    preserved=check();_,manifest=runner.load_suite(SUITE)
    assert len(manifest['releases'])==12 and len(manifest['cases'])==3
    archived=read_json(ARCHIVE/'results.json');baseline_rows={(r['case_id'],r['model']):r for r in archived['results'] if r['method']!='background_only'}
    assert len(baseline_rows)==36
    jobs=[];floor_by_case={}
    for item in manifest['cases']:
        cid=item['id'];case=runner.find_case(SUITE,manifest,cid);floor=read_json(ARCHIVE/cid/'floor.json')
        assert floor['repetitions']==1000;scoring._validate_floor(floor,case,scoring.cfg());floor_by_case[cid]=floor
        for release in manifest['releases']:
            name=release['name'];base=ARCHIVE/cid/name;receipt=read_json(base/'predictions/prediction_manifest.json')
            expected=runner.expected_bundle_files(case,release)
            assert receipt['input_fingerprint']==runner._fingerprint(expected)
            assert receipt['plugin_sha256']==sha256(RV2/'mock_TS_evaluation_20261007/adapter.py')
            verify_files(base/'predictions',receipt['prediction_sha256']);verify_files(ROOT,receipt['code_sha256'])
            runner.verify_bundle(base/'input');verify_files(base/'input',expected)
            jobs.append((cid,name,base/'input',base/'predictions',expected))
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'models':['Rv2','Rv3','Rv4'],'conditions':36,'suite_sha256':sha256(SUITE/'manifest.json'),'source_sha256':code_hashes(),'adapter_sources':{p.name:sha256(p) for p in HERE.glob('*.py')},'preserved':preserved,'Rv2_predictions_reused_with_hash_and_input_verification':True,'Rv4_correction_recomputed':True,'Rv3_retrained_from_each_public_input':True,'floor_repetitions':1000,'floor_reestimated':False,'tuning':False,'private_mask_applied':False,'official_submission':False,'Rv4_status':'candidate_not_finalized','limitations':['One-target diagnostic; self labels and self challenge unavailable','Rv2 and Rv4 retain historical T fitted-model exposure; not fresh-background-only training','Synthetic AIA controls and proxy rare labels','Same finite population and previously used suite; not independent official-score evidence']}
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'adapter_equivalence.json',adapter_check(jobs[0][2],jobs[0][3]))
    rows=[];details=[];pids=[];replayed=0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(predict_job,j) for j in jobs]):
            job,wd=future.result();cid,name,bundle,baseline,expected=job;case=runner.find_case(SUITE,manifest,cid)
            pids.append(read_json(wd/'worker.json')['pid'])
            for version,source in [('Rv2',baseline),('Rv3',wd/'Rv3'),('Rv4',wd/'Rv4')]:
                dest=OUT/cid/name/version;dest.mkdir(parents=True)
                for n in ['F_mia.csv','F_aia.csv']:shutil.copyfile(source/n,dest/n)
                r=scoring.score_predictions(case,read_csv(dest/'F_mia.csv'),read_csv(dest/'F_aia.csv'),floor_by_case[cid],dest/'scoring_truth')
                r.update(case_id=cid,model=name,version=version,input_fingerprint=runner._fingerprint(expected))
                if version=='Rv2':
                    for k in ['score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']:assert abs(r[k]-baseline_rows[(cid,name)][k])<1e-12,(cid,name,k)
                    replayed+=1
                write_json(dest/'result.json',r)
                row={'case_id':cid,'model':name,'version':version,**{k:r[k] for k in ['score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']}}
                rows.append(row);details.append(r)
            pd.DataFrame(rows).to_csv(OUT/'scores.csv',index=False)
            print(f'{len(rows)//3}/36 {cid} {name}: '+', '.join(f"{v}={next(x['score_1'] for x in rows[-3:] if x['version']==v):.6f}" for v in ['Rv2','Rv3','Rv4']),flush=True)
    assert preserved==check();runner.load_suite(SUITE)
    frame=pd.DataFrame(rows);means=frame.groupby('version')[['score_1','mia_power','aia_power']].mean()
    bymodel=frame.groupby(['model','version'])[['score_1','mia_power','aia_power']].mean().reset_index();bymodel.to_csv(OUT/'by_model.csv',index=False)
    bycase=frame.groupby(['case_id','version'])[['score_1','mia_power','aia_power']].mean().reset_index();bycase.to_csv(OUT/'by_case.csv',index=False)
    paired=frame.pivot(index=['case_id','model'],columns='version',values='score_1')
    deltas={}
    for v in ['Rv3','Rv4']:
        d=paired[v]-paired.Rv2;deltas[v]={'mean':float(d.mean()),'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'minimum':float(d.min()),'maximum':float(d.max())}
    summary={'completed':True,'means':means.to_dict(orient='index'),'paired_vs_Rv2':deltas,'conditions':36,'models':12,'cases':3,'Rv2_rescored_results_match_archive':replayed,'fresh_correction_and_Rv3_processes':len(pids),'unique_pids':len(set(pids)),'all_frozen_models_and_submissions_preserved':True,'Rv4_status':'candidate_not_finalized','official_score_prediction':False,'limitations':protocol['limitations']}
    write_json(OUT/'summary.json',summary);write_json(OUT/'results.json',details)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
