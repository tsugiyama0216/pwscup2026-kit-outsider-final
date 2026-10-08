"""Matched mock_TS comparison, development selection then frozen confirmation."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import runtime as rt
import pandas as pd
from mock_TS.common import read_csv,read_json,sha256
from mock_TS.generation import generate
from mock_TS.runner import load_suite,export_bundle,find_case,verify_bundle
from mock_TS.scoring import estimate_floor,score_predictions,validate_predictions
OUT=rt.MOCK/'local-output/mock_TS/pl2_20261008'
NAMES=['Rv2','Rv4','Tv101','Tv102','Tv103','Tv104']

def hashes():
    return {p.name:rt.sha(p) for p in sorted(rt.HERE.iterdir()) if p.suffix in ('.py','.json') and p.name not in ('selection.json','verification.json','publication.json')}

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    path=rt.HERE/'evaluation/source_freeze.json'
    if path.exists():
        if read_json(path)['source_hashes']!=hashes():raise ValueError('Experiment source changed after freeze')
    else:rt.write(path,{'source_hashes':hashes(),'dependencies':rt.dependencies(),
        'rv4_sources':{str(p.relative_to(rt.ROOT)):rt.sha(p) for p in [rt.RV4/'model.py',rt.RV4/'evaluation/mock_TS_comparison_20261008/reproduction/adapter.py']}})

def task(suite,phase,case,release):
    base=OUT/phase/'runs'/case/release;result=base/'results.json'
    if result.exists():return read_json(result)
    bundle=base/'input'
    if not bundle.exists():export_bundle(suite,case,release,bundle)
    _,meta=verify_bundle(bundle)
    work=OUT/'workers'/uuid.uuid4().hex;work.mkdir(parents=True)
    for n in meta['files']:shutil.copyfile(bundle/n,work/n)
    base.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    with (base/'attack.log').open('w',encoding='utf-8') as f:
        proc=subprocess.run([sys.executable,'-B',str(rt.HERE/'worker.py')],cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=3600)
    if proc.returncode:raise RuntimeError('Worker failed: '+str(base/'attack.log'))
    for n,h in meta['files'].items():
        if rt.sha(work/n)!=h:raise ValueError('Public input changed')
    if (base/'predictions').exists():raise ValueError('Refuse duplicate prediction output')
    shutil.copytree(work/'predictions',base/'predictions')
    # Delete only the checked, randomly named workspace created above.
    if work.resolve().parent!=(OUT/'workers').resolve():raise ValueError('Unsafe cleanup')
    shutil.rmtree(work)
    _,manifest=load_suite(suite);case_dir=find_case(suite,manifest,case)
    floor=read_json(OUT/phase/'floors'/f'{case}.json');rows=[]
    for name in NAMES:
        pred=base/'predictions'/name
        fm,fa=read_csv(pred/'F_mia.csv'),read_csv(pred/'F_aia.csv')
        validate_predictions(fm,fa,read_csv(bundle/'B.csv'),read_csv(bundle/'aia_challenge_0.csv'))
        scored=score_predictions(case_dir,fm,fa,floor,base/'scoring'/name)
        scored.update(attack=name,release=release,case_id=case,phase=phase,input_fingerprint=meta['input_fingerprint'],
            prediction_sha256={n:rt.sha(pred/n) for n in ['F_mia.csv','F_aia.csv']})
        rows.append(scored)
    rt.write(result,rows)
    print(phase,case,release,'complete',flush=True)
    return rows

def run(phase,jobs):
    initialize();protocol=read_json(rt.HERE/'protocol.json')
    if phase=='confirmation' and not (rt.HERE/'selection.json').exists():raise ValueError('Freeze development selection first')
    suite=OUT/phase/'suite'
    if not (suite/'manifest.json').exists():
        generate(suite,seed=protocol[phase+'_seed'],count=3,selection_seed=20261007,r_count=10)
    _,manifest=load_suite(suite)
    for i,case in enumerate(manifest['cases']):
        p=OUT/phase/'floors'/f"{case['id']}.json"
        if not p.exists():
            print(phase,case['id'],'shared phi 1000',flush=True)
            rt.write(p,estimate_floor(find_case(suite,manifest,case['id']),protocol[phase+'_seed']+i+1000003,1000))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        pending=[pool.submit(task,suite,phase,c['id'],r['name']) for c in manifest['cases'] for r in manifest['releases']]
        for f in concurrent.futures.as_completed(pending):rows.extend(f.result())
    rows=sorted(rows,key=lambda r:(r['case_id'],r['release'],r['attack']))
    dest=rt.HERE/'evaluation'/phase;dest.mkdir(parents=True,exist_ok=True)
    rt.write(dest/'results.json',rows)
    # Scorer's score_1 and effective powers are extracted below without redefinition.
    flat=[]
    for r in rows:
        record={k:r[k] for k in ['attack','release','case_id','phase']}
        for k in ['score_1','score_2','score_3','score_4','score_5','mia_power','aia_power']:
            record[k]=r[k]
        flat.append(record)
    df=pd.DataFrame(flat);df.to_csv(dest/'scores.csv',index=False)
    keys=['score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']
    means=df.groupby('attack')[keys].mean().reindex(NAMES);means.to_csv(dest/'summary.csv')
    df.groupby(['release','attack'])[keys].mean().to_csv(dest/'by_release.csv')
    rt.write(dest/'summary.json',{'means':means.to_dict(orient='index'),'conditions_per_attack':36,
        'suite_manifest_sha256':rt.sha(suite/'manifest.json'),'source_freeze_sha256':rt.sha(rt.HERE/'evaluation/source_freeze.json'),
        'limitations':protocol['limitations']})
    print(means.to_string(),flush=True)
    synthesize_phase(phase,suite,manifest,protocol)

def synthesize_phase(phase,suite,manifest,protocol):
    from synthesis import MIA,TV,mia_blend,aia_consensus,predict
    conditions=[]
    for case in manifest['cases']:
        case_id=case['id'];case_dir=find_case(suite,manifest,case_id)
        floor=read_json(OUT/phase/'floors'/f'{case_id}.json')
        for release in manifest['releases']:
            base=OUT/phase/'runs'/case_id/release['name']
            answers={n:{'mia':read_csv(base/'predictions'/n/'F_mia.csv')['0'].to_numpy(),
                        'aia':read_csv(base/'predictions'/n/'F_aia.csv').time_hat.to_numpy()} for n in TV}
            fm=read_csv(base/'predictions'/'Rv2'/'F_mia.csv');fa=read_csv(base/'predictions'/'Rv2'/'F_aia.csv')
            conditions.append((case_id,release['name'],case_dir,floor,base,answers,fm,fa))
    def scored(item,a,label):
        ci,re,cd,floor,base,answers,fm,fa=item
        fm=fm.copy();fa=fa.copy();fm['0']=a['mia'];fa['time_hat']=a['aia']
        return score_predictions(cd,fm,fa,floor,base/'synthesis_scoring'/label)
    if phase=='development':
        mi=[];ai=[]
        for j,spec in enumerate(MIA):
            scores=[scored(it,{'mia':mia_blend(it[5],spec),'aia':it[5]['Tv101']['aia']},f'mia{j}')['mia_power'] for it in conditions]
            mi.append({'spec':spec,'mean':sum(scores)/len(scores)})
        for anchor in TV:
            scores=[scored(it,{'mia':it[5]['Tv101']['mia'],'aia':aia_consensus(it[5],anchor)},f'aia{anchor}')['aia_power'] for it in conditions]
            ai.append({'anchor':anchor,'mean':sum(scores)/len(scores)})
        mb=max(mi,key=lambda r:r['mean']);ab=max(ai,key=lambda r:r['mean'])
        selection={'model':'pl2','mia_blend':list(mb['spec']),'aia_anchor':ab['anchor'],
          'development_mia':mb['mean'],'development_aia':ab['mean'],'source_hashes':hashes(),
          'confirmation_used':False,'rule':protocol['selection']}
        p=rt.HERE/'selection.json'
        if p.exists() and read_json(p)!=selection:raise ValueError('Refuse changed selection')
        rt.write(rt.HERE/'evaluation/development/synthesis_search.json',{'mia':mi,'aia':ai})
        rt.write(p,selection)
    selection=read_json(rt.HERE/'selection.json')
    if selection['source_hashes']!=hashes():raise ValueError('Selected source changed')
    rows=[]
    for it in conditions:
        ans=predict(it[5],selection);result=scored(it,ans,'pl2')
        result.update(attack='pl2',case_id=it[0],release=it[1],phase=phase)
        rows.append(result)
        dest=it[4]/'predictions/pl2';dest.mkdir(exist_ok=True)
        fm=it[6].copy();fa=it[7].copy();fm['0']=ans['mia'];fa['time_hat']=ans['aia']
        fm.to_csv(dest/'F_mia.csv',index=False,float_format='%.17g');fa.to_csv(dest/'F_aia.csv',index=False,float_format='%.17g')
    dest=rt.HERE/'evaluation'/phase
    rt.write(dest/'pl2_results.json',rows)
    fields=['attack','release','case_id','phase','score_1','mia_power','aia_power','score_2','score_3','score_4','score_5']
    df=pd.concat([pd.read_csv(dest/'scores.csv'),pd.DataFrame([{k:r[k] for k in fields} for r in rows])],ignore_index=True)
    df.to_csv(dest/'comparison.csv',index=False)
    mean=df.groupby('attack')[fields[4:]].mean().reindex(NAMES+['pl2']);mean.to_csv(dest/'comparison_summary.csv')
    df.groupby(['release','attack'])[fields[4:]].mean().to_csv(dest/'comparison_by_release.csv')
    print('NEW pl2 synthesis',selection['mia_blend'],selection['aia_anchor'],flush=True)
    print(mean.to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','confirmation']);p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args();run(a.phase,a.jobs)
