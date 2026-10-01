"""Learn release profiles on training shadows; route preselected attack experts."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import attack_workspace as workspace

BASE=workspace.TS_OUTPUT/'attack-research'
OUT=BASE/'round03'


def profile(np,c,background):
    from ..features import release_summary,CLINICAL
    cols=CLINICAL+['time','onset','death','sex','smoking']
    r=c[cols].to_numpy(float)
    b=background[cols].to_numpy(float)
    scale=np.maximum(b.std(axis=0),1e-6)
    corr=np.nan_to_num(np.corrcoef(r,rowvar=False),nan=0.)
    non_event=c.loc[(c.onset+c.death).eq(0),'time'].to_numpy(float)
    extra=[float(np.mean(non_event<9.9)) if len(non_event) else 0.,
           float(np.mean(non_event<9.5)) if len(non_event) else 0.,
           float(c.time.std())]
    return np.r_[release_summary(c),r.std(axis=0)/scale,
                 (r.mean(axis=0)-b.mean(axis=0))/scale,corr[np.tril_indices(len(cols),-1)],extra]


def provenance():
    from ..run import sources as first
    from ..v2.run import sources as second
    return {**first(),**second(),'attack_research/v3/fusion.py':workspace.sha256(Path(__file__))}


def freeze(np,pd,protocol):
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from local_eval.data import build_case,make_release
    from local_eval.__main__ import save
    from ..run import read_data
    path=OUT/'selection.json'
    if path.exists():raise ValueError('Frozen routing already exists')
    all_rows=[]
    for round_name in ('round01','round02'):
        f=pd.read_csv(BASE/round_name/'development/results.csv')
        f['expert']=round_name+':'+f.method
        all_rows.append(f)
    scores=pd.concat(all_rows,ignore_index=True)
    routes={}
    for task,metric in (('mia','mia'),('aia','aia_no_phi')):
        means=scores[scores.task.eq(task)].groupby(['style','expert'],sort=False)[metric].mean()
        routes[task]={style:str(group.droplevel('style').idxmax())
                      for style,group in means.groupby(level='style',sort=False)}
    source,bg,schema=read_data(pd)
    x,y=[],[]
    for seed in protocol['training_seeds']:
        case=build_case(source,bg,schema,'dev',seed)
        for style in protocol['primary_styles']+protocol['diagnostic_styles']:
            release=make_release(case.target,case.background,style,seed,schema)
            x.append(profile(np,release,case.background));y.append(style)
    router=ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_leaf=2,
                               random_state=20260927,n_jobs=1)
    router.fit(np.asarray(x),y)
    joblib.dump(router,OUT/'router.joblib')
    model_paths={r:BASE/r/'models.joblib' for r in ('round01','round02')}
    save(path,{'routes':routes,'source_sha256':provenance(),
        'router_sha256':workspace.sha256(OUT/'router.joblib'),
        'models_sha256':{k:workspace.sha256(p) for k,p in model_paths.items()},
        'profile_training_cases':len(y),'basis':'Best expert per development style; unknown test style inferred only from public C and background',
        'confirmation_used':False})
    print(json.dumps(routes,ensure_ascii=False,indent=2),flush=True)


def evaluate(np,pd,cfg,protocol,confirmation):
    import joblib
    from local_eval.data import build_case,make_release
    from local_eval.scoring import score_arrays
    from local_eval.methods import official,multiview
    from local_eval.__main__ import save
    from ..run import learned_predictions as first,read_data
    from ..v2.run import learned_predictions as second
    from prelim_transfer.methods import load_legacy,build_aia
    choice=json.loads((OUT/'selection.json').read_text())
    if choice['source_sha256']!=provenance():raise ValueError('Frozen implementation changed')
    for r,h in choice['models_sha256'].items():
        if workspace.sha256(BASE/r/'models.joblib')!=h:raise ValueError('Model changed')
    models={r:joblib.load(BASE/r/'models.joblib') for r in ('round01','round02')}
    router=joblib.load(OUT/'router.joblib')
    source,bg,schema=read_data(pd)
    key='confirmation' if confirmation else 'development'
    output=OUT/key
    if output.exists():raise ValueError('Do not overwrite evaluation')
    output.mkdir()
    modules=load_legacy()
    rows,routes,cases=[],[],[]
    for seed in protocol[key+'_seeds']:
        case=build_case(source,bg,schema,protocol[key+'_split'],seed)
        cases.append(case.metadata)
        for style in protocol['primary_styles']+protocol['diagnostic_styles']:
            c=make_release(case.target,case.background,style,seed,schema)
            observed=str(router.predict(profile(np,c,case.background)[None,:])[0])
            old=official(c=c,background=case.background,candidates=case.candidates,challenge=case.challenge,cfg=cfg)
            mv=multiview(c=c,background=case.background,candidates=case.candidates,challenge=case.challenge,cfg=cfg)
            frames,_=build_aia(modules,case.background,{'0':c},{'0':case.challenge})
            structured=frames['structured'].set_index('challenge_row_id').loc[case.challenge.challenge_row_id,'time_hat'].to_numpy()
            a1,b1=first(np,models['round01'],c,case)
            a2,b2=second(np,models['round02'],c,case)
            experts={'mia':{},'aia':{}}
            for rnd,m,a in (('round01',a1,b1),('round02',a2,b2)):
                m.update(official=old['mia'],multiview=mv['mia'])
                a.update(official=old['aia'],structured=structured)
                for name,pred in m.items():experts['mia'][rnd+':'+name]=pred
                for name,pred in a.items():experts['aia'][rnd+':'+name]=pred
            selected={task:choice['routes'][task][observed] for task in ('mia','aia')}
            predictions={task:experts[task][selected[task]] for task in ('mia','aia')}
            # The fixed single experts are diagnostic comparisons, not holdout selection options.
            comparisons={'routed':predictions,'official':old,
                'multiview_structured':{'mia':mv['mia'],'aia':structured},
                'fixed_learned':{'mia':a1['learned_rule'],'aia':b2['joint_0.25']}}
            for name,pred in comparisons.items():
                scored=score_arrays(pred['mia'],pred['aia'],case,cfg)
                top=np.argsort(-pred['mia'],kind='stable')[:26]
                rows.append({'method':name,'seed':seed,'style':style,**scored,
                             'mia_top26_recall':float(case.mia_truth[top].sum()/26)})
            routes.append({'seed':seed,'actual_style_diagnostic_only':style,'predicted_style':observed,**selected})
            pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
            print(f'Fusion {key} seed={seed} style={style} inferred={observed}',flush=True)
    frame=pd.DataFrame(rows)
    primary=frame[frame['style'].isin(protocol['primary_styles'])]
    summary=primary.groupby('method',sort=False).agg(cases=('mia','size'),mia=('mia','mean'),aia=('aia_no_phi','mean'),
        mia_min=('mia','min'),aia_min=('aia_no_phi','min'),combined=('combined_no_phi','mean'))
    summary.to_csv(output/'summary.csv')
    frame.groupby(['style','method'])[['mia','aia_no_phi']].mean().to_csv(output/'by_style.csv')
    pd.DataFrame(routes).to_csv(output/'routes.csv',index=False)
    scored=primary[primary.method.eq('routed')]
    complete=bool(((scored.mia>=1-1e-12)&(scored.aia_no_phi>=1-1e-12)).all())
    save(output/'verification.json',{'cases':cases,'frozen_selection_sha256':workspace.sha256(OUT/'selection.json'),
        'input_verification':workspace.verify_hashes(),'source_sha256':provenance(),'goal_achieved':complete,
        'primary_cases':len(scored),'both_perfect_cases':int(((scored.mia>=1-1e-12)&(scored.aia_no_phi>=1-1e-12)).sum()),
        'confirmation_used':confirmation,'selection_changed_after_confirmation':False})
    print(summary.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','evaluate','confirm'])
    args=parser.parse_args()
    workspace.verify_hashes()
    np,pd,_,cfg,_=workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    protocol=json.loads((Path(__file__).parent.parent/'protocol.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.action=='freeze':freeze(np,pd,protocol)
        else:evaluate(np,pd,cfg,protocol,args.action=='confirm')


if __name__=='__main__':main()
