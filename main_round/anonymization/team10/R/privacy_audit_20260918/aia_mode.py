"""AIA sensitivity: optimize the actual +/- .5 year loss, including censoring.

This pre-specified model supplement addresses conditional-mean regression's
mismatch with interval-hit accuracy. Parameters are not selected on B scores.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from audit import HERE,R_DIR,DATA,SEED,BOOTSTRAPS,transformer,transform,row_hashes,rare_mask,strata,aia_metrics,clean_json,write_summary


def time_labels(frame):
    labels=np.minimum(39,np.floor(frame.time.to_numpy(float)*4).astype(int))
    labels[(frame.onset.to_numpy()==0)&(frame.death.to_numpy()==0)]=40
    return labels


def modal_time(probabilities,classes):
    centers=np.where(classes==40,10.,classes*.25+.125)
    choices=np.arange(0,10.00001,.125)
    mass=probabilities@((np.abs(centers[:,None]-choices[None,:])<=.5).astype(float))
    # Select larger time on exact probability ties (appropriate for censor mass).
    return choices[len(choices)-1-np.argmax(mass[:,::-1],axis=1)]


def fit_model(x,frame):
    return ExtraTreesClassifier(n_estimators=200,min_samples_leaf=10,max_features=.8,random_state=SEED,n_jobs=2).fit(x,time_labels(frame))


def main():
    r=json.loads((HERE/'audit_results.json').read_text(encoding='utf-8'))
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    b=read(DATA/'B.csv');bs=read(DATA/'B_self.csv');bg=read(DATA/'A_bg.csv')
    true_rare=b[['record_id']].merge(bs[['record_id','is_rare']],on='record_id',validate='one_to_one').is_rare.to_numpy(bool)
    public=bg.loc[~np.isin(row_hashes(bg),row_hashes(b))].reset_index(drop=True)
    order=np.random.default_rng(SEED).permutation(len(public));nr=len(public)//2;nc=(len(public)-nr)//2
    reference=public.iloc[order[:nr]].reset_index(drop=True);controls=public.iloc[order[nr+nc:]].reset_index(drop=True)
    schema=json.loads((DATA/'schema.json').read_text(encoding='utf-8'));qi=schema['aia_qi_columns']
    proc=transformer(reference,qi);rx=transform(proc,reference,qi)
    frames={'members':b,'controls':controls};xx={k:transform(proc,f,qi) for k,f in frames.items()}
    base=fit_model(rx,reference)
    baseline={k:modal_time(base.predict_proba(x),base.classes_) for k,x in xx.items()}
    for v,vr in r['versions'].items():
        c=read(R_DIR/v/'C.csv');model=fit_model(transform(proc,c,qi),c)
        prediction={k:modal_time(model.predict_proba(x),model.classes_) for k,x in xx.items()}
        result={}
        for name,f in frames.items():
            masks=strata(f,rare_mask(f,schema),true_rare if name=='members' else None)
            result[name]={key:aia_metrics(f.time.to_numpy()[m],prediction[name][m],baseline[name][m]) for key,m in masks.items()}
        contrast={}
        for key in ['overall','censored','any_event','onset','death','detected_rare','detected_rare_event']:
            diffs={}
            for name,f in frames.items():
                mask=strata(f,rare_mask(f,schema))[key];y=f.time.to_numpy()
                diffs[name]=((np.abs(y-prediction[name])<=.5).astype(float)-(np.abs(y-baseline[name])<=.5).astype(float))[mask]
            a,d=diffs['members'],diffs['controls']
            if len(a) and len(d):
                rng=np.random.default_rng(SEED)
                draws=[rng.choice(a,len(a),replace=True).mean()-rng.choice(d,len(d),replace=True).mean() for _ in range(BOOTSTRAPS)]
                contrast[key]={'member_gain_minus_control_gain':a.mean()-d.mean(),'bootstrap_95':np.quantile(draws,[.025,.975]),'n_members':len(a),'n_controls':len(d)}
        result['gain_difference_diagnostics']=contrast
        vr['aia']['extra_trees_interval_mode']=result
    r['aia_interval_mode_method']='ExtraTreesClassifier, 200 trees, min leaf 10, max_features .8; quarter-year outcome bins plus censoring atom; choose time maximizing conditional +/- .5-year probability. No target outcomes enter predictors.'
    r=clean_json(r)
    (HERE/'audit_results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_summary(r)
    print('Added interval-mode nonlinear AIA; regenerated summary.')


if __name__=='__main__':
    main()
