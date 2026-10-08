"""Four prespecified public-input variants. No case labels or truth access."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import runtime as rt

def ranks(x): return (rankdata(np.asarray(x,float),method='average')-.5)/len(x)

def root_expert(attack):
    while hasattr(attack,'previous'): attack=attack.previous
    return attack

def shadow_router(background,schema):
    from local_eval.data import make_release,STYLES,FEATURES
    from attack_research.v3.fusion import profile
    x=[];y=[];v=[];vy=[]
    # Only the current reduced public background; candidate people are never fitted.
    bg=background.sort_values(FEATURES,kind='stable').reset_index(drop=True)
    for seed,count,xx,yy in [(101081,6,x,y),(101091,3,v,vy)]:
        for draw in range(count):
            rng=np.random.default_rng(seed+draw)
            order=rng.permutation(len(bg)); cut=min(1049,len(bg)//2)
            target=bg.iloc[order[:cut]];ref=bg.iloc[order[cut:]]
            for style in STYLES:
                c=make_release(target,ref,style,seed+draw,schema)
                xx.append(profile(np,c,ref));yy.append(style)
    x=np.nan_to_num(np.asarray(x));v=np.nan_to_num(np.asarray(v))
    def fit(xx,yy):
        return ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_leaf=2,
                                    random_state=101,n_jobs=1).fit(xx,yy)
    router=fit(x,y);yp=router.predict(v)
    diag={'validation_accuracy':float(np.mean(yp==vy)),'validation':[
        {'actual':a,'predicted':str(b)} for a,b in zip(vy,yp)],'leave_family_out':[]}
    # Exclude an entire generator, not random rows from its outputs.
    for style in STYLES:
        keep=np.asarray(y)!=style; test=np.asarray(vy)==style
        held=fit(x[keep],np.asarray(y)[keep]);p=held.predict_proba(v[test]);s=np.sort(p,axis=1)
        unknown=(s[:,-1]<.6)|((s[:,-1]-s[:,-2])<.15)
        diag['leave_family_out'].append({'held_family':style,'test_releases':int(test.sum()),
          'unknown_recall':float(unknown.mean()),'forced_predictions':held.predict(v[test]).tolist()})
    return router,diag

def uncertainty(router,c,background):
    from attack_research.v3.fusion import profile
    p=router.predict_proba(np.nan_to_num(profile(np,c,background))[None,:])[0]
    s=np.sort(p);unknown=bool(s[-1]<.6 or s[-1]-s[-2]<.15)
    return unknown,{'probabilities':dict(zip(map(str,router.classes_),map(float,p))),
                    'unknown':unknown,'max_probability':float(s[-1]),'margin':float(s[-1]-s[-2])}

def conditional_evidence(c,bg,b):
    cols=['age','BMI','SBP','TG','HDL','ALT','FPG']
    def matrix(f):
        x=f[cols].to_numpy(float).copy()
        for name in ['TG','ALT']:x[:,cols.index(name)]=np.log1p(np.maximum(x[:,cols.index(name)],0))
        return x
    ref=matrix(bg);scale=np.maximum(ref.std(axis=0),1e-6)
    dc=cdist(matrix(b)/scale,matrix(c)/scale,'sqeuclidean')/len(cols)
    db=cdist(matrix(b)/scale,ref/scale,'sqeuclidean')/len(cols)
    result=[];levels=[]
    for i,row in enumerate(b.itertuples()):
        def masks(f,level):
            a=f.sex.to_numpy()==row.sex
            if level<2:a &= f.prefecture.to_numpy()==row.prefecture
            if level==0:a &= np.abs(f.age.to_numpy()-row.age)<=5
            return a if level<3 else np.ones(len(f),bool)
        for level in range(4):
            cm,bm=masks(c,level),masks(bg,level)
            if cm.sum()>=5 and bm.sum()>=5:break
        # k/n normalization reduces unequal reference-size effects in 7 dimensions.
        cd=np.partition(dc[i,cm],4)[4];bd=np.partition(db[i,bm],4)[4]
        score=3.5*np.log((bd+1e-9)/(cd+1e-9))+np.log(bm.sum()/cm.sum())
        result.append(score);levels.append(level)
    return ranks(result),{'matching_level_counts':{str(j):levels.count(j) for j in range(4)}}

def fallback(base,official,unknown):
    if not unknown:return {'mia':ranks(base['mia']),'aia':np.asarray(base['aia']).copy()}
    return {'mia':ranks(.5*ranks(base['mia'])+.5*ranks(official['mia'])),
            'aia':np.asarray(official['aia']).copy()}

def predict_all(*,c,background,candidates,challenge,cfg,schema):
    from attack_research.v88.predict import CombinedAttack
    from local_eval.methods import official
    attack=CombinedAttack();root=root_expert(attack)
    args=dict(c=c,background=background,candidates=candidates,challenge=challenge,cfg=cfg)
    answers,oldroute=attack.predict(**args);old=answers['combined']
    rv2={'mia':ranks(ranks(old['mia'])),'aia':np.asarray(old['aia']).copy()}
    rv4model=rt.load('_pl2_rv4_numeric',rt.RV4/'model.py')
    correction=rt.load('_pl2_rv4_correction',rt.RV4/'evaluation/mock_TS_comparison_20261008/reproduction/adapter.py')
    rv4=correction.corrected(rv4model,c=c,background=background,candidates=candidates,
                           challenge=challenge,old_mia=rv2['mia'],old_aia=rv2['aia'])
    old_unknown,old_diag=uncertainty(root.router,c,background)
    router,router_diag=shadow_router(background,schema)
    from attack_research.v3.fusion import profile
    newroute=str(router.predict(np.nan_to_num(profile(np,c,background))[None,:])[0])
    if newroute==oldroute: updated=old
    else:
        root.router=router
        revised,inferred=attack.predict(**args)
        if inferred!=newroute:raise ValueError('Router replacement failed')
        updated=revised['combined']
    new_unknown,new_diag=uncertainty(router,c,background)
    off=answers.get('official')
    if off is None:off=official(**args)
    tv101={'mia':ranks(updated['mia']),'aia':np.asarray(updated['aia']).copy()}
    tv102=fallback(old,off,old_unknown)
    evidence,match_diag=conditional_evidence(c,background,candidates)
    tv103={'mia':ranks(.75*rv2['mia']+.25*evidence),'aia':rv2['aia'].copy()}
    tv104=fallback(updated,off,new_unknown)
    tv104['mia']=ranks(.75*tv104['mia']+.25*evidence)
    result={'Rv2':rv2,'Rv4':rv4,'Tv101':tv101,'Tv102':tv102,'Tv103':tv103,'Tv104':tv104}
    for name,a in result.items():
        for key,n in [('mia',len(candidates)),('aia',len(challenge))]:
            x=np.asarray(a[key]);assert x.shape==(n,) and np.isfinite(x).all() and (x>=0).all(),(name,key)
        assert (a['mia']<=1).all()
    return result,{'old_route':str(oldroute),'new_route':newroute,'old_confidence':old_diag,
       'new_confidence':new_diag,'router_validation':router_diag,'conditional_mia':match_diag,
       'historical_training_exposure':True,'private_mask_applied':False}
