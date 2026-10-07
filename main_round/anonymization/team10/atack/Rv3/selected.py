"""Compute only the frozen selected components; checked against exhaustive search."""
import numpy as np
import hashlib
from model import (NUM,CAT,QI,GRID,rank,numeric,distance,local,canonical,
                   HistGradientBoostingClassifier,ExtraTreesRegressor,LedoitWolf)
_BACKGROUND_FORESTS={}

def predict_selected(selection,*,c,background,candidates,challenge,cfg):
    c=canonical(c);bg=canonical(background)
    levels=sorted(set(c.prefecture)|set(bg.prefecture));cache={}
    def group_ratio(q,columns):
        from collections import Counter
        def keys(f):
            a=f.copy();a['agebin']=(a.age//10).astype(int)
            return list(a[columns].itertuples(index=False,name=None))
        cc,bc=Counter(keys(c)),Counter(keys(bg));n=len(set(cc)|set(bc))
        return np.array([np.log((cc[x]+1)/(len(c)+n))-np.log((bc[x]+1)/(len(bg)+n)) for x in keys(q)])
    groups=[['prefecture'],['prefecture','sex'],['prefecture','sex','smoking'],['prefecture','sex','agebin'],['sex','agebin','onset','death']]
    def mia(name):
        if name in cache:return cache[name]
        if name.startswith(('gmm','kde','density')):
            from density import candidates_all
            result=candidates_all(c,bg,candidates,wanted=name)[name]
        elif name in ['ensemble','boost_mix','category_geometry']:
            parts={'ensemble':['full5','boost7','gaussian'],'boost_mix':['boost7','boost15'],'category_geometry':['category_mix','attributes5']}[name]
            result=rank(sum(mia(x) for x in parts))
        elif name.startswith('category'):
            if name=='category_mix':result=rank(sum(group_ratio(candidates,g) for g in groups[:4]))
            else:result=rank(group_ratio(candidates,groups[int(name[-1])]))
        elif name.startswith('attributes') or name.startswith('full'):
            full=name.startswith('full');cols=NUM if full else NUM[:-1]
            k=int(name.removeprefix('full' if full else 'attributes'))
            dc=distance(candidates,c,bg,cols);db=distance(candidates,bg,bg,cols)
            if full:
                for column in ['smoking','onset','death']:
                    dc+=(candidates[column].to_numpy()[:,None]!=c[column].to_numpy()[None,:])*.5
                    db+=(candidates[column].to_numpy()[:,None]!=bg[column].to_numpy()[None,:])*.5
            a=np.partition(dc,k-1,axis=1)[:,k-1];b=np.partition(db,k-1,axis=1)[:,k-1]
            result=rank(np.log(np.maximum(b,1e-12))-np.log(np.maximum(a,1e-12)))
        elif name=='gaussian':
            cols=NUM[:-1];scale=np.maximum(numeric(bg,cols).std(0),1e-6)
            def logdensity(f):
                est=LedoitWolf().fit(numeric(f,cols)/scale)
                delta=numeric(candidates,cols)/scale-est.location_
                return -.5*np.einsum('ij,jk,ik->i',delta,est.precision_,delta)
            result=rank(logdensity(c)-logdensity(bg))
        elif name.startswith('boost'):
            cols=NUM+[x for x in CAT if x!='prefecture']
            def enc(f):return np.column_stack([numeric(f,cols),f.prefecture.to_numpy()[:,None]==np.asarray(levels)[None,:]])
            x=np.vstack([enc(c),enc(bg)]);y=np.r_[np.ones(len(c)),np.zeros(len(bg))]
            w=np.r_[np.full(len(c),.5/len(c)),np.full(len(bg),.5/len(bg))]*len(y)
            clf=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=int(name[5:]),l2_regularization=10,min_samples_leaf=30,learning_rate=.07,random_state=311,early_stopping=False).fit(x,y,sample_weight=w)
            result=rank(clf.predict_proba(enc(candidates))[:,1])
        else:raise ValueError('Unknown MIA '+name)
        cache[name]=result;return result
    mn=selection['mia'];an=selection['aia'];m=mia(mn)
    if an.startswith('forest_'):
        leaf=int(an.split('_')[1].removeprefix('contrast').removeprefix('mode'))
        def enc(f):return np.column_stack([numeric(f,QI+['sex']),f.prefecture.to_numpy()[:,None]==np.asarray(levels)[None,:]])
        def prob(f):
            hit=(np.abs(f.time.to_numpy()[:,None]-GRID[None,:])<=.500000000001).astype(float)
            x=enc(f)
            key=(leaf,tuple(levels),hashlib.sha256(x.tobytes()+hit.tobytes()).hexdigest())
            est=_BACKGROUND_FORESTS.get(key) if f is bg else None
            if est is None:
                est=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=311,n_jobs=1).fit(x,hit)
                if f is bg:
                    _BACKGROUND_FORESTS.clear();_BACKGROUND_FORESTS[key]=est
            return est.predict(enc(challenge))
        pc=prob(c)
        if an.startswith('forest_mode'):a=GRID[pc.argmax(1)]
        else:
            pb=prob(bg);delta=(pc-float(an.split('_')[2])*pb)/np.maximum(1-pb,.05)
            best=delta.argmax(1);a=np.where(delta[np.arange(len(best)),best]>.05,GRID[best],11.)
    else:
        mode,block=an.split('_');dc=distance(challenge,c,bg,QI,block)
        if mode=='nearest':a=c.time.to_numpy()[dc.argmin(1)]
        else:
            k=int(mode.removeprefix('contrast').removeprefix('mode'));pc=local(dc,c.time.to_numpy(float),k)
            if mode.startswith('mode'):a=GRID[pc.argmax(1)]
            else:
                db=distance(challenge,bg,bg,QI,block);pb=local(db,bg.time.to_numpy(float),k)
                delta=(pc-pb)/np.maximum(1-pb,.05);best=delta.argmax(1)
                a=np.where(delta[np.arange(len(best)),best]>.05,GRID[best],11.)
    return {'mia':m,'aia':a}
