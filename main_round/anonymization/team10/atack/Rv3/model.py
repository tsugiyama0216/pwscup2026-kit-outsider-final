"""Rv3: fresh public-table density contrasts. No files, labels or fitted assets."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.covariance import LedoitWolf

NUM = ['age','BMI','SBP','TG','HDL','ALT','FPG','time']
CAT = ['sex','prefecture','smoking','onset','death']
QI = ['age','SBP','TG','HDL','ALT']
GRID = np.linspace(0,10,101)

def rank(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite score')
    return (rankdata(x,method='average')-.5)/len(x)

def numeric(f, cols):
    x=f[cols].to_numpy(float).copy()
    if not np.isfinite(x).all(): raise ValueError('Nonfinite public features')
    for k in ['TG','ALT']:
        if k in cols: x[:,cols.index(k)]=np.log1p(np.maximum(x[:,cols.index(k)],0))
    return x

def distance(q,c,bg,cols,block='soft'):
    scale=np.maximum(numeric(bg,cols).std(0),1e-6)
    d=cdist(numeric(q,cols)/scale,numeric(c,cols)/scale,'sqeuclidean')/len(cols)
    for name,penalty in [('sex',2.),('prefecture',.5)]:
        mismatch=q[name].to_numpy()[:,None]!=c[name].to_numpy()[None,:]
        d+=mismatch*(1000. if block=='hard' else penalty)
    return d

def local(d,t,k):
    ix=np.argsort(d,axis=1,kind='stable')[:,:k]
    dd=np.take_along_axis(d,ix,1)
    w=np.exp(-(dd-dd[:,:1])/np.maximum(dd[:,-1:],1e-8))
    w/=w.sum(1,keepdims=True)
    hits=np.abs(t[ix,None]-GRID[None,None,:])<=.500000000001
    return (w[:,:,None]*hits).sum(1)

def canonical(f):
    cols=sorted(set(NUM+CAT))
    return f.sort_values(cols,kind='stable').reset_index(drop=True)

def candidates_all(*,c,background,candidates,challenge,cfg):
    # Sort training rows for deterministic fitting and ties; never reorder queries.
    c=canonical(c); bg=canonical(background)
    mia={}; aia={}
    for cols,label in [(NUM[:-1],'attributes'),(NUM,'full')]:
        dc=distance(candidates,c,bg,cols)
        db=distance(candidates,bg,bg,cols)
        if label=='full':
            for name in ['smoking','onset','death']:
                dc+=(candidates[name].to_numpy()[:,None]!=c[name].to_numpy()[None,:])*.5
                db+=(candidates[name].to_numpy()[:,None]!=bg[name].to_numpy()[None,:])*.5
        for k in [1,5,20]:
            a=np.partition(dc,k-1,axis=1)[:,k-1]
            b=np.partition(db,k-1,axis=1)[:,k-1]
            mia[f'{label}{k}']=rank(np.log(np.maximum(b,1e-12))-np.log(np.maximum(a,1e-12)))
    cols=NUM+[x for x in CAT if x!='prefecture']
    levels=sorted(set(c.prefecture)|set(bg.prefecture))
    def encode(f):
        return np.column_stack([numeric(f,cols), f.prefecture.to_numpy()[:,None]==np.asarray(levels)[None,:]])
    x=np.vstack([encode(c),encode(bg)])
    y=np.r_[np.ones(len(c)),np.zeros(len(bg))]
    weights=np.r_[np.full(len(c),.5/len(c)),np.full(len(bg),.5/len(bg))]*len(y)
    for leaves in [7,15]:
        clf=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=leaves,l2_regularization=10,
                 min_samples_leaf=30,learning_rate=.07,random_state=311,early_stopping=False)
        clf.fit(x,y,sample_weight=weights)
        mia[f'boost{leaves}']=rank(clf.predict_proba(encode(candidates))[:,1])
    cols=NUM[:-1]
    scale=np.maximum(numeric(bg,cols).std(0),1e-6)
    def logdensity(f,q):
        z=numeric(f,cols)/scale
        est=LedoitWolf().fit(z)
        delta=numeric(q,cols)/scale-est.location_
        return -.5*np.einsum('ij,jk,ik->i',delta,est.precision_,delta)
    mia['gaussian']=rank(logdensity(c,candidates)-logdensity(bg,candidates))
    mia['ensemble']=rank(mia['full5']+mia['boost7']+mia['gaussian'])
    mia['boost_mix']=rank(mia['boost7']+mia['boost15'])
    # Smoothed categorical density ratios retain cohort composition differences.
    def group_ratio(q,columns):
        def keys(f):
            a=f.copy();a['agebin']=(a.age//10).astype(int)
            return list(a[columns].itertuples(index=False,name=None))
        from collections import Counter
        cc,bc=Counter(keys(c)),Counter(keys(bg));levels=len(set(cc)|set(bc))
        return np.array([np.log((cc[x]+1)/(len(c)+levels))-np.log((bc[x]+1)/(len(bg)+levels)) for x in keys(q)])
    groups=[['prefecture'],['prefecture','sex'],['prefecture','sex','smoking'],['prefecture','sex','agebin'],['sex','agebin','onset','death']]
    ratios=[group_ratio(candidates,cols) for cols in groups]
    for j,r in enumerate(ratios):mia[f'category{j}']=rank(r)
    mia['category_mix']=rank(sum(ratios[:4]))
    mia['category_geometry']=rank(mia['category_mix']+mia['attributes5'])
    for block in ['soft','hard']:
        dc=distance(challenge,c,bg,QI,block)
        db=distance(challenge,bg,bg,QI,block)
        aia[f'nearest_{block}']=c.time.to_numpy()[dc.argmin(1)]
        for k in [5,20,50]:
            pc=local(dc,c.time.to_numpy(float),k)
            pb=local(db,bg.time.to_numpy(float),k)
            aia[f'mode{k}_{block}']=GRID[pc.argmax(1)]
            contrast=(pc-pb)/np.maximum(1-pb,.05)
            best=contrast.argmax(1)
            aia[f'contrast{k}_{block}']=np.where(contrast[np.arange(len(best)),best]>.05,GRID[best],11.)
    # Refit conditional window models from the allowed tables on every call.
    qcols=QI+['sex']
    def qencode(f):
        return np.column_stack([numeric(f,qcols),f.prefecture.to_numpy()[:,None]==np.asarray(levels)[None,:]])
    # levels above is the fixed public prefecture vocabulary.
    xc,xb,xq=qencode(c),qencode(bg),qencode(challenge)
    tc=np.abs(c.time.to_numpy()[:,None]-GRID[None,:])<=.500000000001
    tb=np.abs(bg.time.to_numpy()[:,None]-GRID[None,:])<=.500000000001
    for leaf in [5,20]:
        ec=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=311,n_jobs=1).fit(xc,tc.astype(float))
        eb=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,max_features=.8,random_state=311,n_jobs=1).fit(xb,tb.astype(float))
        pc,pb=ec.predict(xq),eb.predict(xq)
        aia[f'forest_mode{leaf}']=GRID[pc.argmax(1)]
        for strength in [0.5,1.]:
            delta=(pc-strength*pb)/np.maximum(1-pb,.05)
            best=delta.argmax(1)
            aia[f'forest_contrast{leaf}_{strength}']=np.where(delta[np.arange(len(best)),best]>.05,GRID[best],11.)
    return {'mia':mia,'aia':aia}

def predict_selected(selection,**public):
    a=candidates_all(**public)
    return {k:a[k][selection[k]] for k in ['mia','aia']}
