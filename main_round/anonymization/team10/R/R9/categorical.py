"""Count-calibrated categorical synthesis using aggregate group centroids."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from cox_calibration import features

def calibrate_prefecture(b,c):
    categories=sorted(set(b.prefecture)|set(c.prefecture))
    target=b.prefecture.value_counts().reindex(categories,fill_value=0)
    count=c.prefecture.value_counts().reindex(categories,fill_value=0)
    surplus=[g for g in categories if count[g]>target[g]]
    deficit=[g for g in categories if count[g]<target[g]]
    rows=np.flatnonzero(c.prefecture.isin(surplus))
    slots=[g for g in surplus for _ in range(int(target[g]))]+[g for g in deficit for _ in range(int(target[g]-count[g]))]
    if len(rows)!=len(slots): raise RuntimeError('Unbalanced category allocation')
    out=c.copy(deep=True)
    if len(rows):
        xb=features(b).assign(onset=b.onset,death=b.death)
        xc=features(c).assign(onset=c.onset,death=c.death)
        sd=xb.std(ddof=1).to_numpy(); sd=np.maximum(sd,1e-6)
        global_mean=xb.mean().to_numpy()
        centers={}
        for g in categories:
            mask=b.prefecture==g; size=int(mask.sum())
            centers[g]=(xb.loc[mask].sum().to_numpy()+10*global_mean)/(size+10)/sd
        z=xc.to_numpy()[rows]/sd
        costs=np.full((len(rows),len(slots)),1e6)
        for k,g in enumerate(slots):
            if g in surplus:
                costs[:,k]=np.where(c.prefecture.to_numpy()[rows]==g,0.,1e6)
            else:
                costs[:,k]=np.sum((z-centers[g])**2,axis=1)
        rr,ss=linear_sum_assignment(costs)
        out.loc[out.index[rows[rr]],'prefecture']=np.asarray(slots)[ss]
    if not out.prefecture.value_counts().reindex(categories,fill_value=0).equals(target):
        raise RuntimeError('Category counts differ')
    return out,{'method':'minimum-cost count allocation using B group centroids shrunk by 10 pseudo-records',
                'changed_rows':int((out.prefecture!=c.prefecture).sum()),'counts_match_B':True,'source_row_matching':False}

def calibrate_age(b,c,cfg,detector):
    """Conditional age quantiles; adjust six labs along a ray to retain radius."""
    from scipy.stats import rankdata
    from model import radii,CONT
    before=radii(c,detector); rare=before>=detector['band']['lo']
    b_rare=radii(b,detector)>=detector['band']['lo']
    age=c.age.to_numpy().copy()
    for value in (False,True):
        mask=rare==value
        levels=(rankdata(c.loc[mask,'age'],method='ordinal')-.5)/mask.sum()
        age[mask]=np.rint(np.quantile(b.loc[b_rare==value,'age'],levels)).astype(int)
    mu=np.asarray(detector['mu']); inv=np.linalg.pinv(detector['sigma'])
    raw=c[CONT].to_numpy(float); d=raw[:,1:]-mu[1:]; da=age-mu[0]
    a=np.einsum('ij,jk,ik->i',d,inv[1:,1:],d)
    bb=da*(d@inv[1:,0]); cc=da**2*inv[0,0]-before**2
    disc=bb**2-a*cc
    factor=(-bb+np.sqrt(np.maximum(0,disc)))/np.maximum(a,1e-12)
    lab=mu[1:]+factor[:,None]*d
    bounds=np.array([cfg['schema']['ranges'][n] for n in CONT[1:]])
    okay=(disc>=0)&(factor>0)&np.all((lab>=bounds[:,0])&(lab<=bounds[:,1]),axis=1)
    out=c.copy(deep=True)
    out.loc[okay,'age']=age[okay]
    for j,name in enumerate(CONT[1:]): out.loc[okay,name]=lab[okay,j]
    if not np.array_equal(radii(out,detector)>=detector['band']['lo'],rare): raise RuntimeError('Age calibration changed tail membership')
    return out,{'method':'conditional empirical age quantiles with fixed Mahalanobis radius',
                'changed_ages':int((out.age!=c.age).sum()),'range_or_geometry_fallbacks':int((~okay).sum()),'tail_membership_preserved':True}
