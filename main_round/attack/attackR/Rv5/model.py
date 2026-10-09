"""Conservative public-data residual attack on frozen Rv2; no mock imports."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

FULL=['age','BMI','SBP','TG','HDL','ALT','FPG']
QI=['age','SBP','TG','HDL','ALT']
ALL=FULL+['time','sex','prefecture','smoking','onset','death']

def numeric(frame,cols):
    x=frame[cols].to_numpy(float).copy()
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    for j,c in enumerate(cols):
        if c in ('TG','ALT'):
            if (x[:,j]<0).any():raise ValueError('Negative laboratory value')
            x[:,j]=np.log1p(x[:,j])
    return x

def canonical(frame):
    return frame.sort_values([c for c in ALL if c in frame],kind='stable').reset_index(drop=True)

def distances(q,r,bg,cols):
    scale=np.maximum(np.std(numeric(bg,cols),axis=0),1e-6)
    d=cdist(numeric(q,cols)/scale,numeric(r,cols)/scale,'sqeuclidean')/len(cols)
    for col,w in [('sex',1.0),('prefecture',.25)]:
        d+=w*(q[col].to_numpy()[:,None]!=r[col].to_numpy()[None,:])
    return d

def nearest(q,r,bg,cols,self_reference=False):
    parts=[]
    for start in range(0,len(q),256):
        d=distances(q.iloc[start:start+256],r,bg,cols)
        if self_reference:
            ix=np.arange(len(d));d[ix,start+ix]=np.inf
        parts.append(d.min(axis=1))
    return np.concatenate(parts)

def prepare(candidates,background):
    bg=canonical(background)
    views=[FULL,FULL+['time']]
    return {'bg':bg,'views':views,
            'candidate_radius':[nearest(candidates,bg,bg,v) for v in views],
            'background_radius':[nearest(bg,bg,bg,v,True) for v in views]}

def membership(c,candidates,cache,old,private,settings):
    bg=cache['bg'];c=canonical(c)
    scores=[];null=[]
    for j,v in enumerate(cache['views']):
        dc=nearest(candidates,c,bg,v);dn=nearest(bg,c,bg,v)
        scores.append(np.log((cache['candidate_radius'][j]+1e-10)/(dc+1e-10)))
        null.append(np.log((cache['background_radius'][j]+1e-10)/(dn+1e-10)))
    # Both views must support the match, limiting an isolated corrupted feature.
    score=np.minimum(*scores);reference=np.minimum(*null)
    p=[];strata=[]
    for i,row in enumerate(candidates.itertuples()):
        use=(bg.sex.to_numpy()==row.sex)&(bg.prefecture.to_numpy()==row.prefecture)
        level='sex_prefecture'
        if use.sum()<40:use=bg.sex.to_numpy()==row.sex;level='sex'
        if use.sum()<40:use=np.ones(len(bg),bool);level='all'
        p.append((1+np.sum(reference[use]>=score[i]))/(1+np.sum(use)));strata.append(level)
    p=np.asarray(p)
    evidence=np.clip(-np.log10(p)+np.log10(settings['mia_null_tail']),0,settings['mia_max_log10_evidence'])
    old=np.asarray(old,float);private=np.asarray(private,bool)
    if old.shape!=(len(candidates),) or private.shape!=old.shape or not np.isfinite(old).all() or ((old<0)|(old>1)).any():raise ValueError('Invalid baseline or mask')
    z=np.clip(old,1e-9,1-1e-9)
    shifted=1/(1+np.exp(-(np.log(z/(1-z))+settings['mia_logit_evidence_weight']*evidence)))
    result=np.where(evidence>0,shifted,old);result[private]=0
    return result,{'supported_rows':int(((evidence>0)&~private).sum()),'calibration_min_p':float(p.min()),
                   'calibration_groups':{x:strata.count(x) for x in set(strata)},'known_negative_rows':int(private.sum()),
                   'note':'Background tails are evidence heuristics, not certified FPR or membership probabilities'}

def window_mass(times,weights,options):
    return ((np.abs(times[:,None]-options[None,:])<=.5+1e-12)*weights[:,None]).sum(axis=0)

def attribute(c,background,challenge,old,settings):
    c=canonical(c);bg=canonical(background)
    dc=distances(challenge,c,bg,QI);db=distances(challenge,bg,bg,QI)
    ct=c.time.to_numpy(float);bt=bg.time.to_numpy(float)
    result=np.asarray(old,float).copy();rows=[]
    if result.shape!=(len(challenge),) or not np.isfinite(result).all() or (result<0).any():raise ValueError('Invalid AIA baseline')
    k=min(settings['aia_neighbors'],len(c),len(bg))
    for i in range(len(challenge)):
        ci=np.argsort(dc[i],kind='stable')[:k];bi=np.argsort(db[i],kind='stable')[:k]
        cw=np.exp(-(dc[i,ci]-dc[i,ci[0]])/max(dc[i,ci[-1]],1e-9));cw/=cw.sum()
        bw=np.exp(-(db[i,bi]-db[i,bi[0]])/max(db[i,bi[-1]],1e-9));bw/=bw.sum()
        times=ct[ci]
        # Midpoints of each maximal one-year support interval, plus existing times.
        st=np.sort(np.unique(times));opts=list(st)+[float(result[i])]
        for lo in st:
            hi=st[st<=lo+1+1e-12][-1];opts.append(float((lo+hi)/2))
        opts=np.asarray(sorted(set(opts)))
        pc=window_mass(times,cw,opts);pb=window_mass(bt[bi],bw,opts)
        contrast=(pc-pb)/np.maximum(1-pb,.05)
        best=int(np.argmax(contrast));oldpos=int(np.flatnonzero(opts==result[i])[0])
        ratio=float((dc[i,ci[0]]+1e-12)/(db[i,bi[0]]+1e-12))
        margin=float(dc[i,ci[0]]/max(dc[i,ci[1]],1e-12)) if k>1 else 1.
        unique=bool(ratio<settings['aia_unique_ratio'] and margin<settings['aia_unique_margin'] and dc[i,ci[1]]>1e-10)
        oldvalue=float(result[i]);reason='baseline'
        # An isolated close record supplies a time without averaging distant modes.
        if unique and abs(oldvalue-times[0])>.5 and pb[np.argmin(abs(opts-times[0]))]<.5:
            result[i]=times[0];reason='unique_link'
        elif ratio<settings['aia_distance_ratio'] and pc[best]>=settings['aia_concentration'] and contrast[best]-contrast[oldpos]>=settings['aia_min_excess_gain']:
            result[i]=opts[best];reason='window_contrast'
        rows.append({'reason':reason,'distance_ratio':ratio,'unique_margin':margin,'changed':bool(result[i]!=oldvalue)})
    return result,{'changed':sum(r['changed'] for r in rows),'unique_links':sum(r['reason']=='unique_link' for r in rows),
                   'window_changes':sum(r['reason']=='window_contrast' for r in rows),'rows':rows}
