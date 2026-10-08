"""New pl2: rank-blended MIA and weighted time-window consensus AIA."""
from itertools import combinations
import numpy as np
from scipy.stats import rankdata
TV=['Tv101','Tv102','Tv103','Tv104']
MIA=[(a,b,w) for a,b in combinations(TV,2) for w in (.25,.5,.75)]

def rank(x):return (rankdata(x,method='average')-.5)/len(x)

def mia_blend(answers,spec):
    a,b,w=spec
    return rank(w*rank(answers[a]['mia'])+(1-w)*rank(answers[b]['mia']))

def aia_consensus(answers,anchor):
    values=np.column_stack([answers[n]['aia'] for n in TV])
    weights=np.asarray([2. if n==anchor else 1. for n in TV])
    mass=((np.abs(values[:,:,None]-values[:,None,:])<=.5)*weights[None,None,:]).sum(axis=2)
    # Prefer anchor among equally supported existing predictions; never average
    # far-apart times into a third, unsupported event time.
    mass[:,TV.index(anchor)]+=1e-9
    best=mass.argmax(axis=1)
    return values[np.arange(len(values)),best]

def predict(answers,selection):
    return {'mia':mia_blend(answers,selection['mia_blend']),
            'aia':aia_consensus(answers,selection['aia_anchor'])}
