"""Fresh multivariate mixture/KDE likelihood ratios; public training tables only."""
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from model import NUM,numeric,canonical,rank

def candidates_all(c,background,candidates,wanted=None):
    c=canonical(c);bg=canonical(background)
    cols=NUM[:-1]+['sex','smoking','onset','death']
    scale=np.maximum(numeric(bg,cols).std(0),.1)
    x,y,q=[numeric(f,cols)/scale for f in [c,bg,candidates]]
    options={}
    for k in [2,4,8]:
        if wanted and wanted not in [f'gmm{k}','gmm_mix'] and not (wanted=='density_mix' and k==4):continue
        a=GaussianMixture(n_components=k,covariance_type='full',reg_covar=.03,random_state=311,n_init=1,max_iter=150).fit(x)
        b=GaussianMixture(n_components=k,covariance_type='full',reg_covar=.03,random_state=311,n_init=1,max_iter=150).fit(y)
        options[f'gmm{k}']=rank(a.score_samples(q)-b.score_samples(q))
    if wanted and wanted.startswith('gmm'):
        if wanted=='gmm_mix':options['gmm_mix']=rank(options['gmm2']+options['gmm4']+options['gmm8'])
        return options
    cols=NUM[:-1];x,y,q=[numeric(f,cols) for f in [c,bg,candidates]]
    cov=np.cov(y,rowvar=False);scale=np.sqrt(np.diag(cov))
    corr=cov/scale[:,None]/scale[None,:]
    transform=np.linalg.cholesky(np.linalg.inv(corr+.05*np.eye(len(cols))))
    x,y,q=[v/scale@transform for v in [x,y,q]]
    dc,db=cdist(q,x,'sqeuclidean'),cdist(q,y,'sqeuclidean')
    for width in [.25,.5,1.]:
        if wanted and wanted!=f'kde{width}' and not (wanted=='density_mix' and width==.5):continue
        scores=logsumexp(-dc/(2*width**2),axis=1)-np.log(len(c))-logsumexp(-db/(2*width**2),axis=1)+np.log(len(bg))
        options[f'kde{width}']=rank(scores)
    if wanted is None:options['gmm_mix']=rank(options['gmm2']+options['gmm4']+options['gmm8'])
    if wanted in [None,'density_mix']:options['density_mix']=rank(options['gmm4']+options['kde0.5'])
    return options
