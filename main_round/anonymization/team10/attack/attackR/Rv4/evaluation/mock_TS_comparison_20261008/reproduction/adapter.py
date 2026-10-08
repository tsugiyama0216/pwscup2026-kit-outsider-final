"""Apply the frozen Rv4 correction to a verified same-input Rv2 prediction."""
import numpy as np
def corrected(model, *, c, background, candidates, challenge, old_mia, old_aia, mask=None):
    mask=np.zeros(len(candidates),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    om=np.asarray(old_mia,float).copy();om[~mask]=model.ranks(om[~mask]);om[mask]=0
    _,t,ratio,support,full=model.evidence(c,background,candidates,challenge)
    raw=om+.35*np.clip((full-3.)/5.,0,1)
    mia=np.zeros(len(raw));mia[~mask]=model.ranks(raw[~mask])
    aia=np.where((ratio<.10)&(support>=.6),t,old_aia)
    return {'mia':mia,'aia':aia}
