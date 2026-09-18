import numpy as np
import pandas as pd
from evaluation import COLUMNS

def artificial(seed):
    rng = np.random.default_rng(seed)
    n = 480
    z = rng.normal(size=(n,7))
    c = pd.DataFrame({'age':np.clip(np.rint(48+8*z[:,0]),20,80).astype(int),
                      'sex':rng.integers(0,2,n), 'prefecture':rng.integers(1,48,n),
                      'BMI':24+2*z[:,1], 'SBP':120+10*z[:,2], 'TG':np.exp(4.5+0.3*z[:,3]),
                      'HDL':52+6*z[:,4], 'ALT':np.exp(3+0.3*z[:,5]),
                      'smoking':rng.integers(0,2,n),'FPG':95+8*z[:,6]})
    risk = 0.15*z[:,0]+0.12*z[:,1]+0.1*z[:,2]+0.2*z[:,5]+0.7*z[:,6]+0.15*c.sex
    time = rng.exponential(12*np.exp(-risk))
    c['time'] = np.minimum(time,10)
    c['onset'] = (time<10).astype(int)
    c['death'] = 0
    return c[COLUMNS]
