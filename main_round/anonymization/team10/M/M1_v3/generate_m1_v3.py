"""M1-v3 local DOMIAS defense.

Starting from M1-v2, identify the rare B_i member that dominated the seed-20260924
pseudo-B_j DOMIAS stress test (record_id 478). Move only the five closest C rows
in standardized 7-D continuous space toward their local A_bg neighborhoods for
BMI/TG/ALT. All other columns and rows are unchanged.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

DATA = Path('participant_data')
BASE = Path('M1_v2/C.csv')
OUT = Path('C.csv')
COLS = ['age','BMI','SBP','TG','HDL','ALT','FPG']
TARGET = ['BMI','TG','ALT']
LEAK_RECORD_ID = 478
N_ROWS = 5
LAMBDA = 0.75
K_BG = 50

A = pd.read_csv(DATA/'A_bg.csv')
B = pd.read_csv(DATA/'B_self.csv')
C0 = pd.read_csv(BASE)
mean = A[COLS].mean().to_numpy(float)
scale = A[COLS].std(ddof=0).to_numpy(float)
scale = np.where(scale > 0, scale, 1.0)
ZA = (A[COLS].to_numpy(float)-mean)/scale
ZC = (C0[COLS].to_numpy(float)-mean)/scale
b = B.loc[B['record_id'].eq(LEAK_RECORD_ID), COLS]
if len(b) != 1:
    raise ValueError('Expected exactly one matching B_self row')
zb = (b.to_numpy(float)[0]-mean)/scale
order = np.argsort(((ZC-zb)**2).sum(axis=1))
idx = order[:N_ROWS]
DA = cdist(ZC[idx], ZA, metric='sqeuclidean')
local = np.empty((N_ROWS, len(COLS)))
for r in range(N_ROWS):
    ids = np.argpartition(DA[r], K_BG-1)[:K_BG]
    d = DA[r, ids]
    w = np.exp(-d/2.0); w /= w.sum()
    local[r] = (ZA[ids] * w[:,None]).sum(axis=0)
ci = [COLS.index(c) for c in TARGET]
Z = ZC.copy()
Z[np.ix_(idx,ci)] = (1-LAMBDA)*Z[np.ix_(idx,ci)] + LAMBDA*local[:,ci]
V = Z*scale + mean
C = C0.copy()
for j,c in enumerate(COLS):
    C[c] = np.clip(V[:,j], A[c].min(), A[c].max())
C.to_csv(OUT,index=False)
print({'changed_rows':idx.tolist(),'columns':TARGET,'lambda':LAMBDA})
