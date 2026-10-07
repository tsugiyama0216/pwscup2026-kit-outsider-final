from pathlib import Path
import json, sys, math
import numpy as np, pandas as pd
from statsmodels.duration.hazard_regression import PHReg

ROOT=Path('/mnt/data/pwsrepo/pwscup2026-kit-private-main/main_round/anonymization/team10')
M1=pd.read_csv(ROOT/'M/M1/C.csv')
Bself=pd.read_csv(ROOT/'participant_data/B_self.csv'); B=Bself.drop(columns=['is_rare'])
REF=json.load(open(ROOT/'participant_data/utility_ref.json'))
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']
FEATS=['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking']

def design(df):
 x=df[['age','sex','BMI','SBP','TG','HDL','ALT','FPG','smoking']].astype(float).copy()
 x['TG']=np.log(np.clip(x['TG'],1e-9,None)); x['ALT']=np.log(np.clip(x['ALT'],1e-9,None))
 x.columns=FEATS
 return x

def fit(df):
 x=design(df)
 r=PHReg(df.time.to_numpy(float),x.to_numpy(float),status=df.onset.to_numpy(int),ties='efron').fit(disp=False)
 return r.params.copy(), np.asarray(r.cov_params()), np.sqrt(np.diag(r.cov_params()))
BETA_B, VAR_B, SE_B = fit(B)

def cox_io(df):
 try: beta,var,se=fit(df)
 except Exception: return -1,None
 loB=BETA_B-1.96*SE_B; hiB=BETA_B+1.96*SE_B
 loC=beta-1.96*se; hiC=beta+1.96*se
 vals=[]; detail={}
 for k,name in enumerate(FEATS):
  sigB=(loB[k]>0) or (hiB[k]<0); sigC=(loC[k]>0) or (hiC[k]<0)
  overlap=max(0,min(hiB[k],hiC[k])-max(loB[k],loC[k]))
  io=.5*(overlap/(hiB[k]-loB[k])+overlap/(hiC[k]-loC[k])) if overlap>0 else 0
  contrib=0 if sigB!=sigC or (sigB and sigC and np.sign(beta[k])!=np.sign(BETA_B[k])) else io
  vals.append(contrib); detail[name]=(float(beta[k]),bool(sigB),bool(sigC),float(contrib))
 return float(np.mean(vals)),detail

def concordance(time,event,risk):
 # Harrell concordance: comparable if earlier time has event; ties risk half
 t=np.asarray(time,float); e=np.asarray(event,int); r=np.asarray(risk,float)
 conc=0.; comp=0.
 n=len(t)
 for i in range(n):
  if e[i]!=1: continue
  mask=t>t[i]
  rr=r[mask]
  comp += len(rr)
  conc += np.sum(r[i]>rr) + .5*np.sum(r[i]==rr)
 return conc/comp if comp else .5
TRTR=concordance(B.time,B.onset,design(B).to_numpy()@BETA_B)
def uspec(df):
 io,detail=cox_io(df)
 if io<0:return 0,{}
 beta,_,_=fit(df)
 ci=concordance(B.time,B.onset,design(B).to_numpy()@beta)
 cal=1.0 if TRTR-.5<.02 else float(np.clip((ci-.5)/(TRTR-.5),0,1))
 return min(io,cal),{'io':io,'tstr_cal':cal,'cindex':ci,'coef':detail}

def rare_mask(df):
 vals=df[CONT].to_numpy(float); d=vals-np.asarray(REF['mu']); inv=np.linalg.pinv(np.asarray(REF['sigma']))
 return np.sqrt(np.clip(np.einsum('ij,jk,ik->i',d,inv,d),0,None))>=float(REF['band']['lo'])
RARE=rare_mask(M1)
TARGET_RARE=int(round(Bself.loc[Bself.is_rare.astype(bool),'onset'].mean()*RARE.sum()))

def cox_moments(x,w,time,event):
 wx=w[:,None]*x; wxx=w[:,None,None]*x[:,:,None]*x[:,None,:]
 score=np.zeros(x.shape[1]); information=np.zeros((x.shape[1],x.shape[1]))
 for t in np.unique(time[event==1]):
  risk=time>=t; died=(time==t)&(event==1); n=int(died.sum())
  r0,r1,r2=w[risk].sum(),wx[risk].sum(0),wxx[risk].sum(0)
  e0,e1,e2=w[died].sum(),wx[died].sum(0),wxx[died].sum(0)
  score+=x[died].sum(0)
  for k in range(n):
   frac=k/n; den=r0-frac*e0; mean=(r1-frac*e1)/den
   score-=mean; information+=(r2-frac*e2)/den-np.outer(mean,mean)
 return score,information

def outcome_calibrate(c,max_steps=100):
 out=c.copy(deep=True); xraw=design(out).to_numpy(float); xb=design(B).to_numpy(float)
 scale=xb.std(0,ddof=1); x=(xraw-xraw.mean(0))/scale
 tbeta=BETA_B*scale; tvar=VAR_B*scale[:,None]*scale[None,:]
 w=np.exp(np.clip(x@tbeta,-30,30))
 time=out.time.to_numpy(float).copy(); event=out.onset.to_numpy(int).copy(); death=out.death.to_numpy(int).copy()
 # Step 1: get rare event count to target by swapping whole outcome triplets rare<->nonrare, choose closest risk
 risk=x@tbeta
 while int(event[RARE].sum())<TARGET_RARE:
  a=np.flatnonzero(RARE & (event==0) & (death==0))
  b=np.flatnonzero((~RARE) & (event==1))
  if not len(a) or not len(b): break
  # choose pair with smallest risk difference
  # block matrix fine 66 x ~100
  dd=np.abs(risk[a,None]-risk[b][None,:]); ia,ib=np.unravel_index(np.argmin(dd),dd.shape); i,j=a[ia],b[ib]
  for arr in (time,event): arr[i],arr[j]=arr[j],arr[i]
 # fixed beta score loss
 def evaluate(tt,ee):
  s,info=cox_moments(x,w,tt,ee); return float(s@tvar@s),s,info
 obj,score,info=evaluate(time,event); initial=obj; swaps=0
 for step in range(max_steps):
  positives=np.flatnonzero(event==1); negatives=np.flatnonzero((event==0)&(death==0))
  if not len(positives) or not len(negatives): break
  xp,xn=x[positives],x[negatives]; grad=tvar@score
  approx=(2*(xn@grad)[None,:]-2*(xp@grad)[:,None]
          +np.sum((xn@tvar)*xn,1)[None,:]+np.sum((xp@tvar)*xp,1)[:,None]
          -2*(xp@tvar)@xn.T)
  # preserve rare onset count: swaps only within same rare status
  allowed=RARE[positives,None]==RARE[negatives][None,:]; approx[~allowed]=np.inf
  finite=np.isfinite(approx); count=min(15,int(finite.sum()))
  if count==0:break
  inds=np.argpartition(approx.ravel(),count-1)[:count]
  inds=sorted(inds,key=lambda k:approx.ravel()[k])
  best=None
  for k in inds:
   aa,bb=np.unravel_index(k,approx.shape); i,j=positives[aa],negatives[bb]
   tt=time.copy(); ee=event.copy(); dd=death.copy()
   for arr in (tt,ee): arr[i],arr[j]=arr[j],arr[i]
   val,ns,ni=evaluate(tt,ee)
   if val<obj-1e-9 and (best is None or val<best[0]):best=(val,ns,ni,tt,ee,dd)
  if best is None:break
  obj,score,info,time,event,death=best; swaps+=1
 out['time']=time;out['onset']=event;out['death']=death
 return out,{'rare_target':TARGET_RARE,'rare_onset':int(event[RARE].sum()),'initial_loss':initial,'final_loss':obj,'swaps':swaps}

def smoking_perturb(base,fraction,seed):
 out=base.copy(deep=True); rng=np.random.default_rng(seed)
 one=np.flatnonzero(out.smoking.to_numpy()==1); zero=np.flatnonzero(out.smoking.to_numpy()==0)
 n=int(round(fraction*min(len(one),len(zero))))
 if n:
  a=rng.choice(one,n,replace=False); b=rng.choice(zero,n,replace=False)
  s=out.smoking.to_numpy().copy();s[a]=0;s[b]=1;out['smoking']=s
 return out

rows=[]; outdir=Path('/mnt/data/M1_v2_finalsearch');outdir.mkdir(exist_ok=True)
for frac in [0,.03,.05,.08,.10,.15]:
 for seed in range(3):
  c=smoking_perturb(M1,frac,20260924+seed)
  c,audit=outcome_calibrate(c,80)
  u,d=uspec(c)
  # extra priority: keep smoking nonsignificant and high io
  sm=d.get('coef',{}).get('smoking',(0,False,True,0))
  rows.append({'frac':frac,'seed':seed,'Uspec':u,'io':d.get('io'), 'tstr':d.get('tstr_cal'), 'sm_beta':sm[0], 'sm_sigC':sm[2], 'rare_onset':audit['rare_onset'],'swaps':audit['swaps'],'loss':audit['final_loss']})
  c.to_csv(outdir/f'C_f{frac:.2f}_s{seed}.csv',index=False)
res=pd.DataFrame(rows).sort_values(['Uspec','io'],ascending=False);res.to_csv(outdir/'search.csv',index=False)
print(res.head(20).to_string(index=False))
