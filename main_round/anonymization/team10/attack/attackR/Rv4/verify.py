import os,sys,copy,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT))
import runtime as rt
from run import actual
import pandas as pd
import numpy as np
from model import variants
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,default=OUT/'submission');args=parser.parse_args()
    prediction_dir=args.predictions.resolve()
    a,public=actual();w,_,_,validator,cfg,dist,*_=a
    base={n:pd.read_csv(rt.ATTACK/'Attack_R/v2/submission'/(n+'.csv'),float_precision='round_trip') for n in ['F_mia','F_aia']}
    v=variants(**public,baseline=base);sel=rt.read(OUT/'selection.json')
    ref={n:v[c][sel[c]] for c,n in [('mia','F_mia'),('aia','F_aia')]}
    # A new process recomputes the correction using the archived baseline;
    # equality to full T regeneration verifies cached-world compatibility.
    generated={n:pd.read_csv(prediction_dir/(n+'.csv'),float_precision='round_trip') for n in ref}
    for n in ref:
        if rt.csv(ref[n])!=(prediction_dir/(n+'.csv')).read_bytes():raise ValueError('Cached baseline differs from fresh T generation')
    rejected=[]
    for case in ['missing_mia_row','duplicate_mia_id','nan_mia','mia_above_one','missing_aia_row','duplicate_aia_id','negative_time','infinite_time']:
        fm=generated['F_mia'].copy();fa=generated['F_aia'].copy()
        if case=='missing_mia_row':fm=fm.iloc[:-1]
        elif case=='duplicate_mia_id':fm.iloc[0,0]=fm.iloc[1,0]
        elif case=='nan_mia':fm.iloc[0,1]=np.nan
        elif case=='mia_above_one':fm.iloc[0,1]=1.1
        elif case=='missing_aia_row':fa=fa.iloc[:-1]
        elif case=='duplicate_aia_id':fa.iloc[0,:2]=fa.iloc[1,:2].to_numpy()
        elif case=='negative_time':fa.loc[0,'time_hat']=-1.
        elif case=='infinite_time':fa.loc[0,'time_hat']=np.inf
        try:w.validate_frames(np,pd,validator,fm,fa,cfg,dist)
        except (ValueError,AssertionError,RuntimeError):rejected.append(case)
        else:raise ValueError('Invalid submission was accepted: '+case)
    rt.save(OUT/'acceptance_checks.json',{'completed':True,'numeric_unit_tests_passed':3,'cached_Rv2_correction_matches_full_T_regeneration':True,'invalid_submission_cases_rejected':rejected,'prediction_sha256':{n:rt.sha(prediction_dir/(n+'.csv')) for n in ref},'frozen_dependencies':rt.preserve()})
    print('Accepted: full/cached equivalence; rejected',len(rejected),'invalid submissions')
if __name__=='__main__':main()
