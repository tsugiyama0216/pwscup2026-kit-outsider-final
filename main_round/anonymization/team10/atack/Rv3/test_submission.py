"""Ensure malformed real-ID submissions cannot pass the pinned validator."""
import unittest
import numpy as np
import pandas as pd
import run

class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator,cls.cfg,_,cls.dist=run.runtime()
        cls.fm=pd.read_csv(run.HERE/'submission/F_mia.csv',float_precision='round_trip')
        cls.fa=pd.read_csv(run.HERE/'submission/F_aia.csv',float_precision='round_trip')
    def validate(self,m,a):return run.w.validate_frames(np,pd,self.validator,m,a,self.cfg,self.dist)
    def test_valid(self):self.validate(self.fm,self.fa)
    def test_bad_variants(self):
        cases=[]
        for value in [np.nan,np.inf,-.1,1.1]:
            m=self.fm.copy();m.iloc[0,1]=value;cases.append((m,self.fa))
        for value in [np.nan,np.inf,-1]:
            a=self.fa.copy();a.loc[0,'time_hat']=value;cases.append((self.fm,a))
        for key,value in [('record_id',.5),('record_id',999999999)]:
            m=self.fm.copy();m[key]=m[key].astype(float);m.loc[0,key]=value;cases.append((m,self.fa))
        m=self.fm.copy();m.loc[0,'record_id']=m.loc[1,'record_id'];cases.append((m,self.fa))
        cases.extend([(self.fm.iloc[:-1],self.fa),(pd.concat([self.fm,self.fm.iloc[:1]]),self.fa),(self.fm.drop(columns=self.fm.columns[1]),self.fa)])
        for key,value in [('target_k',10),('target_k',.5),('challenge_row_id',.5),('challenge_row_id',999999999)]:
            a=self.fa.copy();a[key]=a[key].astype(float);a.loc[0,key]=value;cases.append((self.fm,a))
        a=self.fa.copy();a.iloc[0]=a.iloc[1];cases.append((self.fm,a))
        cases.extend([(self.fm,self.fa.iloc[:-1]),(self.fm,pd.concat([self.fa,self.fa.iloc[:1]]))])
        self.assertEqual(len(cases),20)
        for i,(m,a) in enumerate(cases):
            with self.subTest(variant=i):
                with self.assertRaises((ValueError,AssertionError,RuntimeError)):self.validate(m,a)
if __name__=='__main__':unittest.main()
