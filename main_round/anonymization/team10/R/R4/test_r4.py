"""R4 transport invariants and release safeguards; artificial data only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
import joint_transport
from cox_calibration import features,fit,MUTABLE
from evaluation import COLUMNS,HERE,write_package
import generate

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


class JointTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(710),artificial(711)
        cls.cfg={'schema':{'ranges':{name:[0.00001,1e6] for name in ['BMI','SBP','TG','HDL','ALT','FPG']}}}
        cls.full,cls.full_audit=joint_transport.transport(cls.b,cls.c,cls.cfg,1.0)
        cls.projected,cls.audit=joint_transport.transport(cls.b,cls.c,cls.cfg,0.5,0,True)

    def conditional_statistics(self,frame):
        x=features(frame)
        design=np.column_stack([np.ones(len(x)),x[['age','sex','smoking']].to_numpy()])
        y=x[MUTABLE].to_numpy()
        coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
        return coefficients,np.cov(y-design@coefficients,rowvar=False)

    def test_full_unconstrained_transport_matches_conditional_means(self):
        source,_=self.conditional_statistics(self.b)
        generated,_=self.conditional_statistics(self.full)
        np.testing.assert_allclose(generated,source,atol=1e-9)

    def test_full_unconstrained_transport_matches_residual_covariance(self):
        _,source=self.conditional_statistics(self.b)
        _,generated=self.conditional_statistics(self.full)
        np.testing.assert_allclose(generated,source,atol=1e-9)

    def test_projection_preserves_each_rows_linear_predictor(self):
        beta,_=fit(self.c,'onset')
        before=features(self.c).to_numpy()@beta
        after=features(self.projected).to_numpy()@beta
        np.testing.assert_allclose(after,before,atol=1e-10,rtol=1e-10)
        self.assertEqual(sum(self.audit['clipped_counts'].values()),0)

    def test_projection_preserves_refitted_cox_coefficients(self):
        before,_=fit(self.c,'onset')
        after,_=fit(self.projected,'onset')
        np.testing.assert_allclose(after,before,rtol=2e-4,atol=2e-5)

    def test_discrete_and_outcome_columns_unchanged(self):
        cols=['age','sex','smoking','prefecture','time','onset','death']
        pd.testing.assert_frame_equal(self.c[cols],self.projected[cols])

    def test_inputs_not_mutated(self):
        b,c=copy.deepcopy(self.b),copy.deepcopy(self.c)
        joint_transport.transport(b,c,self.cfg,0.5,0,True)
        pd.testing.assert_frame_equal(b,self.b)
        pd.testing.assert_frame_equal(c,self.c)

    def test_zero_is_exact_identity(self):
        out,_=joint_transport.transport(self.b,self.c,self.cfg,0,0,True)
        pd.testing.assert_frame_equal(out,self.c,check_exact=True)

    def test_deterministic(self):
        out,_=joint_transport.transport(self.b,self.c,self.cfg,0.5,0,True)
        pd.testing.assert_frame_equal(out,self.projected,check_exact=True)

    def test_private_labels_and_ids_ignored(self):
        b=self.b.assign(is_rare=True,record_id=np.arange(len(self.b)))
        out,_=joint_transport.transport(b,self.c,self.cfg,0.5,0,True)
        pd.testing.assert_frame_equal(out,self.projected,check_exact=True)

    def test_invalid_strengths_rejected(self):
        for v in [-0.1,1.1,np.nan,np.inf,True]:
            for name in ('strength','cox_strength'):
                with self.subTest(name=name,value=v),self.assertRaises(ValueError):
                    joint_transport.transport(self.b,self.c,self.cfg,**{name:v})

    def test_bad_row_counts_rejected(self):
        for b,c in [(self.b.iloc[:-1],self.c),(self.b.iloc[:10],self.c.iloc[:10])]:
            with self.assertRaises(ValueError):
                joint_transport.transport(b,c,self.cfg)

    def test_degenerate_scale_rejected(self):
        with self.assertRaisesRegex(ValueError,'scale'):
            joint_transport.transport(self.b.assign(BMI=20),self.c,self.cfg)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            joint_transport.transport(self.b.assign(TG=np.nan),self.c,self.cfg)

    def test_risk_projection_and_recalibration_are_exclusive(self):
        with self.assertRaisesRegex(ValueError,'cannot'):
            joint_transport.transport(self.b,self.c,self.cfg,0.5,0.25,True)

    def test_clipping_is_audited(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['schema']['ranges']['FPG']=[90,100]
        out,audit=joint_transport.transport(self.b,self.c,cfg,0.5,0,True)
        self.assertGreater(audit['clipped_counts']['FPG'],0)
        self.assertTrue(out.FPG.between(90,100).all())


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.base={'U_gen':0.9188,'U_spec':0.9117,'U_rare':0.8661,'U_valid':1.0}
        self.score={'utility':0.88,'facets':{'U_gen':0.93,'U_spec':0.914,'U_rare':0.88,'U_valid':1.0}}
        self.audit={'clipped_counts':{'FPG':0},'risk_projection':{'maximum_log_risk_change_before_clipping':1e-15}}

    def test_release_gate_accepts_improvement(self):
        generate.check_improvement(self.score,self.base,self.audit)

    def test_release_gate_rejects_each_facet_regression(self):
        for name,value in self.base.items():
            score=copy.deepcopy(self.score)
            score['facets'][name]=value-0.001
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                generate.check_improvement(score,self.base,self.audit)

    def test_release_gate_rejects_clipping_risk_drift_or_no_improvement(self):
        for audit in [dict(self.audit,clipped_counts={'FPG':1}),dict(self.audit,risk_projection={'maximum_log_risk_change_before_clipping':1e-4})]:
            with self.assertRaises(RuntimeError):
                generate.check_improvement(self.score,self.base,audit)
        with self.assertRaises(RuntimeError):
            generate.check_improvement(dict(self.score,utility=min(self.base.values())),self.base,self.audit)

    def test_frozen_base_hash_guard(self):
        with patch.object(generate,'digest',return_value='changed'),self.assertRaises(RuntimeError):
            generate.load_base()

    def test_output_directory_guard(self):
        with patch('sys.argv',['generate.py','--output',str(HERE.parent)]),self.assertRaises(ValueError):
            generate.main()

    def test_package_preserves_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=HERE/'local-output') as directory:
            path=Path(directory)/'submission.zip'
            payload,token=b'a,b\n1,2\n',b'FAKE-TEST-TOKEN\n'
            write_package(path,payload,token)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(),['C.csv','token.txt'])
                self.assertEqual(archive.read('C.csv'),payload)
                self.assertEqual(archive.read('token.txt'),token)
            with self.assertRaises(FileExistsError):
                write_package(path,payload,token)


if __name__=='__main__':
    unittest.main()
