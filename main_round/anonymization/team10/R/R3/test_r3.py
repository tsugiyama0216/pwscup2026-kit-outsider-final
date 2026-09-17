"""R3 mathematical invariants and release safeguards; artificial data only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

import calibration
import generate
from evaluation import COLUMNS, HERE, write_package


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


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c = artificial(710),artificial(711)
        cls.cfg={'schema':{'ranges':{name:[0.00001,1e6] for name in ['BMI','SBP','TG','HDL','ALT','FPG']}}}
        cls.result,cls.audit=calibration.calibrate(cls.b,cls.c,cls.cfg,1.0)

    def test_full_calibration_reproduces_target_cox_coefficients(self):
        self.assertEqual(sum(self.audit['clipped_counts'].values()),0)
        target,_=calibration.fit(self.b,'onset')
        transformed,_=calibration.fit(self.result,'onset')
        np.testing.assert_allclose(transformed,target,rtol=2e-4,atol=2e-5)

    def test_full_calibration_preserves_fitted_linear_predictor_differences(self):
        target,_=calibration.fit(self.b,'onset')
        original,_=calibration.fit(self.c,'onset')
        left=calibration.features(self.result).to_numpy()@target
        right=calibration.features(self.c).to_numpy()@original
        np.testing.assert_allclose(left-left.mean(),right-right.mean(),rtol=1e-10,atol=1e-10)

    def test_discrete_and_outcome_columns_unchanged(self):
        cols=['age','sex','smoking','prefecture','time','onset','death']
        pd.testing.assert_frame_equal(self.c[cols],self.result[cols])

    def test_inputs_not_mutated(self):
        b,c=copy.deepcopy(self.b),copy.deepcopy(self.c)
        calibration.calibrate(b,c,self.cfg,0.5)
        pd.testing.assert_frame_equal(b,self.b)
        pd.testing.assert_frame_equal(c,self.c)

    def test_zero_strength_is_exact_identity(self):
        out,_=calibration.calibrate(self.b,self.c,self.cfg,0)
        pd.testing.assert_frame_equal(out,self.c,check_exact=True)

    def test_deterministic(self):
        out,_=calibration.calibrate(self.b,self.c,self.cfg,1)
        pd.testing.assert_frame_equal(out,self.result,check_exact=True)

    def test_ids_and_rare_labels_do_not_affect_calibration(self):
        b=self.b.assign(record_id=np.arange(len(self.b)),is_rare=True)
        out,_=calibration.calibrate(b,self.c,self.cfg,1)
        pd.testing.assert_frame_equal(out,self.result,check_exact=True)

    def test_rejects_invalid_strength(self):
        for value in [-0.1,1.1,np.nan,np.inf,True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                calibration.calibrate(self.b,self.c,self.cfg,value)

    def test_rejects_invalid_cause(self):
        with self.assertRaises(ValueError):
            calibration.calibrate(self.b,self.c,self.cfg,causes=('unknown',))

    def test_rejects_row_count_mismatch(self):
        with self.assertRaises(ValueError):
            calibration.calibrate(self.b.iloc[:-1],self.c,self.cfg)

    def test_rejects_invalid_predictors(self):
        for name,value in [('TG',0),('ALT',-1),('FPG',np.nan),('BMI',np.inf)]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                calibration.features(self.b.assign(**{name:value}))

    def test_rejects_degenerate_target_directions(self):
        with patch.object(calibration,'fit',return_value=(np.zeros(9),np.ones(9))):
            with self.assertRaisesRegex(ValueError,'Degenerate'):
                calibration.calibrate(self.b,self.c,self.cfg)

    def test_clipping_is_recorded(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['schema']['ranges']['FPG']=[90,100]
        out,audit=calibration.calibrate(self.b,self.c,cfg,0.5)
        self.assertGreater(audit['clipped_counts']['FPG'],0)
        self.assertTrue(out.FPG.between(90,100).all())


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.base={'U_gen':0.9357,'U_spec':0.7694,'U_rare':0.8092,'U_valid':1.0}
        self.score={'utility':0.866,'facets':{'U_gen':0.9188,'U_spec':0.9117,'U_rare':0.866,'U_valid':1.0}}
        self.audit={'clipped_counts':{'FPG':0}}

    def test_release_gate_accepts_declared_tradeoff(self):
        generate.check_improvement(self.score,self.base,self.audit)

    def test_release_gate_rejects_regressions_or_clipping(self):
        for key,value in [('U_gen',0.89),('U_spec',0.70),('U_rare',0.80),('U_valid',0.999)]:
            score=copy.deepcopy(self.score)
            score['facets'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                generate.check_improvement(score,self.base,self.audit)
        with self.assertRaises(RuntimeError):
            generate.check_improvement(self.score,self.base,{'clipped_counts':{'FPG':1}})
        with self.assertRaises(RuntimeError):
            generate.check_improvement({**self.score,'utility':0.7},self.base,self.audit)

    def test_frozen_base_hash_guard(self):
        with patch.object(generate,'digest',return_value='changed'):
            with self.assertRaisesRegex(RuntimeError,'Frozen R2'):
                generate.load_base()

    def test_package_exact_flat_members_and_no_overwrite(self):
        scratch=HERE/'local-output'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path=Path(directory)/'submission.zip'
            payload,token=b'example,data\n1,2\n',b'FAKE-TEST-TOKEN\n'
            write_package(path,payload,token)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(),['C.csv','token.txt'])
                self.assertEqual(archive.read('C.csv'),payload)
                self.assertEqual(archive.read('token.txt'),token)
            with self.assertRaises(FileExistsError):
                write_package(path,payload,token)


if __name__=='__main__':
    unittest.main()
