from fractions import Fraction as F
import unittest
from rank_validation import ordinal_calibration as ordinal

class OrdinalCalibrationTests(unittest.TestCase):
    def test_strict_interval_and_nearest_interior_rule(self):
        interval=ordinal.feasible_interval([{'intercept':F(-3,4),'slope':1}])
        self.assertEqual(interval['lower'],F(3,4)); self.assertTrue(interval['lower_open'])
        self.assertEqual(ordinal.select_alpha(interval),F(13,16))
        low=ordinal.feasible_interval([{'intercept':F(1,4),'slope':-1}])
        self.assertEqual(ordinal.select_alpha(low),F(3,16))
        central=ordinal.feasible_interval([{'intercept':F(-1,4),'slope':1},{'intercept':F(3,4),'slope':-1}])
        self.assertEqual(ordinal.select_alpha(central),F(1,2))

    def test_half_on_strict_boundary_is_not_chosen(self):
        interval=ordinal.feasible_interval([{'intercept':F(-1,2),'slope':1}])
        self.assertEqual(ordinal.select_alpha(interval),F(5,8))

    def test_impossible_reversed_constant_tie_and_conflicting_constraints(self):
        for constraints in ([{'intercept':0,'slope':0}],[{'intercept':-1,'slope':0}],
            [{'intercept':F(-3,4),'slope':1},{'intercept':F(1,4),'slope':-1}],
            [{'intercept':-1,'slope':1}],[{'intercept':0,'slope':-1}]):
            interval=ordinal.feasible_interval(constraints)
            self.assertFalse(interval['feasible']); self.assertIsNone(ordinal.select_alpha(interval))

    def test_fit_exact_three_and_fraction_solution(self):
        rows={'A':{'mia':.8,'aia':0},'B':{'mia':.2,'aia':.6},'C':{'mia':.1,'aia':.1}}
        fit=ordinal.fit_alpha(rows,['A','B','C'])
        self.assertEqual(fit['interval']['lower'],F(1,2))
        self.assertEqual(fit['alpha'],F(5,8))
        self.assertTrue(fit['numerically_supported'])
        with self.assertRaises(ValueError):
            ordinal.fit_alpha(rows|{'D':{'mia':0,'aia':0}},['A','B','C','D'])

    def test_heldout_dev_and_confirmation_values_cannot_change_fitted_alpha(self):
        rows={'A':{'mia':.8,'aia':.1},'B':{'mia':.6,'aia':.2},'C':{'mia':.3,'aia':.2},'D':{'mia':.1,'aia':.1}}
        first=ordinal.diagnose(rows,[rows,rows],['A','B','C','D'])
        changed={name:dict(row) for name,row in rows.items()}; changed['D']={'mia':1,'aia':1}
        second=ordinal.diagnose(changed,[changed,changed],['A','B','C','D'])
        self.assertEqual(first['folds'][3]['fit'],second['folds'][3]['fit'])
        self.assertTrue(first['folds'][3]['passes_both_confirmations'])
        self.assertFalse(second['folds'][3]['passes_both_confirmations'])
        third=ordinal.diagnose(rows,[changed,changed],['A','B','C','D'])
        for a,b in zip(first['folds'],third['folds']):
            self.assertEqual(a['fit'],b['fit'])

    def test_ties_never_count_and_both_confirmation_cases_required(self):
        rows={name:{'mia':i/10,'aia':i/10} for name,i in [('A',4),('B',3),('C',2),('D',1)]}
        ties={name:{'mia':.1,'aia':.1} for name in rows}
        result=ordinal.diagnose(rows,[rows,ties],list(rows))
        self.assertEqual(result['passed_folds'],0)
        for fold in result['folds']:
            self.assertEqual(fold['confirmation_checks'][1]['calibrated']['correct'],0)
            self.assertTrue(all(p['tie'] for p in fold['confirmation_checks'][1]['calibrated']['pairs']))
        with self.assertRaises(ValueError):
            ordinal.diagnose(rows,[rows],list(rows))

    def test_impossible_folds_are_preserved_as_failures(self):
        rows={name:{'mia':.1,'aia':.1} for name in ['A','B','C','D']}
        result=ordinal.diagnose(rows,[rows,rows],list(rows))
        self.assertEqual(len(result['folds']),4)
        self.assertEqual(result['confirmation_pair_count'],24)
        self.assertEqual(result['confirmation_pair_hits'],0)
        self.assertTrue(all(f['fit']['status']=='impossible' for f in result['folds']))
        self.assertFalse(result['all_four_folds_pass'])

    def test_nonfinite_invalid_range_and_order_rejected(self):
        rows={'A':{'mia':.8,'aia':0},'B':{'mia':.2,'aia':.6},'C':{'mia':.1,'aia':.1}}
        for value in [float('nan'),float('inf'),True,1.1,-.1]:
            bad={key:dict(row) for key,row in rows.items()}; bad['A']['mia']=value
            with self.assertRaises(ValueError): ordinal.fit_alpha(bad,list(rows))
        with self.assertRaises(ValueError): ordinal.fit_alpha(rows,['A','B','B'])

    def test_tiny_strict_fit_is_not_numerically_supported(self):
        rows={'A':{'mia':.5000000000002,'aia':.5},'B':{'mia':.5000000000001,'aia':.5},'C':{'mia':.5,'aia':.5}}
        fit=ordinal.fit_alpha(rows,list(rows))
        self.assertTrue(fit['interval']['feasible'])
        self.assertFalse(fit['numerically_supported'])
        self.assertEqual(fit['status'],'insufficient_numerical_margin')

    def test_json_preserves_exact_coefficients(self):
        self.assertEqual(ordinal._jsonable({'alpha':F(5,8)})['alpha'],{'decimal':.625,'exact':'5/8'})

if __name__=='__main__': unittest.main()
