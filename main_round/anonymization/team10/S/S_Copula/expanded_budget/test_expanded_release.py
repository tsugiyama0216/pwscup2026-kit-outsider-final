from pathlib import Path
import tempfile,unittest,json
from expanded_release import reserve_and_measure,ReleaseBlocked


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'release'
        self.ledger=Path(self.temp.name)/'ledger'
        self.freeze={'additional_epsilon_exact':'100','composed_epsilon_exact':'114'}
    def run_release(self,producer,freeze=None):
        return reserve_and_measure(self.path,freeze or self.freeze,producer,ledger_dir=self.ledger)
    def test_reserve_before_measure_and_resume_without_query(self):
        calls=[]
        def producer():
            self.assertTrue((self.path/'reservation.json').exists())
            self.assertTrue((self.ledger/'reservation.json').exists())
            calls.append(1); return {'target':'DP'}
        first,reused=self.run_release(producer)
        second,reused2=self.run_release(producer)
        self.assertEqual(first,second); self.assertFalse(reused); self.assertTrue(reused2)
        self.assertEqual(len(calls),1)
    def test_failure_spends_budget_and_blocks_retry(self):
        def failed(): raise RuntimeError('simulated private-stage interruption')
        with self.assertRaises(RuntimeError): self.run_release(failed)
        with self.assertRaises(ReleaseBlocked):
            self.run_release(lambda:self.fail('must not retry'))
    def test_changed_freeze_or_corrupt_saved_model_blocks(self):
        self.run_release(lambda:{'target':'DP'})
        with self.assertRaises(ReleaseBlocked):
            self.run_release(lambda:None,{**self.freeze,'composed_epsilon_exact':'214'})
        (self.path/'model.json').write_text(json.dumps({'target':'altered'}),encoding='utf-8')
        with self.assertRaises(ReleaseBlocked):
            self.run_release(lambda:None)
    def test_removed_output_does_not_restore_budget(self):
        self.run_release(lambda:{'target':'DP'})
        self.path.rename(self.path.with_name('archived_release'))
        with self.assertRaises(ReleaseBlocked):
            self.run_release(lambda:self.fail('must not remeasure after output move'))
    def test_different_output_cannot_reuse_same_ledger(self):
        self.run_release(lambda:{'target':'DP'})
        self.path=self.path.with_name('different_output')
        with self.assertRaises(ReleaseBlocked):
            self.run_release(lambda:self.fail('must not remeasure under new output'))


if __name__=='__main__': unittest.main()
