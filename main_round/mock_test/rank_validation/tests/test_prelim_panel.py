"""Contract tests for preliminary adapters and runtime separation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from rank_validation import prelim_panel as panel

class PreliminaryPanelTests(unittest.TestCase):
    def test_output_must_remain_under_ignored_mock_output(self):
        self.assertEqual(panel.output_path(panel.OUTPUT_ROOT/'new-panel'),(panel.OUTPUT_ROOT/'new-panel').resolve())
        for path in (panel.OUTPUT_ROOT, panel.ROOT/'main_round/attack/Attack_R/v1', panel.OUTPUT_ROOT/'..'/'outside'):
            with self.assertRaises(ValueError):panel.output_path(path)

    def test_predict_and_score_launch_isolated_children(self):
        with patch.object(panel.subprocess,'run') as run:
            panel._launch('_predict',panel.OUTPUT_ROOT/'test',group='shared')
            command=run.call_args.args[0]
            self.assertIn('-I',command);self.assertIn('-B',command)
            self.assertIn('_predict',command)
            self.assertEqual(run.call_args.kwargs['env']['OMP_NUM_THREADS'],'1')

    def test_main_runtime_refuses_already_loaded_official_module(self):
        with patch.dict(panel.sys.modules,{'pwscup2026':object()}):
            with self.assertRaises(RuntimeError):panel.initialize_main()

    def test_shared_rv2_rank_path_equals_frozen_finishing_with_ties(self):
        mod=panel.module_from_file('_test_prelim_rv2',panel.ATTACK/'Attack_R/v2/model.py')
        raw=np.array([0.2,0.7,0.7,0.0,1.,1.,0.4])
        private=np.array([False,True,False,False,False,True,False])
        shared=mod.ranks(mod.ranks(raw));shared[private]=0
        official=mod.finish_mia(mod.ranks(raw)[:,None],private)[:,0]
        np.testing.assert_array_equal(shared,official)

    def test_undefined_legacy_diagnostics_remain_null(self):
        value={"undefined": float("nan"), "nested": [np.float64(float("inf")), np.int64(4)], "array": np.array([1.,np.nan])}
        cleaned=panel.clean_json(value)
        self.assertIsNone(cleaned["undefined"])
        self.assertEqual(cleaned["nested"],[None,4])
        self.assertEqual(cleaned["array"],[1.,None])

    def test_frames_require_complete_targets_and_finite_values(self):
        ids=np.arange(1049);q=pd.DataFrame({'challenge_row_id':np.arange(326)})
        tables={'candidates':pd.DataFrame({'record_id':ids}),'targets':{1:{'challenge':q}}}
        m=pd.DataFrame({'record_id':ids,'1':np.zeros(1049)})
        a=pd.DataFrame({'target_k':1,'challenge_row_id':np.arange(326),'time_hat':np.ones(326)})
        panel._validate_frames(m,a,tables)
        a.loc[0,'time_hat']=np.nan
        with self.assertRaises(ValueError):panel._validate_frames(m,a,tables)
        a.loc[0,'time_hat']=1;a.loc[0,'challenge_row_id']=1
        with self.assertRaises(ValueError):panel._validate_frames(m,a,tables)

if __name__=='__main__':unittest.main()
