import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import model


class Pl1ContractTests(unittest.TestCase):
    def inputs(self):
        return dict(candidates=pd.DataFrame({'record_id':range(1049)}), self_labels=pd.DataFrame(),
            background=pd.DataFrame(),self_challenge=pd.DataFrame(),schema={},cfg={},
            targets={k:{'release':pd.DataFrame(), 'challenge':pd.DataFrame({'challenge_row_id':range(326)})}
                     for k in model.TARGETS})

    def test_reject_self_or_missing_target(self):
        args=self.inputs(); args['targets'][10]=args['targets'].pop(0)
        with self.assertRaises(ValueError): model.predict_world(**args)

    def test_reject_duplicate_candidate_ids(self):
        args=self.inputs(); args['candidates'].loc[1,'record_id']=0
        with self.assertRaises(ValueError): model.predict_world(**args)

    def test_reject_duplicate_query_ids(self):
        args=self.inputs(); args['targets'][0]['challenge'].loc[1,'challenge_row_id']=0
        with self.assertRaises(ValueError): model.predict_world(**args)

    def test_frozen_choice_and_ids(self):
        args=self.inputs()
        mia=pd.DataFrame({'record_id':range(1049),**{k:np.linspace(0,1,1049) for k in sorted(model.TARGETS)}})
        aia=pd.concat([pd.DataFrame({'target_k':k,'challenge_row_id':range(326),'time_hat':10.})
                       for k in sorted(model.TARGETS)],ignore_index=True)
        with patch.object(model.methods,'load_legacy',return_value={}), \
             patch.object(model.methods,'build_mia',return_value=(mia,{},{})), \
             patch.object(model.methods,'build_aia',return_value=({'structured':aia},{})), \
             patch.object(model.methods,'source_hashes',return_value={}):
            got=model.predict_world(**args)
        self.assertEqual(list(got['F_mia']),['record_id',*map(str,sorted(model.TARGETS))])
        pd.testing.assert_frame_equal(got['F_aia'],aia)
        self.assertFalse(got['diagnostics']['historical_target_routing_used'])


if __name__=='__main__': unittest.main()
