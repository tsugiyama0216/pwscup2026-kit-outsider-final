from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from attack_phase_v8.event_additions import add_trusted_events, strict_aia_frame


class EventAdditionTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({"target_k":[1,3,3,3,4],
            "challenge_row_id":[0,2,0,1,0], "time_hat":[1.,11.,9.5,11.,11.]})
        self.releases = {t:pd.DataFrame({"unused":[0]}) for t in ("1","3","4")}
        self.challenges = {"1":pd.DataFrame({"challenge_row_id":[0]}),
            "3":pd.DataFrame({"challenge_row_id":[2,0,1]}),
            "4":pd.DataFrame({"challenge_row_id":[0]})}

    def run_policy(self, **kwargs):
        with patch("attack_phase_v8.event_additions.filter_linkable_releases", return_value=(self.releases,{})) as filtered:
            with patch("attack_phase_v8.event_additions.copula_event_links", return_value=(np.array([2.,4.,6.]),np.array([True,True,False]),{})):
                result = add_trusted_events(self.base,self.releases,self.challenges,target_ids=["3"],**kwargs)
            self.assertEqual(set(filtered.call_args.args[0]),set(self.releases))
        return result

    def test_only_selected_abstention_changes_with_key_alignment(self):
        result, details = self.run_policy()
        self.assertEqual(result.time_hat.tolist(),[1.,11.,9.5,4.,11.])
        self.assertEqual(details["targets"]["3"]["added_challenge_row_ids"],[1])
        self.assertEqual(details["changed_rows"],1)
        self.assertEqual(self.base.time_hat.tolist(),[1.,11.,9.5,11.,11.])

    def test_near_abstention_not_treated_as_sentinel(self):
        self.base.loc[3,"time_hat"]=10.999999
        result, details=self.run_policy()
        self.assertEqual(result.loc[3,"time_hat"],10.999999)
        self.assertEqual(details["changed_rows"],0)

    def test_rejects_bad_keys_before_truncation(self):
        for value in (.5,np.inf,np.nan,-1,2**53):
            frame=self.base.copy()
            frame["challenge_row_id"]=frame.challenge_row_id.astype(float)
            frame.loc[0,"challenge_row_id"]=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_aia_frame(frame)

    def test_rejects_normalized_duplicates_and_invalid_times(self):
        with self.assertRaises(ValueError):
            strict_aia_frame(pd.concat([self.base,self.base.iloc[[0]]]))
        for value in (np.inf,np.nan,-1):
            frame=self.base.copy()
            frame.loc[0,"time_hat"]=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_aia_frame(frame)

    def test_protects_ceiling_targets_and_rejects_bad_target_sets(self):
        for selected in (["1"],["2"],["3","3"],["03"],[],["99"]):
            with self.subTest(targets=selected),self.assertRaises(ValueError):
                add_trusted_events(self.base,self.releases,self.challenges,target_ids=selected)

    def test_rejects_challenge_key_mismatch(self):
        self.challenges["4"].loc[0,"challenge_row_id"]=9
        with self.assertRaises(ValueError):
            self.run_policy()

    def test_rejects_invalid_decoder_outputs(self):
        with patch("attack_phase_v8.event_additions.filter_linkable_releases",return_value=(self.releases,{})):
            with patch("attack_phase_v8.event_additions.copula_event_links",return_value=(np.array([1.,np.inf,2.]),np.ones(3,bool),{})):
                with self.assertRaises(ValueError):
                    add_trusted_events(self.base,self.releases,self.challenges,target_ids=["3"])

if __name__=="__main__":
    unittest.main()
