"""Behavior checks for v1; no credentials or record-level values are logged.

Run from main_round/attack with the pinned environment:
    .venv/Scripts/python.exe -m unittest M.v1.test_v1 -v

Prediction properties use the full generic development protocol, not actual
attack labels. Actual B/B_self/self-challenge are used only to verify the
conditional private-mask deduction and its rejection of inconsistent inputs.
"""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import json
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from attack_workspace import load_runtime, verify_hashes
from M.v1.actual import apply_private_mask, derive_private_mask

ORIGINALS = ROOT.parent / "anonymization/team10/participant_data"


def read_original(name, *, strings=False):
    if strings:
        return pd.read_csv(ORIGINALS / name, dtype=str, keep_default_na=False)
    return pd.read_csv(ORIGINALS / name, float_precision="round_trip")


class PrivateMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verified = verify_hashes()
        if verified["verified_files"] != 94 or verified["warnings"]:
            raise RuntimeError("Unexpected manifest verification result")
        schema = json.loads((ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))
        cls.qi, cls.columns = schema["aia_qi_columns"], schema["submitted_c_columns"]
        cls.b = read_original("B.csv", strings=True)
        cls.bs = read_original("B_self.csv", strings=True)
        cls.challenge = pd.read_csv(ROOT / "targets/aia_challenge_10.csv",
                                    dtype=str, keep_default_na=False)
        cls.mask, cls.report = derive_private_mask(cls.b, cls.bs, cls.challenge,
                                                   cls.qi, cls.columns)

    def derive(self, b=None, bs=None, challenge=None):
        return derive_private_mask(self.b if b is None else b,
                                   self.bs if bs is None else bs,
                                   self.challenge if challenge is None else challenge,
                                   self.qi, self.columns)

    def test_actual_evidence_and_inputs_are_unchanged(self):
        before = [frame.copy(deep=True) for frame in (self.b, self.bs, self.challenge)]
        mask, report = self.derive()
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, (1049,))
        self.assertEqual(int(mask.sum()), 163)
        for name, expected in [("exact_private_rows", 163), ("remaining_unknown_rows", 886),
                               ("private_true_rare", 13), ("private_onset", 16),
                               ("private_rare_onset", 4)]:
            self.assertEqual(report[name], expected)
        self.assertTrue(all(a.equals(b) for a, b in zip(before, (self.b, self.bs, self.challenge))),
                        "Private-mask derivation mutated an input")

    def test_mask_tracks_b_order_and_accepts_independent_b_self_order(self):
        order = np.random.default_rng(681).permutation(len(self.b))
        b = self.b.iloc[order].reset_index(drop=True)
        bs = self.bs.sample(frac=1, random_state=682).reset_index(drop=True)
        challenge = self.challenge.sample(frac=1, random_state=683).reset_index(drop=True)
        mask, report = self.derive(b=b, bs=bs, challenge=challenge)
        self.assertTrue(np.array_equal(mask, self.mask[order]), "B order changed the deduction")
        self.assertEqual(report, self.report)

    def test_mask_does_not_infer_roles_from_record_ids(self):
        b, bs = self.b.copy(deep=True), self.bs.copy(deep=True)
        new_ids = np.random.default_rng(684).permutation(len(b)) + 100000
        mapping = dict(zip(b.record_id, map(str, new_ids)))
        b["record_id"] = b.record_id.map(mapping)
        bs["record_id"] = bs.record_id.map(mapping)
        mask, report = self.derive(b=b, bs=bs)
        self.assertTrue(np.array_equal(mask, self.mask), "ID relabeling changed private labels")
        self.assertEqual(report, self.report)

    def test_sub_float_precision_query_change_is_rejected(self):
        challenge = self.challenge.copy(deep=True)
        b_row = self.b.iloc[int(np.flatnonzero(self.mask)[0])]
        def qkey(row):
            return tuple(row[c] if c == "prefecture" else Decimal(row[c]) for c in self.qi)
        wanted = qkey(b_row)
        index = next(i for i, row in challenge.iterrows() if qkey(row) == wanted)
        column = next(c for c in self.qi if c != "prefecture")
        with localcontext() as context:
            context.prec = 80
            challenge.at[index, column] = str(Decimal(challenge.at[index, column]) + Decimal("1e-50"))
        with self.assertRaises(ValueError):
            self.derive(challenge=challenge)

    def test_duplicate_challenge_is_rejected(self):
        challenge = self.challenge.copy(deep=True)
        challenge.loc[challenge.index[1], self.qi] = challenge.loc[challenge.index[0], self.qi]
        with self.assertRaises(ValueError):
            self.derive(challenge=challenge)

    def test_ambiguous_b_qi_is_rejected_even_when_b_self_agrees(self):
        b, bs = self.b.copy(deep=True), self.bs.copy(deep=True)
        b.loc[b.index[1], self.qi] = b.loc[b.index[0], self.qi]
        destination = bs.index[bs.record_id.eq(b.iloc[1].record_id)][0]
        bs.loc[destination, self.qi] = b.loc[b.index[1], self.qi]
        with self.assertRaises(ValueError):
            self.derive(b=b, bs=bs)

    def test_b_self_public_inconsistency_is_rejected(self):
        bs = self.bs.copy(deep=True)
        bs.at[bs.index[0], "age"] = str(Decimal(bs.iloc[0].age) + 1)
        with self.assertRaises(ValueError):
            self.derive(bs=bs)

    def test_invalid_dimensions_and_record_id_sets_are_rejected(self):
        with self.assertRaises(ValueError):
            self.derive(challenge=self.challenge.iloc[:-1])
        bs = self.bs.copy(deep=True)
        bs.at[bs.index[0], "record_id"] = "99999999"
        with self.assertRaises(ValueError):
            self.derive(bs=bs)
        b = self.b.copy(deep=True)
        b.at[b.index[1], "record_id"] = b.iloc[0].record_id
        with self.assertRaises(ValueError):
            self.derive(b=b)

    def test_quota_violation_is_rejected_even_if_total_rare_is_preserved(self):
        bs = self.bs.copy(deep=True)
        mask_by_id = dict(zip(self.b.record_id, self.mask))
        private = bs.record_id.map(mask_by_id).to_numpy(bool)
        rare = bs.is_rare.to_numpy(int)
        lose = int(np.flatnonzero(private & (rare == 1))[0])
        gain = int(np.flatnonzero(~private & (rare == 0))[0])
        bs.at[bs.index[lose], "is_rare"] = "0"
        bs.at[bs.index[gain], "is_rare"] = "1"
        with self.assertRaises(ValueError):
            self.derive(bs=bs)

    def test_mask_application_preserves_unknown_rank_and_inputs(self):
        scores = np.linspace(0, 1, len(self.b))
        scores[~self.mask] = np.linspace(0, 1, int((~self.mask).sum()))
        before, mask_before = scores.copy(), self.mask.copy()
        result = apply_private_mask(scores, self.mask)
        self.assertTrue(np.array_equal(scores, before), "Score input was mutated")
        self.assertTrue(np.array_equal(self.mask, mask_before), "Mask input was mutated")
        self.assertTrue(np.all(result[self.mask] == 0), "Known negatives are not bottom-ranked")
        self.assertTrue(np.all(result[~self.mask] > 0), "Unknown rows tie with known negatives")
        self.assertTrue(np.all(np.diff(result[~self.mask]) > 0), "Unknown ordering changed")
        self.assertTrue(np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)))

    def test_mask_application_rejects_malformed_predictions(self):
        examples = [(np.array([np.nan, .5]), np.array([True, False])),
                    (np.array([np.inf, .5]), np.array([True, False])),
                    (np.array([-.1, .5]), np.array([True, False])),
                    (np.array([1.1, .5]), np.array([True, False])),
                    (np.array([.1, .5]), np.array([True])),
                    (np.array([[.1, .5]]), np.array([[True, False]]))]
        for scores, mask in examples:
            with self.subTest(kind="invalid score or shape"):
                with self.assertRaises(ValueError):
                    apply_private_mask(scores, mask)


class PublicPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from local_eval.data import build_case, make_release
        from M.v1.model import predict
        verified = verify_hashes()
        if verified["verified_files"] != 94 or verified["warnings"]:
            raise RuntimeError("Unexpected manifest verification result")
        _, _, _, cls.cfg, _ = load_runtime()
        schema = json.loads((ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))
        seed = 2026093000
        case = build_case(read_original("A_bg.csv"), read_original("B.csv"), schema, "dev", seed)
        cls.predict = staticmethod(predict)
        cls.inputs, cls.baseline = {}, {}
        for style in ("copy", "jitter", "rounded"):
            public = {"c": make_release(case.target, case.background, style, seed, schema),
                      "background": case.background.copy(deep=True),
                      "candidates": case.candidates.copy(deep=True),
                      "challenge": case.challenge.copy(deep=True)}
            cls.inputs[style] = public
            cls.baseline[style] = cls.predict(**copy.deepcopy(public), cfg=copy.deepcopy(cls.cfg),
                                              variant="full_reference_aia")

    def predict_public(self, public):
        return self.predict(**public, cfg=self.cfg, variant="full_reference_aia")

    def assert_predictions_equal(self, actual, expected, *, mia_order=None, aia_order=None):
        for name, order in (("mia", mia_order), ("aia", aia_order)):
            reference = np.asarray(expected[name])
            if order is not None:
                reference = reference[order]
            self.assertTrue(np.array_equal(np.asarray(actual[name]), reference),
                            f"{name} predictions changed; row-level values suppressed")

    def test_reproducible_finite_bounded_and_nonmutating(self):
        for style, public in self.inputs.items():
            with self.subTest(style=style):
                frames_before = {key: value.copy(deep=True) for key, value in public.items()}
                cfg_before = pickle.dumps(self.cfg)
                result = self.predict_public(public)
                self.assert_predictions_equal(result, self.baseline[style])
                for name, expected_length in (("mia", 1049), ("aia", 326)):
                    values = np.asarray(result[name])
                    self.assertEqual(values.shape, (expected_length,))
                    self.assertTrue(np.isfinite(values).all(), f"Nonfinite {name} prediction")
                    if name == "mia":
                        self.assertTrue(np.all((values >= 0) & (values <= 1)))
                self.assertTrue(all(public[key].equals(before) for key, before in frames_before.items()),
                                "Prediction mutated a public input")
                self.assertTrue(pickle.dumps(self.cfg) == cfg_before, "Prediction mutated cfg")

    def test_candidate_and_challenge_permutations_preserve_row_alignment(self):
        rng = np.random.default_rng(711)
        mia_order, aia_order = rng.permutation(1049), rng.permutation(326)
        for style, public in self.inputs.items():
            with self.subTest(style=style):
                changed = copy.deepcopy(public)
                changed["candidates"] = changed["candidates"].iloc[mia_order].reset_index(drop=True)
                changed["challenge"] = changed["challenge"].iloc[aia_order].reset_index(drop=True)
                result = self.predict_public(changed)
                self.assert_predictions_equal(result, self.baseline[style],
                                              mia_order=mia_order, aia_order=aia_order)

    def test_release_and_background_permutations_do_not_change_predictions(self):
        for style, public in self.inputs.items():
            with self.subTest(style=style):
                changed = copy.deepcopy(public)
                changed["c"] = changed["c"].sample(frac=1, random_state=712).reset_index(drop=True)
                changed["background"] = changed["background"].sample(frac=1, random_state=713).reset_index(drop=True)
                self.assert_predictions_equal(self.predict_public(changed), self.baseline[style])

    def test_record_id_and_challenge_id_relabeling_does_not_change_predictions(self):
        rng = np.random.default_rng(714)
        for style, public in self.inputs.items():
            with self.subTest(style=style):
                changed = copy.deepcopy(public)
                changed["candidates"]["record_id"] = rng.permutation(1049) + 100000
                changed["challenge"]["challenge_row_id"] = rng.permutation(326) + 200000
                self.assert_predictions_equal(self.predict_public(changed), self.baseline[style])


if __name__ == "__main__":
    unittest.main()
