"""Portable shadow-case export keeps public input and local truth separate."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import attack_workspace as workspace
from local_eval.data import FEATURES, QI
from local_eval.generate import export_case


class GenerateCaseTests(unittest.TestCase):
    def test_same_seed_has_same_files_and_separate_truth(self):
        workspace.TS_OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='shadow-export-test-', dir=workspace.TS_OUTPUT) as tmp:
            first = Path(tmp) / 'first'
            second = Path(tmp) / 'second'
            a = export_case(split='dev', seed=20260927, styles=('rounded',), output=first)
            b = export_case(split='dev', seed=20260927, styles=('rounded',), output=second)
            self.assertEqual(a['files_sha256'], b['files_sha256'])
            self.assertEqual(list(pd.read_csv(first/'public'/'C_rounded.csv').columns), FEATURES)
            self.assertEqual(list(pd.read_csv(first/'public'/'aia_challenge.csv').columns),
                             ['challenge_row_id'] + QI)
            self.assertTrue((first/'private'/'mia_truth.csv').is_file())
            self.assertTrue((first/'private'/'aia_truth.csv').is_file())
            self.assertFalse((first/'public'/'mia_truth.csv').exists())
            with self.assertRaises(ValueError):
                export_case(split='dev', seed=20260927, styles=('rounded',), output=first)

    def test_rejects_output_outside_ignored_workspace(self):
        with self.assertRaises(ValueError):
            export_case(split='dev', seed=20260927, styles=('rounded',),
                        output=workspace.TS_ROOT / 'accidental-public-output')


if __name__ == '__main__':
    unittest.main()
