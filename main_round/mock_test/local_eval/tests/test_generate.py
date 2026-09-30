"""Portable shadow-case export keeps public input and local truth separate."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import attack_workspace as workspace
from local_eval.data import FEATURES, QI
from local_eval.generate import export_case, export_suite


class GenerateCaseTests(unittest.TestCase):
    def test_same_seed_has_same_files_and_separate_truth(self):
        workspace.MOCK_OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='shadow-export-test-', dir=workspace.MOCK_OUTPUT) as tmp:
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
                        output=workspace.MOCK_ROOT / 'accidental-public-output')

    def test_suite_has_distinct_reproducible_cases_and_balanced_rare_members(self):
        import json
        workspace.MOCK_OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='shadow-suite-test-', dir=workspace.MOCK_OUTPUT) as tmp:
            root = Path(tmp)
            first = export_suite(split='dev', seed=20260927, count=3,
                                 styles=('rounded',), output=root / 'first')
            second = export_suite(split='dev', seed=20260927, count=3,
                                  styles=('rounded',), output=root / 'second')
            self.assertEqual(first, second)
            cases = [json.loads((root / 'first' / entry['path'] / 'manifest.json').read_text())
                     for entry in first['cases']]
            self.assertEqual({c['metadata']['mia_rare_members'] for c in cases}, {0, 1, 2})
            self.assertEqual(len({c['metadata']['role_assignment_sha256'] for c in cases}), 3)
            with self.assertRaises(ValueError):
                export_suite(split='dev', seed=20260927, count=3,
                             styles=('rounded',), output=root / 'first')


if __name__ == '__main__':
    unittest.main()
