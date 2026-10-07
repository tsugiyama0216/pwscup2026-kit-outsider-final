"""Verify the two frozen evaluations and archive non-credential evidence."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RV3 = HERE.parent
ROOT = RV3.parents[4]
BASE = ROOT / 'main_round/mock_test/local-output/mock_TS_using_prelim'
RUNS = [BASE / 'Rv3_20261007', BASE / 'Rv3_20261007_repeat']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    if path.name == 'token.txt':
        raise ValueError('Credentials are outside this verification')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def main():
    dependencies = read(RV3 / 'production_dependencies.json')
    preserved = {}
    for name, expected in dependencies['files_sha256'].items():
        actual = sha(ROOT / name)
        require(actual == expected, 'Frozen dependency changed: ' + name)
        preserved[name] = actual
    require(sha(RV3 / 'selection.json') == dependencies['selection_sha256'], 'Selection changed')
    original_submission = read(RV3 / 'submission/verification.json')['file_sha256']
    for name, expected in original_submission.items():
        require(sha(RV3 / 'submission' / name) == expected, 'Main-round submission changed: ' + name)

    results = [read(run / 'evaluation/result.json') for run in RUNS]
    require(results[0]['summary'] == results[1]['summary'], 'Repeated scores differ')
    require(results[0]['by_target'] == results[1]['by_target'], 'Repeated target scores differ')
    require(results[0]['prediction_provenance'] == results[1]['prediction_provenance'], 'Repeated provenance differs')
    metadata = [read(run / 'predictions/plugin_metadata.json') for run in RUNS]
    require(metadata[0] == metadata[1], 'Repeated adapter diagnostics differ')
    require(metadata[0]['diagnostics']['blocked_data_file_reads'] == [], 'Unexpected data-file read')
    require(metadata[0]['diagnostics']['algorithm_or_parameters_changed'] is False, 'Algorithm changed')

    public = RUNS[0] / 'public'
    manifest = read(public / 'bundle_manifest.json')
    targets = manifest['target_ids']
    require(len(targets) == 20 and 24 not in targets, 'Wrong preliminary targets')
    for run in RUNS:
        require(read(run / 'public/bundle_manifest.json') == manifest, 'Repeated public inputs differ')
        require(read(run / 'input_verification.json') == read(RUNS[0] / 'input_verification.json'), 'Bundle verification differs')
        for name, expected in manifest['files'].items():
            require(sha(run / 'public' / name) == expected, 'Public input changed: ' + name)

    prediction_hashes = {}
    for name in ('F_mia.csv', 'F_aia.csv'):
        digest = sha(RUNS[0] / 'predictions' / name)
        for run in RUNS:
            require(sha(run / 'predictions' / name) == digest, 'Repeated prediction bytes differ')
            require(sha(run / 'evaluation' / name) == digest, 'Scored prediction bytes differ')
        require(digest == results[0]['prediction_sha256'][name], 'Recorded prediction hash differs')
        prediction_hashes[name] = digest
    require(sha(HERE / 'adapter.py') == results[0]['prediction_provenance']['plugin_sha256'], 'Adapter changed')

    mia = csv(RUNS[0] / 'predictions/F_mia.csv')
    aia = csv(RUNS[0] / 'predictions/F_aia.csv')
    require(mia.shape == (1049, 21) and aia.shape == (6520, 3), 'Wrong prediction shape')
    require(list(mia.columns) == ['record_id'] + list(map(str, targets)), 'Wrong MIA columns')
    require(sorted(aia.target_k.unique()) == targets, 'Wrong AIA targets')
    candidates = csv(public / 'team24_processing/B.csv')
    require(np.array_equal(mia.record_id, candidates.record_id), 'MIA candidate order changed')
    for k in targets:
        rows = aia[aia.target_k == k]
        challenge = csv(public / f'attack_targets/aia_challenge_{k}.csv')
        require(len(rows) == 326 and np.array_equal(rows.challenge_row_id, challenge.challenge_row_id), 'AIA IDs differ')

    mask_path = ROOT / 'main_round/attack/Attack_M/v1/actual.py'
    spec = importlib.util.spec_from_file_location('_verify_public_mask', mask_path)
    mask_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_module)
    schema = read(public / 'team24_processing/schema.json')
    mask, proof = mask_module.derive_private_mask(
        candidates.astype(str), csv(public / 'team24_processing/B_self.csv').astype(str),
        csv(public / 'attack_targets/aia_challenge_24.csv').astype(str),
        schema['aia_qi_columns'], schema['submitted_c_columns'])
    require(int(mask.sum()) == 163 and np.all(mia.iloc[:, 1:].to_numpy()[mask] == 0), 'Public negative mask differs')

    details = results[0]['by_target']
    summary = results[0]['summary']
    require([r['target_k'] for r in details] == targets, 'Scored target list differs')
    for key, value in [('mia_power', np.mean([r['mia_score'] for r in details])),
                       ('aia_power', np.mean([r['aia_score'] for r in details]))]:
        require(abs(summary[key] - value) < 1e-15, 'Incorrect mean: ' + key)
    require(abs(summary['attack_power'] - summary['mia_power'] - summary['aia_power']) < 1e-15, 'Incorrect total')

    for name in ('result.json', 'by_target.csv', 'report.html'):
        shutil.copyfile(RUNS[0] / 'evaluation' / name, HERE / name)
    for name in ('plugin_metadata.json', 'prediction_provenance.json'):
        shutil.copyfile(RUNS[0] / 'predictions' / name, HERE / name)
    record = {
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'runs': [p.relative_to(ROOT).as_posix() for p in RUNS],
        'evaluation_only_no_tuning': True, 'attacker_id': 24, 'target_ids': targets,
        'summary': summary, 'two_fresh_processes_byte_identical': True,
        'two_official_scoring_runs_identical': True, 'prediction_sha256': prediction_hashes,
        'mia_shape': list(mia.shape), 'aia_shape': list(aia.shape),
        'public_negative_mask': proof, 'input_audit_blocked_reads': [],
        'isolation_limit': 'Trusted source review and extension-based Python open audit, not an OS filesystem sandbox.',
        'frozen_dependency_count': len(preserved), 'frozen_dependency_sha256': preserved,
        'selection_sha256': dependencies['selection_sha256'],
        'main_round_submission_unchanged_sha256': original_submission,
        'zero_score_targets': {k: sum(r[k + '_score'] == 0 for r in details) for k in ('mia', 'aia')},
        'official_main_round_score_measured': False, 'official_submission_performed': False,
        'official_attempts_consumed': 0,
    }
    (HERE / 'verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'summary': summary,
                      'frozen_dependencies': len(preserved), 'prediction_sha256': prediction_hashes}, indent=2))


if __name__ == '__main__':
    main()
