"""Prepare public single-target bundles, execute trusted attacks, score CSVs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

from .common import (ROOT, HERE, MOCK_ROOT, OUTPUT_ROOT, cfg, code_hashes,
                     ensure_new_output, file_hashes, public_tables, read_csv,
                     read_json, require, sha256, verify_files, within, write_json)


def load_suite(path):
    path = Path(path).resolve()
    manifest = read_json(path / 'manifest.json')
    require(manifest.get('version') == 1, 'Unsupported suite version')
    verify_files(path, manifest['files'])
    for release in manifest['releases']:
        require(sha256(within(ROOT, release['path'])) == release['sha256'],
                f"Original C changed: {release['name']}")
    return path, manifest


def find_case(suite, manifest, case_id):
    selected = [c for c in manifest['cases'] if c['id'] == case_id]
    require(len(selected) == 1, f'Unknown case: {case_id}')
    return within(suite, selected[0]['path'])


def _fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expected_bundle_files(case, release):
    files = {p.name: sha256(p) for p in sorted((case / 'public').iterdir()) if p.is_file()}
    files['C.csv'] = release['sha256']
    return files


def export_bundle(suite, case_id, model, output):
    suite, manifest = load_suite(suite)
    case = find_case(suite, manifest, case_id)
    releases = [r for r in manifest['releases'] if r['name'] == model]
    require(len(releases) == 1, f'Unknown model: {model}')
    release = releases[0]
    output = ensure_new_output(output)
    for source in (case / 'public').iterdir():
        if source.is_file():
            shutil.copyfile(source, output / source.name)
    shutil.copyfile(within(ROOT, release['path']), output / 'C.csv')
    files = expected_bundle_files(case, release)
    verify_files(output, files)
    meta = {'version': 1, 'target_id': 0, 'files': files, 'input_fingerprint': _fingerprint(files),
            'note': 'Public one-target mock inputs. No labels or source row IDs.'}
    write_json(output / 'bundle.json', meta)
    return output


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    meta = read_json(bundle / 'bundle.json')
    require(meta.get('version') == 1 and meta.get('target_id') == 0, 'Invalid public bundle')
    expected = {'B.csv', 'A_bg.csv', 'C.csv', 'aia_challenge_0.csv', 'schema.json', 'cfg.json'}
    require(set(meta['files']) == expected, 'Unexpected public bundle files')
    require(_fingerprint(meta['files']) == meta.get('input_fingerprint'), 'Invalid bundle fingerprint')
    verify_files(bundle, meta['files'])
    return bundle, meta


def attack_bundle(bundle, output, *, plugin=None, method='official', timeout=600):
    bundle, meta = verify_bundle(bundle)
    output = ensure_new_output(output)
    plugin = Path(plugin).resolve() if plugin else None
    if plugin:
        require(plugin.is_file() and plugin.suffix == '.py', 'Plugin must be a Python source file')
    plugin_hash = sha256(plugin) if plugin else None
    # UUID working folders provide no model/version/seed names to the child.
    work = OUTPUT_ROOT / 'workers' / uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    for name in meta['files']:
        shutil.copyfile(bundle / name, work / name)
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(MOCK_ROOT), 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'})
    command = [sys.executable, '-B', '-m', 'mock_TS.worker', '--method', method]
    if plugin:
        command += ['--plugin', str(plugin)]
    try:
        with (output / 'attack.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        require(result.returncode == 0, f'Attack failed; see {output / "attack.log"}')
        verify_files(work, meta['files'])
        if plugin:
            require(sha256(plugin) == plugin_hash, 'Plugin changed during prediction')
        for name in ('F_mia.csv', 'F_aia.csv', 'worker.json'):
            require((work / 'predictions' / name).is_file(), f'Missing attack output: {name}')
            shutil.copyfile(work / 'predictions' / name, output / name)
        receipt = {'version': 1, 'input_fingerprint': meta['input_fingerprint'],
                   'prediction_sha256': {n: sha256(output / n) for n in ('F_mia.csv', 'F_aia.csv')},
                   'plugin_file': str(plugin) if plugin else None, 'plugin_sha256': plugin_hash,
                   'method': 'plugin' if plugin else method,
                   'isolation': 'Fresh process and public-only arguments; trusted Python, not an OS sandbox',
                   'additional_dependency_hashes': 'External fitted assets must be recorded by the plugin author',
                   'code_sha256': code_hashes()}
        write_json(output / 'prediction_manifest.json', receipt)
    finally:
        # Only our UUID subtree is disposable, never an existing source or output.
        require(work.resolve().parent == (OUTPUT_ROOT / 'workers').resolve(), 'Unsafe worker cleanup')
        shutil.rmtree(work)
    verify_files(bundle, meta['files'])
    return output


def score_submission(suite, case_id, model, predictions, output, *, floor=None, floor_repetitions=1000):
    from .scoring import estimate_floor, score_predictions, validate_predictions
    from .report import write_report
    suite, manifest = load_suite(suite)
    case = find_case(suite, manifest, case_id)
    selected = [r for r in manifest['releases'] if r['name'] == model]
    require(len(selected) == 1, f'Unknown model: {model}')
    expected = _fingerprint(expected_bundle_files(case, selected[0]))
    predictions = Path(predictions).resolve()
    receipt_path = predictions / 'prediction_manifest.json'
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        require(receipt['input_fingerprint'] == expected, 'Predictions belong to a different case or C')
        verify_files(predictions, receipt['prediction_sha256'])
        provenance = 'Verified runner receipt and input hash'
        method = receipt.get('method', 'external')
    else:
        provenance = 'External CSV: case/C correspondence declared by caller, not independently verifiable'
        method = 'external'
    # Validate before creating expensive floor or claiming a successful run.
    f_mia = read_csv(predictions / 'F_mia.csv')
    f_aia = read_csv(predictions / 'F_aia.csv')
    validate_predictions(f_mia, f_aia, read_csv(case / 'public' / 'B.csv'),
                         read_csv(case / 'public' / 'aia_challenge_0.csv'))
    output = ensure_new_output(output)
    if floor is None:
        case_index = next(i for i, c in enumerate(manifest['cases']) if c['id'] == case_id)
        floor = estimate_floor(case, int(manifest['seed']) + case_index + 1000003, floor_repetitions)
    result = score_predictions(case, f_mia, f_aia, floor, output / 'scoring_truth')
    result.update({'case_id': case_id, 'model': model, 'method': method,
                   'prediction_provenance': provenance, 'C_sha256': selected[0]['sha256'],
                   'input_fingerprint': expected,
                   'prediction_sha256': {n: sha256(predictions / n) for n in ('F_mia.csv', 'F_aia.csv')},
                   'code_sha256': code_hashes()})
    write_json(output / 'result.json', result)
    write_json(output / 'floor.json', floor)
    write_report(output, [result], {'limitations': manifest.get('limitations', []),
                                   'suite_manifest_sha256': sha256(suite / 'manifest.json')})
    verify_files(suite, manifest['files'])
    return result


def benchmark(suite, output, *, plugin=None, method='official', floor_repetitions=1000,
              models=None, cases=None, timeout=600):
    from .scoring import estimate_floor
    from .report import write_report
    suite, manifest = load_suite(suite)
    releases = [r for r in manifest['releases'] if models is None or r['name'] in models]
    cases_selected = [c for c in manifest['cases'] if cases is None or c['id'] in cases]
    require(releases and cases_selected, 'No selected models/cases')
    if models:
        require(set(models) == {r['name'] for r in releases}, 'Unknown requested model')
    if cases:
        require(set(cases) == {c['id'] for c in cases_selected}, 'Unknown requested case')
    output = ensure_new_output(output)
    write_json(output / 'run_started.json', {'suite': str(suite), 'time_utc': datetime.now(timezone.utc).isoformat(),
               'method': method, 'plugin': str(plugin) if plugin else None, 'code_sha256': code_hashes()})
    rows = []
    for case_index, case_item in enumerate(manifest['cases']):
        if case_item not in cases_selected:
            continue
        case_id = case_item['id']
        case = find_case(suite, manifest, case_id)
        print(f'{case_id}: estimating shared mock phi ({floor_repetitions} repetitions)', flush=True)
        floor = estimate_floor(case, int(manifest['seed']) + case_index + 1000003, floor_repetitions)
        write_json(output / case_id / 'floor.json', floor)
        for release in releases:
            model = release['name']
            base = output / case_id / model
            print(f'{case_id}: independently attacking {model}', flush=True)
            bundle = export_bundle(suite, case_id, model, base / 'input')
            predictions = attack_bundle(bundle, base / 'predictions', plugin=plugin, method=method, timeout=timeout)
            row = score_submission(suite, case_id, model, predictions, base / 'evaluation', floor=floor)
            row['method'] = Path(plugin).stem if plugin else method
            rows.append(row)
            # C-free result is computed once per case. No target predictions are reused.
            if release is releases[0]:
                diagnostic = output / case_id / 'background_only'
                bp = attack_bundle(bundle, diagnostic / 'predictions', method='background_only', timeout=timeout)
                baseline = score_submission(suite, case_id, model, bp, diagnostic / 'evaluation', floor=floor)
                baseline.update(model='Cなし', method='background_only')
                rows.append(baseline)
        write_report(output, rows, {'limitations': manifest.get('limitations', []),
                     'suite_manifest_sha256': sha256(suite / 'manifest.json'), 'partial': True})
    verify_files(suite, manifest['files'])
    write_report(output, rows, {'limitations': manifest.get('limitations', []),
                 'suite_manifest_sha256': sha256(suite / 'manifest.json'), 'partial': False,
                 'model_count': len(releases), 'case_count': len(cases_selected),
                 'floor_repetitions': floor_repetitions, 'code_sha256': code_hashes()})
    return output / 'detailed_results.html'
